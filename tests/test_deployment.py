"""Can this actually be deployed?

The runtime image installs `backend/requirements.txt` and nothing else: no
PyTorch, no scikit-learn, no SciPy, no pandas. That is deliberate -- it keeps
the image inside a free-tier instance and means a training-only CVE cannot
reach production -- but it is also fragile in exactly one way: somebody adds a
convenient `import pandas` to a serving module, every test passes locally
because the dev environment has pandas, and the deploy dies on boot.

So this file simulates the deployed image by blocking those imports outright
and driving every endpoint through them.
"""

from __future__ import annotations

import importlib
import importlib.abc
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

#: Present in the dev environment, absent from the runtime image.
NOT_IN_IMAGE = ("torch", "sklearn", "scipy", "pandas", "matplotlib", "seaborn")


class _Blocker(importlib.abc.MetaPathFinder):
    """Pretend the training dependencies are not installed."""

    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in NOT_IN_IMAGE:
            raise ModuleNotFoundError(
                f"No module named '{root}' (blocked: not in the runtime image)")
        return None


@pytest.fixture(scope="module")
def slim_client():
    """A TestClient booted as if only the runtime requirements were installed."""
    blocker = _Blocker()
    saved = {k: v for k, v in sys.modules.items()
             if k.split(".")[0] in NOT_IN_IMAGE}
    for k in saved:
        del sys.modules[k]
    for mod in [m for m in sys.modules if m.startswith("app")]:
        del sys.modules[mod]
    sys.meta_path.insert(0, blocker)
    try:
        from fastapi.testclient import TestClient
        from app.main import app
        with TestClient(app) as c:
            yield c
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)
        for mod in [m for m in sys.modules if m.startswith("app")]:
            del sys.modules[mod]


# ==========================================================================
# the image can actually serve
# ==========================================================================
def test_the_app_boots_without_the_training_dependencies(slim_client):
    r = slim_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_the_trained_model_still_loads_without_torch(slim_client):
    """The whole point of exporting weights to .npz and replaying in NumPy."""
    m = slim_client.get("/health").json()["model"]
    assert m["fell_back"] is False, (
        f"the served model fell back to {m['loaded']!r} without torch — the "
        f"NumPy serving path is broken, which is the deploy story")


@pytest.mark.parametrize("method,path,body", [
    ("GET", "/api/city", None),
    ("GET", "/api/places", None),
    ("GET", "/api/models", None),
    ("GET", "/api/providers", None),
    ("GET", "/api/lifecycle", None),
    ("GET", "/api/demo", None),
    ("GET", "/api/insights", None),
    ("GET", "/", None),
    ("POST", "/api/compare", {"origin": "College (Shanthinagar)",
                              "destination": "M.G. Road"}),
    ("POST", "/api/book", {"origin": "College (Shanthinagar)",
                           "destination": "M.G. Road",
                           "provider_id": "bike_taxi", "demo": True}),
])
def test_every_public_endpoint_serves_in_the_slim_image(slim_client, method, path, body):
    r = (slim_client.get(path) if method == "GET"
         else slim_client.post(path, json=body))
    assert r.status_code == 200, f"{method} {path} -> {r.status_code}"


def test_enterprise_serves_in_the_slim_image(slim_client):
    r = slim_client.get("/api/enterprise/overview",
                        headers={"X-API-Key": "demo-analyst-key"})
    assert r.status_code == 200


# ==========================================================================
# the deployment configuration is real
# ==========================================================================
def test_runtime_requirements_exclude_training_packages():
    req = (ROOT / "backend" / "requirements.txt").read_text(encoding="utf-8")
    for pkg in ("torch", "scikit-learn", "pandas", "scipy"):
        assert pkg not in req, (
            f"{pkg} crept into the runtime requirements — it belongs in "
            f"requirements-train.txt")


def test_dockerfile_ships_what_the_app_needs():
    df = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    for needed in ("COPY data/", "COPY models/", "COPY backend/"):
        assert needed in df, f"Dockerfile does not {needed}"
    assert "--from=frontend" in df, "the built UI is not copied into the image"
    assert "${PORT:-8000}" in df, "the container ignores Render's $PORT"
    assert "USER" in df, "the container runs as root"


def test_the_container_runs_a_single_worker():
    """Booking sessions and the audit log live in process memory.

    With more than one worker a rider could press TRY AGAIN and hit a process
    that has never heard of their booking. Until those move to shared storage,
    one worker is a correctness requirement, not a performance choice.
    """
    df = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    m = re.search(r"--workers\s+(\d+)", df)
    assert m and m.group(1) == "1", (
        "the image must run exactly one worker while sessions are in-memory")


def test_render_config_is_complete():
    y = (ROOT / "render.yaml").read_text(encoding="utf-8")
    for needed in ("healthCheckPath: /health", "dockerfilePath: ./Dockerfile",
                   "JM_API_KEYS", "runtime: docker"):
        assert needed in y, f"render.yaml is missing {needed!r}"


def test_dockerignore_does_not_exclude_the_data_or_models():
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    patterns = [ln.strip() for ln in ignore if ln.strip() and not ln.startswith("#")]
    for essential in ("data", "models", "data/", "models/", "*.csv", "*.npz"):
        assert essential not in patterns, (
            f".dockerignore excludes {essential!r}, which the image needs")


def test_the_bundled_artefacts_the_image_copies_all_exist():
    for path in ("data/city/bengaluru_south/nodes.csv",
                 "data/city/bengaluru_south/fares.json",
                 "data/mobility/bookings.csv",
                 "models/gat_model.npz",
                 "models/reliability_model.npz"):
        assert (ROOT / path).exists(), f"{path} is missing and the image needs it"


def test_env_example_documents_every_setting_the_code_reads():
    """A setting the code reads but nobody documents is a deploy-time surprise."""
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    src = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (ROOT / "backend" / "app").rglob("*.py"))
    referenced = set(re.findall(r'getenv\(\s*["\'](JM_[A-Z_]+)["\']', src))
    referenced |= set(re.findall(r'_bool\(\s*["\'](JM_[A-Z_]+)["\']', src))
    referenced |= set(re.findall(r'_int\(\s*["\'](JM_[A-Z_]+)["\']', src))
    referenced |= set(re.findall(r'_float\(\s*["\'](JM_[A-Z_]+)["\']', src))
    missing = sorted(v for v in referenced if v not in env)
    assert not missing, f"undocumented environment variables: {missing}"
