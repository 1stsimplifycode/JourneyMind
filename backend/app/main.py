"""JourneyMind application entry point.

One service: the API and the built frontend are served from the same process,
which keeps deployment to a single Render web service and removes cross-origin
configuration from the list of things that can break in production.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.booking import router as booking_router
from .api.mobility import router as mobility_router
from .api.routes import router
from .config import get_settings

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("journeymind")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the graph and load the model at startup, not on the first request."""
    s = get_settings()
    t0 = time.perf_counter()
    try:
        from .services.engine import warm_up
        info = warm_up()
        log.info("ready in %.2fs — %s | %d nodes, %d edges | model: %s | "
                 "%d bookings", time.perf_counter() - t0, info["city"],
                 info["nodes"], info["edges"], info["model"],
                 info.get("bookings", 0))
    except Exception:
        # A failed warm-up must not stop the service from booting: /health will
        # report the degradation and the engine will retry on first request.
        log.exception("warm-up failed; the service will start anyway")
    yield
    log.info("shutting down")


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=(
        "A travel advisor that plans your whole trip — not just one ride.\n\n"
        "Recommends complete multi-modal journeys under a budget and a deadline. "
        "Ride-hailing fares are transparent estimates, never quotes; travel times "
        "are model predictions; the bundled study-area data is labelled as demo data."
    ),
    version=settings.version,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,          # the API is public and carries no credentials
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=600,
)


# --------------------------------------------------------------------------
# a small in-process rate limit, so one client cannot monopolise a small dyno
# --------------------------------------------------------------------------
_hits: dict[str, deque] = {}
_LIMITED_PREFIXES = ("/api/recommend", "/api/demo")


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    s = get_settings()
    if s.rate_limit_per_min > 0 and request.url.path.startswith(_LIMITED_PREFIXES):
        key = request.client.host if request.client else "anonymous"
        now = time.monotonic()
        window = _hits.setdefault(key, deque())
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= s.rate_limit_per_min:
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests. Try again in a minute.",
                         "code": "rate_limited"},
                headers={"Retry-After": "60"},
            )
        window.append(now)
        if len(_hits) > 4096:                 # bound the bookkeeping
            for k in [k for k, v in _hits.items() if not v or now - v[-1] > 300][:2048]:
                _hits.pop(k, None)
    return await call_next(request)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    """Never leak a stack trace to a client."""
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Something went wrong handling that request.",
                 "code": "internal_error"},
    )


app.include_router(router)
app.include_router(mobility_router)
app.include_router(booking_router)


# --------------------------------------------------------------------------
# built frontend, served from the same process
# --------------------------------------------------------------------------
def _mount_frontend() -> None:
    s = get_settings()
    if not s.serve_frontend:
        return
    static_dir: Path = s.static_dir
    index = static_dir / "index.html"
    if not index.exists():
        log.warning("no built frontend at %s — API only. Run `npm run build` "
                    "in frontend/ to produce it.", static_dir)
        return

    assets = static_dir / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/", include_in_schema=False)
    def index_page():
        return FileResponse(str(index))

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        """Serve a real file when one exists, otherwise the SPA shell.

        `resolve()` plus the containment check is what stops `../` in a URL
        from reaching anything outside the build directory.
        """
        if path.startswith("api/"):
            return JSONResponse(status_code=404,
                                content={"error": "Not found", "code": "not_found"})
        candidate = (static_dir / path).resolve()
        try:
            candidate.relative_to(static_dir.resolve())
        except ValueError:
            return FileResponse(str(index))
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(index))

    log.info("serving frontend from %s", static_dir)


_mount_frontend()


if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run("app.main:app", host=s.host, port=s.port, reload=False)
