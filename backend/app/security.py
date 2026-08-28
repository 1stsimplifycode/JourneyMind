"""Authentication, authorisation and the audit trail.

Three controls, each of which does something real:

    API keys      who is calling
    roles         what they may see
    audit log     what was decided, on what evidence, and by which model

WHY THE ENTERPRISE ENDPOINTS ARE GATED AND THE RIDER ONES ARE NOT
-----------------------------------------------------------------
A journey quote is about the caller's own trip and reveals nothing about anyone
else. An enterprise aggregate is about a *population* -- which campus travels
when, which teams spend what -- and even at cohort level that is commercially
sensitive. So the gate sits exactly where the sensitivity changes, rather than
being sprayed across every route to look thorough.

THE FAIL-CLOSED RULE
--------------------
With no keys configured, a deployed instance serves NO enterprise data. It does
not fall back to open access, because a security control whose failure mode is
"allow everything" is not a control. The single exception is local demo mode,
which enables one clearly-named demo key and announces it loudly in /health and
in every response it authorises -- so a demo can never be mistaken for a
configured deployment.

WHAT THIS IS NOT
----------------
This is API-key auth suitable for a pilot behind a gateway. It is not SSO, not
OAuth, not multi-tenant isolation, and it does not encrypt data at rest. Those
are named in the limitations rather than implied by a middleware.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path

from fastapi import Header, HTTPException, Request

log = logging.getLogger("journeymind.security")

DEMO_KEY = "demo-analyst-key"


class Role(IntEnum):
    """Ordered, so a check is `>=` rather than a set membership puzzle."""

    RIDER = 10
    ANALYST = 20
    ADMIN = 30


ROLE_NAMES = {r.name.lower(): r for r in Role}


@dataclass(frozen=True)
class Principal:
    key_id: str
    role: Role
    is_demo: bool = False

    def as_dict(self) -> dict:
        return {"key_id": self.key_id, "role": self.role.name.lower(),
                "demo": self.is_demo}


def _parse_keys(raw: str) -> dict[str, Principal]:
    """`JM_API_KEYS="key1:analyst,key2:admin"` -> a lookup.

    Keys are held as SHA-256 digests and compared with `hmac.compare_digest`,
    so a timing side-channel cannot be used to recover one character at a time.
    """
    out: dict[str, Principal] = {}
    for i, chunk in enumerate(p.strip() for p in raw.split(",")):
        if not chunk:
            continue
        key, _, role_name = chunk.partition(":")
        role = ROLE_NAMES.get(role_name.strip().lower() or "analyst")
        if role is None:
            log.warning("unknown role %r in JM_API_KEYS entry %d — skipped", role_name, i)
            continue
        digest = hashlib.sha256(key.strip().encode()).hexdigest()
        out[digest] = Principal(key_id=f"key{i + 1}", role=role)
    return out


class KeyStore:
    def __init__(self) -> None:
        from .config import get_settings
        s = get_settings()
        raw = os.getenv("JM_API_KEYS", "").strip()
        self.keys = _parse_keys(raw) if raw else {}
        self.demo_enabled = bool(not self.keys and s.demo_mode)
        if self.demo_enabled:
            self.keys[hashlib.sha256(DEMO_KEY.encode()).hexdigest()] = Principal(
                key_id="demo", role=Role.ANALYST, is_demo=True)
            log.warning(
                "DEMO AUTH ENABLED: enterprise endpoints accept the built-in key %r. "
                "Set JM_API_KEYS before deploying anywhere real.", DEMO_KEY)
        elif not self.keys:
            log.warning(
                "no JM_API_KEYS configured and DEMO_MODE is off — enterprise "
                "endpoints will refuse every request (fail closed).")

    @property
    def configured(self) -> bool:
        return bool(self.keys) and not self.demo_enabled

    def lookup(self, presented: str | None) -> Principal | None:
        if not presented:
            return None
        digest = hashlib.sha256(presented.strip().encode()).hexdigest()
        for known, principal in self.keys.items():
            if hmac.compare_digest(digest, known):
                return principal
        return None

    def status(self) -> dict:
        return {"configured": self.configured, "demo_auth": self.demo_enabled,
                "keys": len(self.keys)}


_store: KeyStore | None = None


def get_keystore() -> KeyStore:
    global _store
    if _store is None:
        _store = KeyStore()
    return _store


def reset_keystore() -> None:
    """Test hook."""
    global _store
    _store = None


def require_role(minimum: Role):
    """FastAPI dependency factory. Fails closed and says why, without leaking
    whether a given key exists."""

    def dependency(request: Request,
                   x_api_key: str | None = Header(default=None, alias="X-API-Key")
                   ) -> Principal:
        store = get_keystore()
        if not store.keys:
            raise HTTPException(status_code=503, detail={
                "error": "Enterprise access is not configured on this deployment.",
                "code": "auth_not_configured",
                "detail": "Set JM_API_KEYS to enable the enterprise endpoints."})
        principal = store.lookup(x_api_key)
        if principal is None:
            raise HTTPException(status_code=401, detail={
                "error": "A valid X-API-Key header is required.",
                "code": "unauthorised",
                "detail": ("Enterprise endpoints expose population-level data and "
                           "are never open." if not store.demo_enabled else
                           f"This demo deployment accepts X-API-Key: {DEMO_KEY}")})
        if principal.role < minimum:
            raise HTTPException(status_code=403, detail={
                "error": "Your key does not have access to this resource.",
                "code": "forbidden",
                "detail": f"Requires role {minimum.name.lower()} or above."})
        request.state.principal = principal
        return principal

    return dependency


# --------------------------------------------------------------------------
# audit trail
# --------------------------------------------------------------------------
@dataclass
class AuditEntry:
    """One recorded decision.

    Deliberately shaped like the Recommendation Record in
    V2_TRUST_SECURITY_GOVERNANCE.md §81: what was asked, what was decided, by
    which model version, on what evidence, with what confidence. That is what
    makes "why did the system say that, last Tuesday?" answerable.
    """

    at: str
    kind: str                 # recommendation | enterprise_query | override
    actor: str
    request: dict
    decision: dict
    model_versions: dict = field(default_factory=dict)
    confidence: float | None = None
    data_classes: list[str] = field(default_factory=list)
    human_override: dict | None = None

    def as_dict(self) -> dict:
        return {"at": self.at, "kind": self.kind, "actor": self.actor,
                "request": self.request, "decision": self.decision,
                "model_versions": self.model_versions, "confidence": self.confidence,
                "data_classes": self.data_classes, "human_override": self.human_override}


class AuditLog:
    """Append-only, in memory, optionally mirrored to a JSONL file.

    In memory because a pilot does not need a database to be auditable, and a
    ring buffer cannot fill a disk. `JM_AUDIT_LOG=/path/audit.jsonl` turns on
    durable append; the file is only ever appended to, never rewritten.
    """

    def __init__(self, capacity: int = 2000, path: str | None = None) -> None:
        self._entries: deque[AuditEntry] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.path = Path(path) if path else None

    def record(self, entry: AuditEntry) -> AuditEntry:
        with self._lock:
            self._entries.append(entry)
            if self.path:
                try:
                    with open(self.path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(entry.as_dict(), default=str) + "\n")
                except OSError as exc:      # auditing must never break serving
                    log.warning("audit append failed: %s", exc)
        return entry

    def recent(self, limit: int = 100, kind: str | None = None) -> list[dict]:
        with self._lock:
            items = list(self._entries)
        if kind:
            items = [e for e in items if e.kind == kind]
        return [e.as_dict() for e in reversed(items[-limit:])]

    def __len__(self) -> int:
        return len(self._entries)


_audit: AuditLog | None = None


def get_audit_log() -> AuditLog:
    global _audit
    if _audit is None:
        _audit = AuditLog(path=os.getenv("JM_AUDIT_LOG") or None)
    return _audit


def audit(kind: str, actor: str, request: dict, decision: dict,
          model_versions: dict | None = None, confidence: float | None = None,
          data_classes: list[str] | None = None) -> AuditEntry:
    return get_audit_log().record(AuditEntry(
        at=datetime.now().replace(microsecond=0).isoformat(),
        kind=kind, actor=actor, request=request, decision=decision,
        model_versions=model_versions or {}, confidence=confidence,
        data_classes=data_classes or []))
