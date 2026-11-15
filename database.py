"""
Authentication (module name kept from the original project): hashed passwords in a users file, expiring JWTs, login throttling.

No credentials or secrets live in the source code:
  * the JWT secret comes from the JWT_SECRET environment variable,
  * users live in USERS_FILE (managed with manage_users.py), passwords stored as scrypt hashes.
"""
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Optional

import jwt
from fastapi import Depends, Header, HTTPException

import config

log = logging.getLogger(__name__)

ROLES = {"admin", "viewer"}
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1
_MAXMEM = 64 * 1024 * 1024


# ------------------------------------------------------------------ passwords
def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R,
                            p=_SCRYPT_P, maxmem=_MAXMEM, dklen=32)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_b64, hash_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(hash_b64)
        digest = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt_b64), n=int(n),
                                r=int(r), p=int(p), maxmem=_MAXMEM, dklen=len(expected))
        return hmac.compare_digest(digest, expected)
    except (ValueError, TypeError):
        return False


_dummy_hash: Optional[str] = None


def _dummy() -> str:
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = hash_password(secrets.token_urlsafe(8))
    return _dummy_hash


# ------------------------------------------------------------------ user store
@dataclass(frozen=True)
class User:
    username: str
    role: str
    password_hash: str


def load_users_file(path: str) -> dict[str, User]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f).get("users", {})
    users = {}
    for name, rec in raw.items():
        if rec.get("role") not in ROLES or not rec.get("password_hash"):
            log.warning("Skipping invalid user record %r in %s", name, path)
            continue
        users[name] = User(name, rec["role"], rec["password_hash"])
    return users


def save_users_file(path: str, users: dict[str, User]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = {"users": {u.username: {"role": u.role, "password_hash": u.password_hash}
                      for u in users.values()}}
    tmp = f"{path}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


class UserStore:
    """Reads the users file and transparently reloads it when it changes on disk."""

    def __init__(self, path: str):
        self.path = path
        self._users: dict[str, User] = {}
        self._mtime = None
        self._lock = threading.Lock()

    def _refresh(self) -> None:
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            self._users, self._mtime = {}, None
            return
        if mtime != self._mtime:
            try:
                self._users, self._mtime = load_users_file(self.path), mtime
            except (OSError, ValueError):
                log.exception("Could not read users file %s", self.path)

    def get(self, username: str) -> Optional[User]:
        with self._lock:
            self._refresh()
            return self._users.get(username)

    def count(self) -> int:
        with self._lock:
            self._refresh()
            return len(self._users)

    def authenticate(self, username: str, password: str) -> Optional[User]:
        user = self.get(username)
        # Always run one scrypt, so response time does not reveal whether the user exists.
        ok = verify_password(password, user.password_hash if user else _dummy())
        return user if (user and ok) else None


_store: Optional[UserStore] = None


def get_user_store() -> UserStore:
    global _store
    if _store is None:
        _store = UserStore(config.USERS_FILE)
    return _store


def set_user_store(store: Optional[UserStore]) -> None:
    global _store
    _store = store


def bootstrap_users_if_needed() -> None:
    """First run only (no users yet): create `admin` and optionally `viewer` from environment passwords."""
    store = get_user_store()
    if store.count() > 0:
        return
    wanted = {}
    if config.BOOTSTRAP_ADMIN_PASSWORD:
        wanted[config.BOOTSTRAP_ADMIN_USERNAME] = ("admin", config.BOOTSTRAP_ADMIN_PASSWORD)
    if config.BOOTSTRAP_VIEWER_PASSWORD:
        wanted[config.BOOTSTRAP_VIEWER_USERNAME] = ("viewer", config.BOOTSTRAP_VIEWER_PASSWORD)
    if not wanted:
        log.warning("No users exist. Create one with: python manage_users.py add <name> --role admin")
        return
    users = {}
    for name, (role, password) in wanted.items():
        if len(password) < config.MIN_PASSWORD_LENGTH:
            raise RuntimeError(f"Bootstrap password for {name!r} must be at least {config.MIN_PASSWORD_LENGTH} characters")
        users[name] = User(name, role, hash_password(password))
    save_users_file(store.path, users)
    log.warning("Created bootstrap users %s. Remove the BOOTSTRAP_* passwords from the environment now.",
                ", ".join(users))


# ------------------------------------------------------------------ JWT
def ensure_jwt_secret() -> str:
    """Return the signing secret; refuse to run in production without a strong one."""
    secret = config.JWT_SECRET
    if secret and (len(secret) >= 32 or config.IS_DEV):
        return secret
    if secret:
        raise RuntimeError("JWT_SECRET is too short: use at least 32 characters "
                           "(e.g. `python -c \"import secrets; print(secrets.token_urlsafe(48))\"`).")
    if config.IS_DEV:
        config.JWT_SECRET = secrets.token_urlsafe(48)
        log.warning("JWT_SECRET is not set: using a random per-process secret (dev mode only; "
                    "all tokens become invalid on restart).")
        return config.JWT_SECRET
    raise RuntimeError("JWT_SECRET is not set. Refusing to start without a signing secret.")


def create_access_token(username: str, role: str, minutes: Optional[int] = None,
                        now: Optional[float] = None) -> tuple[str, int]:
    """Returns (token, expires_in_seconds)."""
    issued = int(now if now is not None else time.time())
    ttl = int((minutes if minutes is not None else config.ACCESS_TOKEN_MINUTES) * 60)
    token = jwt.encode({"sub": username, "role": role, "iat": issued, "exp": issued + ttl},
                       ensure_jwt_secret(), algorithm=config.ALGORITHM)
    return token, ttl


def verify_token(token: Optional[str]) -> Optional[dict]:
    """Decoded payload, or None if the token is missing, expired, tampered with or malformed."""
    if not token or not isinstance(token, str):
        return None
    try:
        return jwt.decode(token, ensure_jwt_secret(), algorithms=[config.ALGORITHM],
                          options={"require": ["exp", "iat", "sub"]})
    except jwt.PyJWTError as exc:
        log.info("Rejected token: %s", type(exc).__name__)
        return None


def user_from_token(token: Optional[str]) -> Optional[User]:
    """Valid token AND the account still exists. The stored role wins over the role in the token."""
    payload = verify_token(token)
    return get_user_store().get(payload["sub"]) if payload else None


# ------------------------------------------------------------------ FastAPI dependencies
def get_current_user(authorization: str = Header(None)) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization header")
    user = user_from_token(authorization[7:].strip())
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user


def get_current_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user


# ------------------------------------------------------------------ login throttling
class LoginThrottle:
    """Blocks a (client, username) pair after too many failed logins in a window."""

    def __init__(self, max_failures: int, window_seconds: int, clock: Callable[[], float] = time.time):
        self.max_failures, self.window, self.clock = max_failures, window_seconds, clock
        self._fails: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, key: str) -> deque:
        q, now = self._fails[key], self.clock()
        while q and now - q[0] >= self.window:
            q.popleft()
        if not q:
            self._fails.pop(key, None)
        return q

    def retry_after(self, key: str) -> int:
        """0 when allowed, otherwise seconds until the oldest failure expires."""
        with self._lock:
            q = self._prune(key)
            if len(q) < self.max_failures:
                return 0
            return int(self.window - (self.clock() - q[0])) + 1

    def record_failure(self, key: str) -> None:
        with self._lock:
            self._fails[key].append(self.clock())

    def reset(self, key: str) -> None:
        with self._lock:
            self._fails.pop(key, None)
