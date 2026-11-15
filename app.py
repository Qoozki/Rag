# app.py
import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from openinference.instrumentation.langchain import LangChainInstrumentor
import logfire

import config
import database
from filters import FilterError, describe_filters, normalize_filters, parse_filter_command
from logging_config import request_id_var, setup_logging
from rate_limiter import RateLimitStore, TokenBudget

log = logging.getLogger("app")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
_MAGIC = {".pdf": (b"%PDF-",), ".docx": (b"PK\x03\x04",)}


class RagBackend:
    """Thin wrapper around rag_service so the web layer can be tested with a fake."""

    def init(self):
        import rag_service
        return rag_service.init_rag()

    async def run(self, text, ws, history, filters):
        import rag_service
        return await rag_service.run_two_pass_rag(text, ws, history, filters)

    def sync(self, full: bool):
        import rag_service
        return rag_service.rebuild_vector_store_service() if full else rag_service.sync_vector_store_service()

    def delete(self, source_id: str):
        import rag_service
        rag_service.delete_document_service(source_id)

    def documents(self):
        import rag_service
        return rag_service.list_documents_service()

    def error_text(self, exc: Exception) -> str:
        import rag_service
        return rag_service.friendly_error(exc)

    def close(self):
        import rag_service
        rag_service.close_vector_client()


class AppState:
    def __init__(self, backend):
        self.backend = backend
        self.ready = False
        self.init_error = None
        self.sync_lock = asyncio.Lock()
        self.rate = RateLimitStore(config.RATE_LIMIT_REQUESTS, config.RATE_LIMIT_WINDOW_SECONDS)
        self.budget = TokenBudget(config.TOKEN_BUDGET, config.TOKEN_BUDGET_WINDOW_SECONDS)
        self.throttle = database.LoginThrottle(config.LOGIN_MAX_FAILURES, config.LOGIN_WINDOW_SECONDS)


def safe_filename(name):
    base = os.path.basename((name or "").replace("\\", "/")).strip()
    if not base or base.startswith((".", "~$")) or "\x00" in base:
        raise HTTPException(400, "Invalid file name.")
    if os.path.splitext(base)[1].lower() not in config.SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(config.SUPPORTED_EXTENSIONS)}")
    return base


def resolve_source(source_id: str) -> str:
    root = os.path.realpath(config.DATA_PATH)
    full = os.path.realpath(os.path.join(root, source_id))
    if not full.startswith(root + os.sep):
        raise HTTPException(400, "Invalid document path.")
    return full


def create_app(backend=None) -> FastAPI:
    backend = backend or RagBackend()

    async def initialise(state: AppState):
        try:
            report = await asyncio.to_thread(backend.init)
            state.ready = True
            log.info("Knowledge base ready: %s", report.summary() if hasattr(report, "summary") else report)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state.init_error = f"{type(exc).__name__}: {exc}"
            log.exception("Initialisation failed; the app stays up but chat is unavailable")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(config.LOG_LEVEL, config.LOG_JSON)
        database.ensure_jwt_secret()                 # fail fast: no secret, no start
        database.bootstrap_users_if_needed()
        os.makedirs(config.DATA_PATH, exist_ok=True)
        state = AppState(backend)
        app.state.rag_state = state
        task = asyncio.create_task(initialise(state))
        yield
        task.cancel()
        try:
            await asyncio.to_thread(backend.close)
        except Exception:
            log.exception("Error while closing the vector store")

    app = FastAPI(lifespan=lifespan)

    def st() -> AppState:
        return app.state.rag_state

    def require_ready() -> AppState:
        s = st()
        if not s.ready:
            raise HTTPException(503, s.init_error or "The knowledge base is still starting up.")
        return s

    # ------------------------------------------------------------ pages / health
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        s = st()
        if s.ready:
            return {"status": "ready"}
        return JSONResponse({"status": "starting", "error": s.init_error}, status_code=503)

    @app.get("/login")
    async def login_page(request: Request):
        return templates.TemplateResponse(request, "login.html")

    @app.get("/")
    async def get_homepage(request: Request):
        return templates.TemplateResponse(request, "index.html")

    # ------------------------------------------------------------ auth
    @app.post("/login")
    async def login(request: Request, username: str = Form(...), password: str = Form(...)):
        key = f"{request.client.host if request.client else '?'}|{username.lower()}"
        wait = st().throttle.retry_after(key)
        if wait:
            raise HTTPException(429, f"Too many failed attempts. Try again in {wait} seconds.",
                                headers={"Retry-After": str(wait)})
        user = await asyncio.to_thread(database.get_user_store().authenticate, username, password)
        if not user:
            st().throttle.record_failure(key)
            log.warning("Failed login for %r", username)
            raise HTTPException(400, "Invalid username or password")
        st().throttle.reset(key)
        token, ttl = database.create_access_token(user.username, user.role)
        log.info("Login: %s (%s)", user.username, user.role)
        return {"access_token": token, "username": user.username, "role": user.role, "expires_in": ttl}

    # ------------------------------------------------------------ documents
    @app.get("/documents")
    async def list_documents(user=Depends(database.get_current_user)):
        require_ready()
        return {"documents": backend.documents()}

    @app.post("/admin/upload")
    async def upload_document(file: UploadFile = File(...), admin=Depends(database.get_current_admin)):
        name = safe_filename(file.filename)
        ext = os.path.splitext(name)[1].lower()
        limit = config.MAX_UPLOAD_MB * 1024 * 1024
        os.makedirs(config.DATA_PATH, exist_ok=True)
        final = os.path.join(config.DATA_PATH, name)
        tmp = os.path.join(config.DATA_PATH, f".upload-{uuid.uuid4().hex}.tmp")   # hidden: never indexed
        size, first = 0, True
        try:
            with open(tmp, "wb") as out:
                while chunk := await file.read(1 << 20):
                    if first:
                        if not chunk.startswith(_MAGIC[ext]):
                            raise HTTPException(400, f"File content does not look like a real {ext} file.")
                        first = False
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_MB} MB.")
                    out.write(chunk)
            if size == 0:
                raise HTTPException(400, "The file is empty.")
            os.replace(tmp, final)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        log.info("Upload by %s: %s (%d bytes)", admin.username, name, size)
        return {"message": f"Successfully uploaded {name}. Click 'Sync Index' to index it."}

    async def run_sync(full: bool, admin) -> dict:
        s = require_ready()
        if s.sync_lock.locked():
            raise HTTPException(409, "An index update is already running.")
        async with s.sync_lock:
            log.info("Index %s requested by %s", "rebuild" if full else "sync", admin.username)
            report = await asyncio.to_thread(backend.sync, full)
        return {"message": f"Index {'rebuilt' if full else 'synced'}: {report.summary()}.",
                "report": report.to_dict()}

    @app.post("/admin/reindex")
    async def reindex(full: bool = False, admin=Depends(database.get_current_admin)):
        return await run_sync(full, admin)

    @app.post("/admin/rebuild")     # original endpoint: full rebuild
    async def rebuild_vector_store(admin=Depends(database.get_current_admin)):
        return await run_sync(True, admin)

    @app.delete("/admin/documents/{source_id:path}")
    async def delete_document(source_id: str, admin=Depends(database.get_current_admin)):
        require_ready()
        path = resolve_source(source_id)
        existed = os.path.isfile(path)
        if existed:
            os.remove(path)
        await asyncio.to_thread(backend.delete, source_id)
        if not existed:
            raise HTTPException(404, "No such document.")
        log.info("Document deleted by %s: %s", admin.username, source_id)
        return {"message": f"Deleted {source_id} and removed it from the index."}

    # ------------------------------------------------------------ chat
    async def ws_authenticate(ws: WebSocket):
        try:
            msg = json.loads(await asyncio.wait_for(ws.receive_text(), config.WS_AUTH_TIMEOUT))
        except (asyncio.TimeoutError, ValueError):
            return None
        if not isinstance(msg, dict) or msg.get("type") != "auth":
            return None
        payload = database.verify_token(msg.get("token"))
        user = database.get_user_store().get(payload["sub"]) if payload else None
        return (user, payload["exp"]) if user else None

    @app.websocket("/ws/chat")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            authed = await ws_authenticate(websocket)
            if not authed:
                await websocket.close(code=1008)     # bad / missing / expired token
                return
            user, expires_at = authed
            user_name = user.username
            s = st()
            filters: dict = {}
            history = []   # [(question, answer)] for this connection only

            await websocket.send_text(
                f"SYSTEM: Welcome {user_name}! Ask me anything (English or Arabic). "
                f"I remember the last {config.HISTORY_TURNS} turns, so follow-up questions work "
                f"(type /reset to clear the memory). "
                f"Use /filter to restrict the search, e.g. /filter file=Company.pdf lang=en page=2-5 "
                f"(/filter clear to remove).\n\n")
            used = s.budget.used(user_name)
            if used:
                await websocket.send_text(f'METADATA:{{"tokens": {used}}}')   # keeps the UI counter honest

            while True:
                remaining = expires_at - time.time()
                if remaining <= 0:
                    await websocket.send_text("SYSTEM: Your session has expired. Please log in again.\n")
                    await websocket.close(code=1008)
                    return
                try:
                    raw = (await asyncio.wait_for(websocket.receive_text(), timeout=remaining)).strip()
                except asyncio.TimeoutError:
                    continue
                if not raw:
                    continue

                kind, text = "question", raw
                if raw.startswith("{"):
                    try:
                        obj = json.loads(raw)
                        if isinstance(obj, dict) and obj.get("type") in {"question", "filters", "reset"}:
                            kind = obj["type"]
                            text = obj.get("filters") if kind == "filters" else str(obj.get("text", "")).strip()
                    except ValueError:
                        pass
                elif raw == "/reset":
                    kind = "reset"
                elif raw == "/filter" or raw.startswith("/filter "):
                    kind, text = "filters", raw[7:].strip()

                if kind == "reset":
                    history.clear()
                    await websocket.send_text("SYSTEM: Conversation memory cleared.\n\n")
                    continue

                if kind == "filters":
                    try:
                        if isinstance(text, str):
                            if text.lower() in {"clear", "off", "none"}:
                                filters = {}
                            elif text:
                                filters = parse_filter_command(text)
                        else:
                            filters = normalize_filters(text)
                        await websocket.send_text(f"SYSTEM: Active filters: {describe_filters(filters)}\n\n")
                    except FilterError as exc:
                        await websocket.send_text(f"SYSTEM: Invalid filter: {exc}\n\n")
                    continue

                user_message = text
                if not user_message:
                    continue
                if len(user_message) > config.MAX_QUESTION_CHARS:
                    await websocket.send_text(f"SYSTEM: Question too long (max {config.MAX_QUESTION_CHARS} characters).\n\n")
                    continue
                if not s.ready:
                    await websocket.send_text("SYSTEM: The knowledge base is still starting up. Please try again shortly.\n\n")
                    continue
                if s.budget.remaining(user_name) <= 0:
                    await websocket.send_text("SYSTEM: Your token allowance is used up. Try again later.\n\n")
                    continue

                allowed, retry_after = s.rate.check(user_name)
                if not allowed:
                    log.info("Rate limit hit for %s (retry in %ss)", user_name, retry_after)
                    await websocket.send_text(f'RATE_LIMIT:{{"retry_after": {retry_after}}}')
                    await websocket.send_text(f"\n[SYSTEM WARNING] Rate limit exceeded! Please wait {retry_after} seconds.\n\n")
                    continue

                await websocket.send_text(f"AI ({user_name}): ")

                # Delegate heavy RAG operations to the service layer
                try:
                    result = await backend.run(user_message, websocket, history, filters)
                    s.budget.add(user_name, result.tokens)
                    if result.answer and not result.refused:
                        history.append((user_message, result.answer))
                        del history[:-config.HISTORY_TURNS]
                except WebSocketDisconnect:
                    raise
                except Exception as exc:
                    # keep the connection alive if one question fails; details stay in the log
                    ref = request_id_var.get()
                    log.exception("Question failed (ref %s)", ref)
                    try:
                        await websocket.send_text(f"\n[SERVER ERROR] {backend.error_text(exc)} (ref: {ref})\n")
                    except Exception:
                        pass
                try:
                    await websocket.send_text("\n\n")
                except Exception:
                    pass
        except WebSocketDisconnect:
            log.info("Client disconnected.")

    return app


# Initialize Logfire and Instrumentation (as in the original)
logfire.configure(send_to_logfire="if-token-present")
LangChainInstrumentor().instrument()

app = create_app()
