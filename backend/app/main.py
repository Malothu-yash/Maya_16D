import os
# backend/app/main.py

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import asyncio

from app.routers import auth, chat, sessions, feedback, health, debug, metrics as metrics_router
from app.routers import mini_agent
from app.routers import assistant as assistant_router
from app.routers import emotion as emotion_router
from app.routers import health_extended
from app.routers import user as user_router
from app.routers import memories as memories_router
from app.routers import annotations as annotations_router
from app.routers import tasks, profile, dashboard, ops
from app.routers import youtube as youtube_router
from app.routers import enhanced_memory, database_inspector, data_management
from app.config import settings
from app.services import advanced_emotion
from app.utils.rate_limit import RateLimiter
from app.services.neo4j_service import neo4j_service
from app.services import pinecone_service, redis_service
from app.services.enhanced_memory_service import enhanced_memory_service
from app.database import db_client

# --- Lifespan Management for Connections ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for all external services."""
    print("--- Application starting up... ---")
    
    # Initialize services (non-blocking with short timeouts)
    try:
        await asyncio.wait_for(asyncio.to_thread(pinecone_service.initialize_pinecone), timeout=15.0)
    except Exception as e:
        print(f"⚠️ Pinecone init skipped or timed out: {e}")

    # Neo4j connect with clearer progressive retry (avoid misleading 'timed out' + later 'connected').
    try:
        neo4j_timeout = float(os.getenv("NEO4J_STARTUP_TIMEOUT_SECS", "10"))
    except Exception:
        neo4j_timeout = 10.0
    start_ts = asyncio.get_event_loop().time()
    last_log = 0.0
    while (asyncio.get_event_loop().time() - start_ts) < neo4j_timeout and not getattr(neo4j_service, '_driver', None):
        # One retry cycle (neo4j_service.connect itself has internal multi-attempt logic when retries>1)
        await neo4j_service.connect(retries=1)
        if getattr(neo4j_service, '_driver', None):
            break
        await asyncio.sleep(0.5)
        elapsed = asyncio.get_event_loop().time() - start_ts
        if elapsed - last_log > 2.5:
            remaining = neo4j_timeout - elapsed
            print(f"⏳ Waiting for Neo4j… (elapsed {elapsed:.1f}s, ~{max(0, remaining):.1f}s left)")
            last_log = elapsed
    
    # Initialize enhanced memory service
    try:
        await asyncio.wait_for(enhanced_memory_service.initialize(), timeout=20.0)
        print("✅ Enhanced memory service initialized")
    except Exception as e:
        print(f"⚠️ Enhanced memory service init skipped or timed out: {e}")
    if not getattr(neo4j_service, '_driver', None):
        print(f"⚠️ Neo4j not available after {neo4j_timeout:.1f}s – continuing in degraded mode (graph features disabled)")
    
    # Create Mongo indexes early
    try:
        # Ensure Mongo connects explicitly before creating indexes
        await asyncio.to_thread(db_client.connect)
        await asyncio.to_thread(db_client.ensure_indexes)
    except Exception:
        pass

    # Verify connections and print status messages
    print("--- Verifying Connections ---")
    print("✅ MongoDB connected")
    try:
        redis_ok = await redis_service.ping()
    except Exception:
        redis_ok = False
    print("✅ Redis connected" if redis_ok else "❌ Redis not connected")
    print(f"✅ Pinecone connected" if getattr(pinecone_service, 'index', None) else "❌ Pinecone not connected")
    print(f"✅ Neo4j connected" if getattr(neo4j_service, '_driver', None) else "❌ Neo4j not connected")
    
    # Settings already normalize SMTP_USER/SMTP_PASS from MAIL_* if present
    if not (settings.SMTP_USER and settings.SMTP_PASS):
        print("WARNING: SMTP credentials not set (SMTP_USER/SMTP_PASS or MAIL_USERNAME/MAIL_PASSWORD). Email features will be disabled until configured.")

    print("--- Startup complete. ---")
    yield
    
    # Gracefully close connections on shutdown
    print("--- Application shutting down... ---")
    await neo4j_service.close()
    if redis_service.redis_client:
        await redis_service.redis_client.close()
    if db_client and db_client._client:
        db_client._client.close()
    print("--- Shutdown complete. ---")

from fastapi import Request
from fastapi.responses import JSONResponse
import logging

app = FastAPI(
    title="Personal AI Assistant API",
    lifespan=lifespan,
)

# Global error handler to log all exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )
# Response compression for faster API over network
app.add_middleware(GZipMiddleware, minimum_size=1024)

# --- CORS Middleware ---
# Allow configuration via env var CORS_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"
import os, time

raw_origins = os.getenv("CORS_ORIGINS")
allow_all = os.getenv("CORS_ALLOW_ALL") in {"1", "true", "TRUE"}
# Always allow localhost:3000 and localhost:8000 for React dev
default_dev_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]

# Add known frontend (Vercel) domains so the deployed frontend is allowed by default.
# If you prefer to keep domains configurable via env var, remove or comment these lines.
vercel_frontend_origins = [
    "https://maya-16-d-git-patch-1-malothu-yashs-projects.vercel.app",
    "https://maya-16-d.vercel.app",
]
for o in vercel_frontend_origins:
    if o not in default_dev_origins:
        default_dev_origins.append(o)

if allow_all:
    origins = ["*"]
elif raw_origins:
    origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    for o in default_dev_origins:
        if o not in origins:
            origins.append(o)
else:
    origins = default_dev_origins.copy()

# If an extra DEV_ORIGINS env is present append them
extra = os.getenv("DEV_EXTRA_ORIGINS")
if extra:
    for o in extra.split(','):
        o = o.strip()
        if o and o not in origins:
            origins.append(o)

print(f"CORS origins configured: {origins} allow_all={allow_all}")

# --- Optional CORS debug / dynamic dev reflection ---
DEBUG_CORS = os.getenv("DEBUG_CORS") in {"1", "true", "TRUE"}
ALLOW_DYNAMIC_LOCAL = os.getenv("CORS_DYNAMIC_LOCAL") in {"1", "true", "TRUE"}

# Pre-compute dev host prefixes for dynamic acceptance (ONLY used if ALLOW_DYNAMIC_LOCAL)
_LOCAL_DEV_PREFIXES = ("http://localhost:", "http://127.0.0.1:")

def _is_dynamic_local_origin(origin: str | None) -> bool:
    if not origin:
        return False
    if not ALLOW_DYNAMIC_LOCAL:
        return False
    return origin.startswith(_LOCAL_DEV_PREFIXES)

def _append_dynamic_origin_if_needed(origin: str | None):
    if not origin:
        return
    if origin in origins or origins == ["*"]:
        return
    if _is_dynamic_local_origin(origin):
        # Append for this process lifetime (dev only). Avoid unbounded growth.
        if len(origins) < 20:
            origins.append(origin)
            if DEBUG_CORS:
                print(f"[CORS][dynamic] Added origin at runtime: {origin}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # may be mutated at runtime for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Requested-With", "X-CSRF-Token", "X-Debug-CORS"],
    expose_headers=["X-Session-Id", "X-App-Version", "X-Deprecation"],
)

APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
BOOT_TIME = time.time()

# Lightweight request logger (dev aid) — logs method, path & origin
@app.middleware("http")
async def _cors_and_logging_mw(request: Request, call_next):
    origin = request.headers.get("origin")
    method = request.method
    path = request.url.path

    # Dynamic dev origin learning (before calling next to ensure middleware sees updated list)
    _append_dynamic_origin_if_needed(origin)

    # Handle preflight early (so even unknown routes show CORS headers for diagnosis)
    if method == "OPTIONS":
        from fastapi.responses import PlainTextResponse
        resp = PlainTextResponse("preflight ok", status_code=200)
        if origins == ["*"] and origin:
            resp.headers["Access-Control-Allow-Origin"] = origin
        elif origin and (origin in origins or _is_dynamic_local_origin(origin)):
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers.setdefault("Vary", "Origin")
        # Standard CORS preflight headers
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,PATCH,DELETE,OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = request.headers.get("Access-Control-Request-Headers", "Authorization,Content-Type,Accept")
        resp.headers["Access-Control-Max-Age"] = "600"
        resp.headers["X-Debug-CORS"] = "1"
        if DEBUG_CORS:
            print(f"[CORS][preflight] {path} origin={origin} allowed={resp.headers.get('Access-Control-Allow-Origin') is not None}")
        return resp

    response = await call_next(request)

    # Ensure debug header present for troubleshooting
    response.headers.setdefault("X-Debug-CORS", "1")

    try:
        if origins == ["*"] and origin:
            response.headers.setdefault("Access-Control-Allow-Origin", origin)
        elif origin and (origin in origins or _is_dynamic_local_origin(origin)) and "access-control-allow-origin" not in response.headers:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers.setdefault("Vary", "Origin")
    except Exception:
        pass

    if DEBUG_CORS:
        allowed = response.headers.get("Access-Control-Allow-Origin")
        print(f"[CORS][req] {method} {path} origin={origin} allow={bool(allowed)} status={response.status_code}")
    else:
        # Lightweight single-line log for non-health endpoints
        if not path.startswith("/health"):
            print(f"REQ {method} {path} origin={origin} -> {response.status_code}")
    return response

# --- Global Exception Handlers (standard error envelope) ---
@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    payload = {"error": {"code": exc.status_code, "message": exc.detail}}
    return JSONResponse(status_code=exc.status_code, content=payload)

@app.exception_handler(RequestValidationError)
async def validation_exc_handler(request: Request, exc: RequestValidationError):
    payload = {"error": {"code": 422, "message": "Validation error", "details": exc.errors()}}
    return JSONResponse(status_code=422, content=payload)

# Map common infra RuntimeErrors (e.g., DB not available) to a clearer 503 instead of 500
@app.exception_handler(RuntimeError)
async def runtime_exc_handler(request: Request, exc: RuntimeError):
    msg = str(exc) or "Service unavailable"
    payload = {"error": {"code": 503, "message": msg}}
    return JSONResponse(status_code=503, content=payload)

# Generic fallback
@app.middleware("http")
async def _wrap_unhandled(request: Request, call_next):
    try:
        return await call_next(request)
    except HTTPException:
        # Let FastAPI's HTTPException flow to its handler
        raise
    except RuntimeError:
        # Allow our RuntimeError handler to map to 503
        raise
    except Exception as e:  # noqa: BLE001
        # Last-resort safety net
        payload = {"error": {"code": 500, "message": "Internal server error"}}
        try:
            print(f"Unhandled exception at {request.url.path}: {e}")
        except Exception:
            pass
        return JSONResponse(status_code=500, content=payload)

# Global per-user rate limit (e.g., 120 req/min)
app.middleware("http")(RateLimiter(max_requests=120, window_seconds=60))

# --- Routers ---
app.include_router(auth.router)
app.include_router(auth.legacy_router)
app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(feedback.router)
app.include_router(health.router)
from app.routers.user import public_router
app.include_router(public_router)
app.include_router(user_router.router)
app.include_router(debug.router)
app.include_router(memories_router.router)
app.include_router(metrics_router.router)
app.include_router(health_extended.router)
app.include_router(emotion_router.router)
app.include_router(tasks.router)
app.include_router(ops.router)
app.include_router(profile.router)
app.include_router(dashboard.router)
app.include_router(mini_agent.router)
app.include_router(annotations_router.router)
app.include_router(youtube_router.router)
app.include_router(assistant_router.router)
app.include_router(enhanced_memory.router)
app.include_router(database_inspector.router)
app.include_router(data_management.router)

@app.on_event("startup")
async def _warm_advanced_emotion():
    if settings.ADV_EMOTION_ENABLE:
        try:
            await advanced_emotion.load_model()
            print("Advanced emotion model warmed (stub)")
        except Exception as e:  # noqa: BLE001
            print(f"Advanced emotion warm-load failed: {e}")


# --- Final CORS enforcement middleware (hardens error paths) ---
@app.middleware("http")
async def _final_cors_enforcer(request: Request, call_next):
    response = await call_next(request)
    try:
        origin = request.headers.get("origin")
        if origin:
            allowed = False
            if allow_all:
                allowed = True
            elif origin in origins or _is_dynamic_local_origin(origin):
                allowed = True
            if allowed:
                # Always set (overwrites missing header on 4xx/5xx)
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers.setdefault("Vary", "Origin")
                response.headers.setdefault("Access-Control-Allow-Credentials", "true")
                # Ensure standard exposed headers for frontend
                existing_expose = response.headers.get("Access-Control-Expose-Headers", "")
                needed = {"X-Session-Id", "X-App-Version", "X-Deprecation"}
                merged = set(h.strip() for h in existing_expose.split(",") if h.strip()) | needed
                response.headers["Access-Control-Expose-Headers"] = ", ".join(sorted(merged))
    except Exception:  # noqa: BLE001
        pass
    return response

# --- Deprecation Header Middleware (non-breaking guidance) ---
def _is_deprecated_path(path: str) -> tuple[bool, str | None, str | None]:
    sunset = os.getenv("DEPRECATION_SUNSET", "2025-12-31")
    # Exact matches
    if path == "/api/debug/echo":
        return True, "Endpoint is for internal debugging and will be removed.", sunset
    if path == "/api/memories":
        return True, "Use /api/memories/ (trailing slash).", sunset
    if path == "/api/chat/new/stream":
        return True, "Use /api/chat/{id}/stream or consolidate on sessions chat.", sunset
    # Prefix/heuristic matches
    if path.startswith("/metrics/"):
        return True, "Use /api/metrics/* endpoints.", sunset
    if path.startswith("/auth/"):
        return True, "Legacy route; use /api/auth/*.", sunset
    if path.startswith("/api/chat/") and path.endswith("/stream"):
        return True, "Chat streaming path may be consolidated; prefer sessions chat.", sunset
    if path in {"/api/ops/celery_health_check", "/api/ops/peek_task", "/api/ops/list_recent_tasks"}:
        return True, "Operational endpoint may be removed or role-gated.", sunset
    return False, None, None


@app.middleware("http")
async def _deprecation_middleware(request: Request, call_next):
    response = await call_next(request)
    try:
        deprecated, msg, sunset = _is_deprecated_path(request.url.path)
        if deprecated:
            response.headers["X-Deprecation"] = msg or "This endpoint is deprecated."
            if sunset:
                response.headers.setdefault("X-Deprecation-Sunset", sunset)
    except Exception:
        pass
    return response

# --- CORS diagnostic endpoint ---
@app.get("/health/cors", tags=["Health"])
def cors_echo(request: Request):  # type: ignore[override]
    origin = request.headers.get("origin")
    return {
        "received_origin": origin,
        "configured_origins": origins,
        "dynamic_dev_enabled": ALLOW_DYNAMIC_LOCAL,
        "debug_mode": DEBUG_CORS,
        "allowed_via": (
            "*" if origins == ["*"] else ("list" if origin in origins else ("dynamic" if _is_dynamic_local_origin(origin) else "none"))
        ),
    }
try:
    # include mirrored /api/metrics router if present
    app.include_router(metrics_router.api_router)  # type: ignore[attr-defined]
except Exception:
    pass

@app.get("/", tags=["Root"])
def read_root():
    return {"status": "API is running"}

@app.get("/api/info", tags=["Meta"])
def api_info():
    uptime = time.time() - BOOT_TIME
    return {
        "version": APP_VERSION,
        "uptime_seconds": int(uptime),
        "origins": origins,
        "allow_all_cors": allow_all,
    }
