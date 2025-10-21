# backend/app/routers/chat.py

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pymongo.collection import Collection
from bson import ObjectId, errors
from datetime import datetime
from typing import Optional, Any, Dict, Union
import asyncio
import json
import re
import logging

# -----------------------------
# Core App Imports
# -----------------------------
from app.celery_app import celery_app
from app.security import get_current_active_user
from app.services import task_nlp, task_flow_service
from app.services.neo4j_service import neo4j_service
from app.services.memory_coordinator import gather_memory_context, post_message_update
from app.services import deterministic_extractor, profile_service
from app.services import nlu
from app.services import redis_service
from app.services import llm_brain
from app.services import nlg_engine
from app.services import memory_manager
import os, json
from app.services.ai_service import get_response as ai_get_response
import re
import httpx
import os
from app.config import settings
from app.database import (
    get_sessions_collection,
    get_tasks_collection,
    get_chat_log_collection,
)
from app.database import get_activity_logs_collection
# Email sending removed
from datetime import timezone, timedelta
import dateparser
from dateparser.search import search_dates
# Celery task names (they are registered in celery_worker)
# we don't import functions directly, just use task names
# extract_and_store_facts, prefetch_destination_info

# -----------------------------
# Router Setup
# -----------------------------
router = APIRouter(prefix="/api/chat", tags=["Chat"])
logger = logging.getLogger(__name__)
DEBUG_BRAIN = os.getenv("DEBUG_BRAIN", "false").lower() == "true"

# -----------------------------
# Pydantic Models
# -----------------------------
class ChatMessage(BaseModel):
    message: str

class TaskCreate(BaseModel):
    content: str
    due_date: str

class TaskUpdate(BaseModel):
    content: Optional[str] = None
    due_date: Optional[str] = None

class NewChatRequest(BaseModel):
    message: str

class NewChatResponse(BaseModel):
    session_id: str
    response_text: str
    # Optional fast-path video payload for inline embedding on the client
    video: Optional[dict] = None
    # Expose intent metadata to the client for clarification flows
    video_intent: Optional[dict] = None
    # IDs of persisted messages (when available)
    ai_message_id: Optional[str] = None
    user_message_id: Optional[str] = None

class ContinueChatRequest(BaseModel):
    message: str

class ContinueChatResponse(BaseModel):
    response_text: str
    video: Optional[dict] = None
    video_intent: Optional[dict] = None
    ai_message_id: Optional[str] = None
    user_message_id: Optional[str] = None

class ChatRequest(BaseModel):
    # Compatibility model for external clients
    user_id: str
    session_id: Optional[str] = None
    message: str
    metadata: Optional[Dict] = None

class GenerateRequest(BaseModel):
    prompt: str

class GenerateJSONResponse(BaseModel):
    response: Optional[str] = None
    provider_used: Optional[str] = None
    error: Optional[str] = None

class VideoControlRequest(BaseModel):
    action: str  # play|pause|replay|next|lyrics
    session_id: Optional[str] = None
    current_video_id: Optional[str] = None
    current_title: Optional[str] = None
    current_context: Optional[dict] = None  # e.g., movie/artist derived

class VideoControlResponse(BaseModel):
    response_text: str
    video: Optional[dict] = None
    lyrics: Optional[str] = None
# Small helper to normalize to naive UTC
def _to_utc_naive(dt: datetime) -> datetime:
    if dt is None:  # Check if the datetime is None
        return dt  # Return None if input is None
    if dt.tzinfo is not None:  # If the datetime has timezone info
        return dt.astimezone(timezone.utc).replace(tzinfo=None)  # Convert to UTC and remove timezone info
    return dt  # Return the naive datetime if no timezone info is present

def _format_eta_for_user(eta_utc_naive: datetime, user_tz: str | None, time_format: str | None = None) -> str:
    """Return a friendly time string in the user's timezone when possible, else UTC.

    Example: 2025-10-10 05:32 IST (Asia/Kolkata), falling back to 'YYYY-MM-DD HH:MM UTC'.
    """
    try:  # Try to format the ETA
        if not isinstance(eta_utc_naive, datetime):  # Check if the input is a datetime instance
            return str(eta_utc_naive)  # Return as string if not a datetime
        aware_utc = eta_utc_naive.replace(tzinfo=timezone.utc)  # Make the datetime aware in UTC
        if user_tz:  # If user timezone is provided
            try:
                from zoneinfo import ZoneInfo  # Python 3.9+
                local_dt = aware_utc.astimezone(ZoneInfo(user_tz))  # Convert to user's timezone
                fmt = "%Y-%m-%d %I:%M %p %Z" if (time_format or "").lower().startswith("12") else "%Y-%m-%d %H:%M %Z"
                return local_dt.strftime(fmt)  # Format and return the local datetime
            except Exception:
                pass  # Ignore exceptions and fall back to UTC
        # Fallback to UTC format
        fmt = "%Y-%m-%d %I:%M %p UTC" if (time_format or "").lower().startswith("12") else "%Y-%m-%d %H:%M UTC"
        return aware_utc.strftime(fmt)  # Return formatted UTC datetime
    except Exception:  # Catch any exceptions
        return str(eta_utc_naive)  # Return as string if any error occurs

async def _log_nlu(activity_logs: Optional[Collection], user_id: str, session_id: str, message: str, nlu_result: Dict[str, Any]) -> None:
    """Persist NLU outcome asynchronously without blocking request path."""
    if not activity_logs:
        return
    try:
        doc = {
            "type": "nlu_result",
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "nlu": nlu_result,
            "timestamp": datetime.utcnow(),
        }
        await run_in_threadpool(activity_logs.insert_one, doc)
    except Exception:
        try:
            logger.debug("nlu log insert failed", exc_info=True)
        except Exception:
            pass

async def _nlu_and_log(activity_logs: Optional[Collection], user_id: str, session_id: str, message: str) -> None:
    """Run NLU and log result; designed for fire-and-forget via create_task."""
    try:
        res = await nlu.extract_intent_entities(message=message, session_id=session_id, user_id=user_id)
        await _log_nlu(activity_logs, user_id, session_id, message, res)
    except Exception:
        try:
            logger.debug("nlu call/log failed", exc_info=True)
        except Exception:
            pass

async def _brain_and_log(
    *,
    user_id: str,
    session_id: str,
    message: str,
    nlu_result: Optional[Dict[str, Any]],
    include_in_response: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run LLM Brain planner using NLU outputs and log/return plan for debug if requested."""
    try:
        intent = (nlu_result or {}).get("intent") if isinstance(nlu_result, dict) else None
        entities = (nlu_result or {}).get("entities") if isinstance(nlu_result, dict) else {}
        plan = await llm_brain.plan_actions(
            intent=intent,
            entities=entities,
            user_id=user_id,
            session_id=session_id,
            required_function="TextCompletion",
            hint_text=message,
        )
        return plan if include_in_response else None
    except Exception:
        try:
            logger.debug("llm_brain.plan_actions failed", exc_info=True)
        except Exception:
            pass
        return None

async def _brain_nlg_pipeline(
    *,
    user_id: str,
    session_id: str,
    message: str,
    nlu_result: Optional[Dict[str, Any]],
    include_in_response: bool = False,
) -> Optional[Dict[str, Any]]:
    """End-to-end background pipeline: Brain → MemoryManager (async) → NLG.
    Returns {plan, nlg} for debug when include_in_response is True; otherwise None.
    """
    plan = None
    try:
        intent = (nlu_result or {}).get("intent") if isinstance(nlu_result, dict) else None
        entities = (nlu_result or {}).get("entities") if isinstance(nlu_result, dict) else {}
        plan = await llm_brain.plan_actions(
            intent=intent,
            entities=entities,
            user_id=user_id,
            session_id=session_id,
            required_function="TextCompletion",
            hint_text=message,
        )
    except Exception:
        try:
            logger.debug("llm_brain.plan_actions failed (pipeline)", exc_info=True)
        except Exception:
            pass

    # Kick off memory updates in background (non-blocking)
    try:
        if plan and isinstance(plan.get("memory_updates"), dict):
            asyncio.create_task(memory_manager.update_memories(plan["memory_updates"], user_id, session_id))
    except Exception:
        pass

    # Generate NLG response using plan.nlg_input (non-blocking of main request; here awaited since in bg task)
    nlg = None
    try:
        if plan and isinstance(plan.get("nlg_input"), dict):
            nlg = await nlg_engine.generate_response(plan["nlg_input"], user_id, session_id)
    except Exception:
        try:
            logger.debug("nlg_engine.generate_response failed", exc_info=True)
        except Exception:
            pass

    return {"plan": plan, "nlg": nlg} if include_in_response else None

class FeedbackBody(BaseModel):
    fact_id: str
    correction: str
def _auto_title_from_first_message(raw: str) -> str:
    try:
        if not raw:
            return "New Chat"
        low = raw.lower().strip()
        if "remind me" in low or low.startswith("reminder"):
            m = re.search(r"remind\s+me\s+(?:to|about)\s+(.+)", low)
            core = (m.group(1) if m else raw)
            core = re.sub(r"\s+in\s+\d+.*$", "", core).strip()
            title = f"Reminder: {core[:30]}"
        elif raw.endswith("?"):
            title = f"Question: {raw}"
        else:
            words = re.split(r"\s+", raw)
            short = " ".join(words[:8]).replace("\n", " ").strip(" -_.,")
            title = short.title() if short else "New Chat"
        return (title[:50] + ("" if len(title) <= 50 else "")) if len(title) <= 50 else title[:47] + "..."
    except Exception:
        return (raw[:47] + "...") if len(raw) > 50 else (raw or "New Chat")

# =====================================================
# 🔹 Helper: Try creating and scheduling a reminder from a free-text message
# =====================================================
async def _maybe_create_and_schedule_reminder(
    message: str,
    current_user: dict,
    tasks: Collection,
    context: dict,
):
    """Parse reminder intent, create a task document, schedule Celery, and send confirmation email.

    Returns a dict with creation details when a reminder is created, else None.
      {"created": True, "task_id": str, "title": str, "eta_utc": datetime}
    """
    # 1) Try structured NLU
    try:
        nlu_result = await nlu.get_structured_intent(message)
    except Exception:
        nlu_result = {}

    data = nlu_result.get("data", {}) if isinstance(nlu_result, dict) else {}
    action = nlu_result.get("action") if isinstance(nlu_result, dict) else None
    title = (data.get("title") or "").strip() if data else ""
    # We'll try to extract a time phrase more robustly; default to message
    dt_text = (data.get("datetime") if data else None) or message

    # 2) Fallback heuristic if NLU didn't identify action
    if not action:
        lower = message.lower()
        if "remind me" in lower or lower.startswith("reminder"):
            action = "create_task"
            # Try to extract time phrase and title in common patterns
            # Examples:
            #   remind me in 10 minutes to drink water
            #   remind me at 5pm to call mom
            #   remind me on 12/10 to pay rent
            #   remind me to stretch in 30 minutes
            patterns = [
                r"remind me (?:in|after)\s+(.+?)\s+(?:to|about)\s+(.+)",
                r"remind me\s+(?:at|on|by)\s+(.+?)\s+(?:to|about)\s+(.+)",
                r"remind me\s+(?:to|about)\s+(.+?)\s+(?:in|at|on|by)\s+(.+)",
                r"remind me\s+(?:to|about)\s+(.+)",
            ]
            m = None
            for p in patterns:
                m = re.search(p, lower)
                if m:
                    break
            if m:
                if m.lastindex == 2:
                    # two groups: (time_phrase, title) or (title, time_phrase)
                    g1, g2 = m.group(1).strip(), m.group(2).strip()
                    # Heuristic: if g1 parses as date, treat g1=time, else swap
                    parsed_probe = None
                    try:
                        parsed_probe = dateparser.parse(
                            g1,
                            settings={
                                "PREFER_DATES_FROM": "future",
                            },
                        )
                    except Exception:
                        parsed_probe = None
                    if parsed_probe is not None:
                        dt_text = g1
                        if not title:
                            title = g2
                    else:
                        dt_text = g2
                        if not title:
                            title = g1
                else:
                    # single group likely title; keep dt_text as full message for search_dates below
                    if not title:
                        title = m.group(1).strip()
        else:
            return None  # Not a reminder-like message

    # Determine user timezone (fallback to UTC)
    profile = context.get("profile") or {}
    user_tz = (profile.get("timezone") if isinstance(profile, dict) else None) or "UTC"
    try:
        parsed_dt = dateparser.parse(
            dt_text,
            settings={
                "TIMEZONE": user_tz,
                "TO_TIMEZONE": "UTC",
                "RETURN_AS_TIMEZONE_AWARE": True,
                "PREFER_DATES_FROM": "future",
            },
        )
    except Exception:
        parsed_dt = None

    # If direct parse failed, search within the message for any date/time phrase
    if not parsed_dt:
        try:
            search = search_dates(
                message,
                settings={
                    "TIMEZONE": user_tz,
                    "TO_TIMEZONE": "UTC",
                    "RETURN_AS_TIMEZONE_AWARE": True,
                    "PREFER_DATES_FROM": "future",
                },
            )
        except Exception:
            search = None
        if search:
            # pick the first future time
            now_utc = datetime.utcnow().replace(tzinfo=None)
            for text_span, dt_val in search:
                dt_norm = _to_utc_naive(dt_val)
                if dt_norm and dt_norm > now_utc:
                    parsed_dt = dt_val
                    dt_text = text_span
                    # If title missing, try to remove the time phrase and leading 'remind me' + 'to/about'
                    if not title:
                        lowered = message.lower()
                        remainder = lowered.replace(text_span.lower(), "").strip()
                        # extract after 'to/about'
                        m2 = re.search(r"(?:to|about)\s+(.+)$", remainder)
                        if m2:
                            title = m2.group(1).strip()
                    break

    if not parsed_dt:
        try:
            logger.info("[Reminder Parse] No datetime found in message; skipping reminder creation")
        except Exception:
            pass
        return None  # don't create past/invalid reminders

    if parsed_dt <= datetime.utcnow():
        try:
            logger.info("[Reminder Parse] Parsed datetime is not in the future; skipping")
        except Exception:
            pass
        return None  # don't create past/invalid reminders

    eta_norm = _to_utc_naive(parsed_dt)
    if not title:
        lower = message.lower().strip()
        # Try better fallback to capture title portion
        m = re.search(r"remind me (?:in|after|at|on|by)\s+.+?\s+(?:to|about)\s+(.+)", lower)
        if not m:
            m = re.search(r"remind me (?:to|about)\s+(.+)", lower)
        title = (m.group(1).strip() if m else "Reminder").capitalize()

    # Duplicate check within ±60 seconds
    from datetime import timedelta as _td
    window_start = eta_norm - _td(seconds=60)
    window_end = eta_norm + _td(seconds=60)

    user_id_str = str(current_user.get("user_id") or current_user.get("_id"))
    try:
        dup = tasks.find_one({
            "user_id": user_id_str,
            "title": title,
            "due_date": {"$gte": window_start, "$lte": window_end},
        })
    except Exception:
        dup = None

    if dup:
        return None  # already have a near-duplicate scheduled

    # Create and schedule
    try:
        task_id = str(ObjectId())
        now = datetime.now(timezone.utc)
        doc = {
            "_id": task_id,
            "user_id": user_id_str,
            "title": title,
            "description": None,
            "priority": "medium",
            "due_date": eta_norm,
            "tags": [],
            "status": "pending",
            "created_at": now,
            "updated_at": now,
            "attempts": 0,
            "last_error": None,
        }
        tasks.insert_one(doc)
        celery_id = None
        try:
            logger.info(f"[Sessions] Created task {task_id} title='{title}' due={eta_norm} user={user_id_str}")
        except Exception:
            logger.warning(f"Failed to log task creation for {task_id}")
        # Schedule at the exact time using ETA for precision
        # Email reminder scheduling removed
        celery_id = None
        try:
            tasks.update_one({"_id": task_id}, {"$set": {"celery_task_id": None}})
            logger.info(f"[Sessions] Reminder scheduling via email disabled for task {task_id} at {eta_norm}")
        except Exception:
            logger.warning(f"Failed to update celery_task_id for {task_id}")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to create and schedule task: {e}")
        # Emit audit log into worker terminal
        try:
            to_email = current_user.get("email") or current_user.get("user_email")
            celery_app.send_task(
                "audit_log",
                args=["Task Scheduled"],
                kwargs={
                    "payload": {"task_id": task_id, "eta_utc": str(eta_norm), "to": to_email},
                    "source": "chat",
                    "immediate": False,
                },
            )
        except Exception:
            logger.warning(f"Failed to send audit log for {task_id}")
        # Confirmation email removed
        # Record in user profile (recent tasks + stats)
        try:
            profile_service.record_task_created(user_id_str, task_id=task_id, title=title, due_date=eta_norm)
        except Exception:
            logger.warning(f"Failed to record task creation for {task_id}")
        # Return creation details for UI confirmation
        return {"created": True, "task_id": task_id, "title": title, "eta_utc": eta_norm, "celery_id": celery_id}
        return None

# =====================================================
# 🔹 YouTube Video Intent Detection — strict auto-play extractor
# =====================================================
_VIDEO_PLAY_TRIGGERS = (
    "play", "watch", "show me", "open", "listen", "stream", "video", "vedio", "song", "music", "music video", "track", "trailer", "paly"
)

def _clean_video_query(user_text: str) -> str:
    """Extract a clean query string from a natural language request.

    Removes fillers like 'please', 'play', 'song', etc., collapses spaces, and returns
    the remaining title/keywords only. Always lower->cleanup->original case best effort.
    """
    text = (user_text or "").strip()
    low = text.lower()
    # quick exit if no trigger, but still may include a raw title
    fillers = [
        "please", "can you", "could you", "i want to", "show me", "play", "watch", "video", "vedio",
        "song", "music", "of", "from", "full", "video song", "official", "the"
    ]
    cleaned = low
    for f in fillers:
        cleaned = cleaned.replace(f, " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # If nothing remains, fallback to original minus leading verb
    if not cleaned:
        m = re.search(r"(?:play|watch)\s+(.+)$", low)
        cleaned = (m.group(1) if m else low).strip()
    # keep only sensible chunk length
    tokens = cleaned.split()
    if len(tokens) > 10:
        cleaned = " ".join(tokens[:10])
    # Return as title-cased-ish while preserving acronyms
    try:
        return " ".join(w.capitalize() if not re.fullmatch(r"[A-Z0-9_-]+", w) else w for w in cleaned.split())
    except Exception:
        return cleaned

def _make_yt_query(base: str) -> str:
    base = (base or "").strip()
    if not base:
        return ""
    # Encourage official/popular results
    return f"{base} official video"

def _extract_video_entities_and_confidence(message: str) -> dict:
    """Fast intent detection that always returns a query when a play intent is present.

    We do not ask for clarifications; the caller will auto-play top result.
    """
    text = (message or "").strip()
    low = text.lower()
    detected = any(t in low for t in _VIDEO_PLAY_TRIGGERS)
    if not detected:
        # Semantic-like patterns without exact keywords
        semantic_phrases = (
            "i want to see", "can i hear", "can you put on", "let's watch", "lets watch", "queue up",
            "bro,", "bro ", "open ", "that track", "that song"
        )
        if any(p in low for p in semantic_phrases):
            detected = True
    if not detected:
        return {"detected": False, "confidence": 0.0, "entities": {}, "query": None, "action": "none"}

    cleaned = _clean_video_query(text)
    # Minimal entity hints (language); optional for ranking
    lang = None
    if "hindi" in low:
        lang = "Hindi"
    elif "tamil" in low:
        lang = "Tamil"
    elif "telugu" in low:
        lang = "Telugu"
    query = _make_yt_query(cleaned)
    return {
        "detected": True,
        "confidence": 0.95,
        "entities": {"lang": lang} if lang else {},
        "query": query,
        "action": "play",
    }

# =====================================================
# 🔹 Pending Video Clarification Support
# =====================================================
async def _store_pending_video(session_id: str, det: dict):
    try:
        await redis_service.set_prefetched_data(f"session:video:pending:{session_id}", det, ttl_seconds=10 * 60)
    except Exception:
        pass

async def _clear_pending_video(session_id: str):
    try:
        # Expire quickly by setting to None
        await redis_service.set_prefetched_data(f"session:video:pending:{session_id}", None, ttl_seconds=1)
    except Exception:
        pass

async def _get_pending_video(session_id: str) -> Optional[dict]:
    try:
        return await redis_service.get_prefetched_data(f"session:video:pending:{session_id}")
    except Exception:
        return None

def _merge_video_entities(base: dict, incoming: dict) -> dict:
    b = (base or {}).copy()
    be = b.get("entities") or {}
    ie = (incoming or {}).get("entities") or {}
    # Prefer incoming clarifications if present
    for k in ("song", "movie", "artist", "lang"):
        if ie.get(k):
            be[k] = ie[k]
    # Recompute query from merged entities
    parts = []
    if be.get("song"):
        parts.append(be["song"])
    if be.get("movie"):
        parts.append(be["movie"])
    if be.get("artist") and not be.get("movie"):
        parts.append(be["artist"])
    if be.get("lang"):
        parts.append(be["lang"])
    b["entities"] = be
    b["query"] = " ".join(p for p in parts if p).strip() or (incoming.get("query") or b.get("query"))
    # Boost confidence slightly after clarification
    conf = max(float(b.get("confidence") or 0.0), float(incoming.get("confidence") or 0.0))
    b["confidence"] = min(1.0, round(conf + 0.15, 2))
    return b


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^\w]+", (text or "").lower()) if t]


def _text_relevance(query: str, title: str, description: str = "", lang: str | None = None) -> float:
    """Compute a lightweight cosine-like similarity between query and candidate text.

    Uses bag-of-words over alphanumeric tokens; fast and dependency-free.
    Returns a value in [0,1].
    """
    q_tokens = _tokenize(query)
    d_tokens = _tokenize(" ".join([title or "", description or ""]))
    if not q_tokens or not d_tokens:
        return 0.0
    from collections import Counter

    q_counts = Counter(q_tokens)
    d_counts = Counter(d_tokens)
    # dot product
    dot = sum(q_counts[t] * d_counts.get(t, 0) for t in q_counts)
    if dot == 0:
        return 0.0
    import math

    q_norm = math.sqrt(sum(v * v for v in q_counts.values()))
    d_norm = math.sqrt(sum(v * v for v in d_counts.values()))
    if q_norm == 0 or d_norm == 0:
        return 0.0
    score = dot / (q_norm * d_norm)
    # Light language alignment: if a language is specified and present in the title, nudge up
    if lang:
        lt = (lang or "").lower()
        if lt in (title or "").lower():
            score = min(1.0, score + 0.1)
    # clamp
    return max(0.0, min(1.0, float(score)))


def _detect_lang_by_script(text: str) -> str | None:
    """Very lightweight script-based heuristic for common Indian languages.

    - Devanagari (Hindi/Marathi/etc.): U+0900–U+097F
    - Tamil: U+0B80–U+0BFF
    - Telugu: U+0C00–U+0C7F
    """
    if not text:
        return None
    try:
        for ch in text:
            code = ord(ch)
            if 0x0900 <= code <= 0x097F:
                return "Hindi"
            if 0x0B80 <= code <= 0x0BFF:
                return "Tamil"
            if 0x0C00 <= code <= 0x0C7F:
                return "Telugu"
    except Exception:
        return None
    return None


def _script_bonus_for_lang(title: str, lang: str | None) -> float:
    if not lang or not title:
        return 0.0
    L = lang.lower()
    try:
        if L == "hindi":
            return 0.1 if any(0x0900 <= ord(ch) <= 0x097F for ch in title) else 0.0
        if L == "tamil":
            return 0.1 if any(0x0B80 <= ord(ch) <= 0x0BFF for ch in title) else 0.0
        if L == "telugu":
            return 0.1 if any(0x0C00 <= ord(ch) <= 0x0C7F for ch in title) else 0.0
    except Exception:
        return 0.0
    return 0.0


def _official_channel_score(channel_title: str, entities: Optional[dict] = None) -> int:
    ch = (channel_title or "").lower()
    official_channels = {
        "t-series",
        "sony music india",
        "sonymusicindiavevo",
        "zee music company",
        "yrf",
        "tips official",
        "saregama",
        "vevo",
        "aditya music",
        "think music india",
        "lahari music",
    }
    if any(name in ch for name in official_channels):
        return 1
    # Heuristic: channels containing "official", "vevo", or "music" are likely official
    if any(tag in ch for tag in ("official", "vevo", "music")):
        return 1
    # If entities are provided, boost if channel contains artist or movie tokens
    if entities:
        for k in ("artist", "movie", "song"):
            val = (entities.get(k) or "").lower()
            if val and any(tok in ch for tok in _tokenize(val)):
                return 1
    return 0


def _recency_bonus(published_at: str) -> float:
    """Map recency to a small bonus in [0, 0.1]. Newer -> closer to 0.1."""
    if not published_at:
        return 0.0
    from datetime import datetime, timezone
    try:
        # ISO 8601, often ends with 'Z'
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        days = max(0.0, (now - dt).total_seconds() / 86400.0)
        # 0 days -> 0.1, 3650 days (~10 years) -> ~0.0
        return max(0.0, 0.1 * (1.0 - min(days / 3650.0, 1.0)))
    except Exception:
        return 0.0


async def _youtube_best(query: str, entities: Optional[dict] = None) -> Optional[dict]:
    api_key = (getattr(settings, "YOUTUBE_API_KEY", None) or os.getenv("YOUTUBE_API_KEY") or "").strip()
    if not api_key or not query:
        return None
    # Redis cache for best pick
    cache_key = f"yt:best:{query.strip().lower()}"
    try:
        cached = await redis_service.get_prefetched_data(cache_key)
        if cached and cached.get("videoId"):
            return cached
    except Exception:
        pass
    params = {
        "part": "snippet",
        "type": "video",
        "safeSearch": "strict",
        "maxResults": 10,
        "q": query,
        "key": api_key,
    }
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get("https://www.googleapis.com/youtube/v3/search", params=params)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items") or []
            if not items:
                return None
            vids = [((it.get("id") or {}).get("videoId") or "").strip() for it in items]
            vids = [v for v in vids if v]
            if not vids:
                return None
            v_params = {"part": "statistics,snippet", "id": ",".join(vids), "key": api_key}
            v_resp = await client.get("https://www.googleapis.com/youtube/v3/videos", params=v_params)
            stats_map: Dict[str, Dict[str, Any]] = {}
            v_items = []
            if v_resp.status_code == 200:
                vj = v_resp.json()
                v_items = vj.get("items") or []
                for v in v_items:
                    stats_map[v.get("id") or ""] = v.get("statistics") or {}
            # Weighted scoring combining relevance, official, views, and recency
            import math
            max_views = 0
            for v in v_items:
                try:
                    max_views = max(max_views, int((stats_map.get(v.get("id") or "") or {}).get("viewCount") or 0))
                except Exception:
                    pass

            def final_score(it: Dict[str, Any]) -> float:
                sn = (it.get("snippet") or {})
                vid = it.get("id") or ""
                ch_title = sn.get("channelTitle") or ""
                published_at = sn.get("publishedAt") or ""
                title = sn.get("title") or ""
                desc = sn.get("description") or ""
                try:
                    views = int((stats_map.get(vid) or {}).get("viewCount") or 0)
                except Exception:
                    views = 0
                relevance = _text_relevance(query, title, desc, (entities or {}).get("lang"))
                official = _official_channel_score(ch_title, entities)
                view_score = (views / max_views) if max_views > 0 else 0.0
                recency = _recency_bonus(published_at)  # 0..0.1
                # Penalize undesirable variants (covers/reactions/slowed/remix/teaser/shorts) for precision
                lower_title = (title or "").lower()
                penalty = 0.0
                bad_terms = [
                    "reaction", "teaser", "tease", "cover", "karaoke", "slowed", "reverb",
                    "8d", "nightcore", "remix", "fan made", "shorts", "status", "whatsapp status",
                ]
                if any(bt in lower_title for bt in bad_terms):
                    penalty += 0.2
                # Prefer "official" in title subtly
                bonus = 0.05 if "official" in lower_title else 0.0
                # Add script bonus when language and script match
                bonus += _script_bonus_for_lang(title, (entities or {}).get("lang"))
                base_score = 0.45 * relevance + 0.35 * official + 0.20 * view_score + recency
                return max(0.0, base_score + bonus - penalty)

            best = None
            best_s = -math.inf
            for it in v_items:
                s = final_score(it)
                if s > best_s:
                    best_s = s
                    best = it
            if not best:
                return None
            sn = best.get("snippet") or {}
            vid = best.get("id")
            thumb = (((sn.get("thumbnails") or {}).get("high") or (sn.get("thumbnails") or {}).get("medium") or (sn.get("thumbnails") or {}).get("default")) or {}).get("url", "")
            payload = {
                "videoId": vid,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "title": sn.get("title") or "",
                "channelTitle": sn.get("channelTitle") or "",
                "thumbnail": thumb,
                "statistics": {
                    "viewCount": (stats_map.get(vid) or {}).get("viewCount"),
                },
                "_score": round(best_s, 4),
            }
            # Cache selection for faster follow-ups
            try:
                await redis_service.set_prefetched_data(cache_key, payload, ttl_seconds=12 * 3600)
            except Exception:
                pass
            return payload
    except Exception:
        return None


async def _maybe_handle_video_intent(user_text: str) -> Optional[dict]:
    """If a play intent is present, auto-search and return top result or a friendly fallback.

    Never ask clarifying questions.
    """
    det = _extract_video_entities_and_confidence(user_text)
    if not det.get("detected"):
        return None
    best = await _youtube_best(det.get("query") or user_text, entities=det.get("entities"))
    if best and best.get("videoId"):
        # Generic response without revealing video name per requirements
        reply = "🎬 Playing your requested video"
        return {"handled": True, "response_text": reply, "video": best, "video_intent": det}
    return {"handled": True, "response_text": "Sorry, I couldn’t find that video.", "video": None, "video_intent": det}

# =====================================================
# 🔹 Video Controls Endpoint (Play/Pause/Replay/Next/Lyrics)
# =====================================================
@router.post("/video/control", response_model=VideoControlResponse)
async def video_control(
    body: VideoControlRequest,
    current_user: dict = Depends(get_current_active_user),
):
    action = (body.action or "").lower().strip()
    vid = (body.current_video_id or "").strip()
    title = (body.current_title or "").strip()
    # Play/Pause are UI-level toggles; we acknowledge without changing video.

    if action in {"play", "resume"}:
        return VideoControlResponse(response_text="▶️ Resuming playback")
    if action == "pause":
        return VideoControlResponse(response_text="⏸️ Paused. Click play to resume.")
    if action == "replay":
        # Client should reload the same iframe with autoplay=1
        return VideoControlResponse(response_text="🔁 Replaying from the start.", video={"videoId": vid} if vid else None)
    if action == "next":
        # Try to fetch a related video using our youtube router
        if not vid:
            return VideoControlResponse(response_text="I don't have a current video to advance from.")
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                base = os.getenv("BACKEND_BASE_URL") or "http://localhost:8000"
                r = await client.get(f"{base}/api/youtube/related", params={"video_id": vid, "max_results": 5})
                if r.status_code == 200:
                    data = r.json()
                    nxt = (data.get("top") or (data.get("items") or [None])[0])
                    if nxt and nxt.get("videoId"):
                        nn = nxt
                        return VideoControlResponse(response_text="⏭️ Playing next video", video={
                            "videoId": nn.get("videoId"),
                            "title": nn.get("title"),
                            "channelTitle": nn.get("channelTitle"),
                            "thumbnail": nn.get("thumbnail"),
                            "statistics": nn.get("statistics"),
                        })
                return VideoControlResponse(response_text="No suitable next video found. Want a similar track instead?")
        except Exception:
            return VideoControlResponse(response_text="Couldn't fetch the next video right now. Try again in a moment.")
    if action in {"lyrics", "show_lyrics"}:
        # Placeholder: integrate a lyrics API/service later. Attempt to use session cached current video as context when title missing.
        if not title and body.session_id:
            try:
                cur = await redis_service.get_prefetched_data(f"session:video:{body.session_id}")
            except Exception:
                cur = None
            if cur:
                title = cur.get("title") or title
                vid = cur.get("videoId") or vid
        sample = (
            f"💫 {title} Lyrics 💫\n"
            "Chaleya, mera ishq hai tu...\n"
            "Chaleya, mera junoon hai tu...\n"
            "...\n\n"
            "🎵 Singer: Arijit Singh | Composer: Anirudh"
        )
        return VideoControlResponse(response_text="Here are the lyrics:", lyrics=sample)
    return VideoControlResponse(response_text="Unknown control. Try play, pause, replay, next, or lyrics.")

# =====================================================
# 🔹 JSON AI Generation Endpoint (Resilient Orchestrator)
# =====================================================
@router.post("/generate-json", response_model=GenerateJSONResponse)
async def generate_json(
    body: GenerateRequest,
    current_user: dict = Depends(get_current_active_user),
):
    """Return unified JSON using multi-provider orchestrator.

    Always returns a stable JSON with either response+provider_used or an error string.
    """
    prompt = (body.prompt or "").strip()
    if not prompt:
        return GenerateJSONResponse(response=None, provider_used=None, error="prompt is required")
    # Soft cap input length for stability; reject extremely large inputs
    if len(prompt) > 8000:
        return GenerateJSONResponse(response=None, provider_used=None, error="prompt too long")

    try:
        result = await ai_get_response(prompt)
        # Use new response format: success, output, provider_used, error
        if result.get("success"):
            return GenerateJSONResponse(
                response=result.get("output"),
                provider_used=result.get("provider_used"),
                error=None,
            )
        else:
            return GenerateJSONResponse(
                response=None,
                provider_used=None,
                error=result.get("error") or "Sorry, all AI services are temporarily unavailable. Please try again in a few moments. If this persists, contact support."
            )
    except Exception as e:  # noqa: BLE001
        # Do not raise; return structured error
        try:
            logger.exception("/generate-json failed")
        except Exception:
            pass
        return GenerateJSONResponse(response=None, provider_used=None, error=str(e))

# =====================================================
# 🔹 Intent Detection (Simple Rule-Based)
# =====================================================
def _detect_intent_and_entities(message: str) -> dict:
    """
    Detects basic intents (e.g., planning trips) and extracts entities.
    Extendable to ML-based NLU in future.
    """
    trip_patterns = [
        r"(?:plan|organize|book|take)\s+(?:a\s+)?(?:trip|vacation|journey)\s+to\s+([\w\s]+)",
        r"(?:go|travel)\s+to\s+([\w\s]+)",
        r"let'?s\s+go\s+to\s+([\w\s]+)"
    ]
    for pattern in trip_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            destination = match.group(1).strip()
            return {"intent": "PLAN_TRIP", "entities": {"destination": destination}}
    return {"intent": "GENERAL_INQUIRY", "entities": {}}

# Removed duplicate _gather_context; unified with memory_coordinator.gather_memory_context


# =====================================================
# 🔹 Start New Chat Session
# =====================================================
async def handle_chat_message(
    message: str,
    user_id: str,
    session_id: str,
    current_user: dict,
    sessions_collection: Collection,
    tasks_collection: Collection,
    chat_log_collection: Collection,
):
    """
    Central handler for processing a chat message, including intent detection.
    """
    # 0. Immediate deterministic profile extraction (name, timezone, favorites, etc.)
    # This ensures cross-session recall (e.g., user name) before gathering context
    try:
        det = deterministic_extractor.extract(message, "")
        if det.get("profile_update"):
            try:
                profile_service.merge_update(user_id, **det["profile_update"])  # type: ignore[arg-type]
            except Exception:
                # Non-fatal; proceed without blocking the chat
                pass
    except Exception:
        pass
    # 1. Intent Detection for Reminders with confirmation flow
    if task_nlp.detect_task_intent(message):
        try:
            # Get user timezone from profile
            profile = context.get("profile") or {}
            user_tz = profile.get("timezone") if isinstance(profile, dict) else None
            
            # Handle task intent with Redis-backed confirmation flow
            flow_result = await task_flow_service.handle_task_intent(message, user_id, user_tz)
            
            if flow_result.get("action") == "created":
                # Task was created successfully
                task_data = flow_result.get("task_data", {})
                confirmation_message = flow_result["message"]
                # Log the creation
                asyncio.create_task(log_message(session_id, "user", message, chat_log_collection))
                asyncio.create_task(log_message(session_id, "assistant", confirmation_message, chat_log_collection))
                return confirmation_message
            elif flow_result.get("needs_response"):
                # Need clarification or confirmation
                clarification_message = flow_result["message"]
                # Log the clarification request
                asyncio.create_task(log_message(session_id, "user", message, chat_log_collection))
                asyncio.create_task(log_message(session_id, "assistant", clarification_message, chat_log_collection))
                return clarification_message
        except Exception as e:
            logger.exception(f"Task flow handling failed: {e}")
            # Fall through to general conversation

    # 2. Fallback to General Conversation
    # (The existing logic for handling general chat would go here)
    # For now, just echoing back for demonstration
    
    try:
        context = await gather_memory_context(
            user_id=user_id,
            user_key=current_user.get("email", user_id),
            session_id=session_id,
            latest_user_message=message,
            recent_messages=[],  # In a real scenario, you'd fetch recent messages
        )
    except Exception as _e:  # noqa: BLE001
        try:
            logger = logging.getLogger(__name__)
            logger.warning("gather_memory_context failed; proceeding with minimal context")
        except Exception:
            pass
        context = {
            "history": [],
            "state": "general_conversation",
            "pinecone_context": None,
            "neo4j_facts": None,
            "profile": {},
            "user_facts_semantic": None,
            "persistent_memories": None,
        }

    # Generate assistant response with a resilient fallback to avoid surfacing 500s
    try:
        ai_response_text = await ai_get_response(
            prompt=message,
            history=context.get("history"),
            state=context.get("state", "general_conversation"),
            pinecone_context=context.get("pinecone_context"),
            neo4j_facts=context.get("neo4j_facts"),
            profile=context.get("profile"),
            user_facts_semantic=context.get("user_facts_semantic"),
            persistent_memories=context.get("persistent_memories"),
        )
    except Exception as _e:  # noqa: BLE001
        # Soft-fail so the endpoint still returns 200 with a friendly message
        try:
            logger = logging.getLogger(__name__)
            logger.exception("AI provider error during get_response; returning fallback text")
        except Exception:
            pass
        ai_response_text = (
            "I'm having trouble connecting to the AI service right now. "
            "Let's try again in a moment."
        )

    asyncio.create_task(log_message(session_id, "user", message, chat_log_collection))
    asyncio.create_task(log_message(session_id, "assistant", ai_response_text, chat_log_collection))

    # Unified session history logging (memory_coordinator)
    try:
        from app.services.memory_coordinator import _append_history
        await _append_history(session_id, message, ai_response_text)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.debug(f"Failed to append to session history (non-fatal): {e}")

    post_message_update(
        user_id=user_id,
        user_key=current_user.get("email", user_id),
        session_id=session_id,
        user_message=message,
        ai_message=ai_response_text,
        state="general_conversation",
    )

    return ai_response_text

async def log_message(session_id: str, sender: str, text: str, collection: Collection):
    """Logs a message to the chat log collection."""
    try:
        await run_in_threadpool(
            collection.insert_one,
            {
                "session_id": session_id,
                "sender": sender,
                "text": text,
                "timestamp": datetime.utcnow(),
            },
        )
    except Exception:
        # Do not surface logging failures to client; keep conversation flowing
        try:
            logger = logging.getLogger(__name__)
            logger.debug("chat log insert failed", exc_info=True)
        except Exception:
            pass

@router.post("/new", response_model=NewChatResponse)
async def start_new_chat(
    request: Union[NewChatRequest, ChatRequest],
    current_user: dict = Depends(get_current_active_user),
    sessions: Collection = Depends(get_sessions_collection),
    tasks: Collection = Depends(get_tasks_collection),
    chat_log: Collection = Depends(get_chat_log_collection),
    activity_logs: Collection = Depends(get_activity_logs_collection),
):
    """
    Starts a new chat session.
    """
    user_id = str(current_user["_id"])
    
    # Create a new session
    now = datetime.utcnow()
    session_data = {
        "userId": user_id,
        "title": "New Chat",
        "createdAt": now,
        "updatedAt": now,
        "lastMessageAt": None,
        "lastMessage": "",
        "messageCount": 0,
        "pinned": False,
        "saved": False,
    }
    result = await run_in_threadpool(sessions.insert_one, session_data)
    session_id = str(result.inserted_id)

    # Normalize request body (compat layer)
    req_message: str = request.message  # type: ignore[attr-defined]
    # Fire-and-forget NLU (non-blocking) and chain Brain planning
    nlu_future = None
    try:
        nlu_future = asyncio.create_task(nlu.extract_intent_entities(message=req_message, session_id=session_id, user_id=user_id))
        # also log NLU in background
        asyncio.create_task(_nlu_and_log(activity_logs, user_id, session_id, req_message))
    except Exception:
        nlu_future = None

    brain_future = None
    try:
        async def _chain_pipeline():
            nlu_result = None
            try:
                if nlu_future is not None:
                    nlu_result = await nlu_future
            except Exception:
                nlu_result = None
            return await _brain_nlg_pipeline(
                user_id=user_id,
                session_id=session_id,
                message=req_message,
                nlu_result=nlu_result,
                include_in_response=DEBUG_BRAIN,
            )
        brain_future = asyncio.create_task(_chain_pipeline())
    except Exception:
        brain_future = None

    # Try fast-path video handling first
    video_payload = None
    video_intent_meta = None
    try:
        vi = await _maybe_handle_video_intent(req_message)
        if vi and vi.get("handled"):
            asyncio.create_task(log_message(session_id, "user", req_message, chat_log))
            resp_txt = vi.get("response_text") or ""
            asyncio.create_task(log_message(session_id, "assistant", resp_txt, chat_log))
            # Persist into session messages for history
            try:
                now2 = datetime.utcnow()
                user_doc2 = {"_id": ObjectId(), "sender": "user", "text": req_message, "timestamp": now2}
                ai_doc2 = {"_id": ObjectId(), "sender": "assistant", "text": resp_txt, "timestamp": now2}
                await run_in_threadpool(
                    sessions.update_one,
                    {"_id": ObjectId(session_id)},
                    {
                        "$push": {"messages": {"$each": [user_doc2, ai_doc2]}},
                        "$set": {"updatedAt": now2, "lastMessageAt": now2, "lastMessage": resp_txt[:200]},
                        "$inc": {"messageCount": 2},
                    },
                )
            except Exception:
                pass
            # If we asked for clarification, store pending intent for this session
            if vi.get("video") is None and vi.get("video_intent"):
                try:
                    await _store_pending_video(session_id, vi.get("video_intent"))
                except Exception:
                    pass
            # Remember current video for this session (used by controls like Next/Lyrics)
            try:
                if vi.get("video") and vi["video"].get("videoId"):
                    await redis_service.set_prefetched_data(f"session:video:{session_id}", vi["video"], ttl_seconds=24 * 3600)
            except Exception:
                pass
            # Update session metadata so it appears on top in history
            try:
                await run_in_threadpool(
                    sessions.update_one,
                    {"_id": ObjectId(session_id)},
                    {"$set": {"updatedAt": datetime.utcnow(), "lastMessageAt": datetime.utcnow(), "lastMessage": resp_txt[:200]}},
                )
            except Exception:
                pass
            # Persist selection in activity logs for preference learning
            try:
                vid = (vi.get("video") or {}).get("videoId")
                if vid:
                    activity_logs.insert_one({
                        "type": "youtube_play",
                        "user_id": str(current_user.get("_id")),
                        "session_id": session_id,
                        "user_query": req_message,
                        "matched_title": (vi.get("video") or {}).get("title"),
                        "video_id": vid,
                        "timestamp": datetime.utcnow(),
                    })
            except Exception:
                pass

            return NewChatResponse(
                session_id=session_id,
                response_text=resp_txt,
                video={**vi.get("video", {}), "autoplay": True} if vi.get("video") else None,
                video_intent=vi.get("video_intent"),
                ai_message_id=str(ai_doc2["_id"]) if isinstance(ai_doc2.get("_id"), ObjectId) else None,
                user_message_id=str(user_doc2["_id"]) if isinstance(user_doc2.get("_id"), ObjectId) else None,
            )
    except Exception:
        pass

    # Before deeper handling, update user's profile from the first message if applicable
    try:
        det0 = deterministic_extractor.extract(request.message, "")
        if det0.get("profile_update"):
            profile_service.merge_update(user_id, **det0["profile_update"])  # type: ignore[arg-type]
    except Exception:
        pass

    response_text = await handle_chat_message(
        request.message, user_id, session_id, current_user, sessions, tasks, chat_log
    )

    # Update session document with last message metadata (and messageCount approx); also set smart title
    try:
        smart_title = _auto_title_from_first_message(request.message)
        await run_in_threadpool(
            sessions.update_one,
            {"_id": ObjectId(session_id)},
            {"$set": {"updatedAt": datetime.utcnow(), "lastMessageAt": datetime.utcnow(), "lastMessage": response_text[:200], "title": smart_title}, "$inc": {"messageCount": 2}}
        )
    except Exception:
        pass

    # Try to fetch the last two messages to return IDs
    user_mid = None
    ai_mid = None
    try:
        snap = await run_in_threadpool(
            sessions.find_one,
            {"_id": ObjectId(session_id)},
            {"messages": {"$slice": -2}},
        )
        msgs = (snap or {}).get("messages", [])
        if len(msgs) == 2:
            # Expect order [older, newer] depending on slice; normalize by timestamps
            m_sorted = sorted(msgs, key=lambda m: m.get("timestamp") or datetime.utcnow())
            # First should be user, second assistant (best effort)
            if m_sorted[0].get("sender") == "user":
                user_mid = m_sorted[0].get("_id")
            if m_sorted[1].get("sender") == "assistant":
                ai_mid = m_sorted[1].get("_id")
    except Exception:
        pass

    if DEBUG_BRAIN and brain_future is not None and brain_future.done():
        try:
            dbg = brain_future.result()
            if dbg:
                if isinstance(dbg.get("plan"), dict):
                    response_text = (response_text or "") + "\n\n[debug plan]\n" + json.dumps(dbg["plan"], ensure_ascii=False)[:1500]
                if isinstance(dbg.get("nlg"), dict):
                    response_text = (response_text or "") + "\n\n[debug NLG]\n" + json.dumps(dbg["nlg"], ensure_ascii=False)[:600]
        except Exception:
            pass

    return NewChatResponse(
        session_id=session_id,
        response_text=response_text,
        ai_message_id=str(ai_mid) if ai_mid else None,
        user_message_id=str(user_mid) if user_mid else None,
    )

@router.post("/{session_id}/continue", response_model=ContinueChatResponse)
async def continue_chat(
    session_id: str,
    request: ContinueChatRequest,
    current_user: dict = Depends(get_current_active_user),
    sessions: Collection = Depends(get_sessions_collection),
    tasks: Collection = Depends(get_tasks_collection),
    chat_log: Collection = Depends(get_chat_log_collection),
    activity_logs: Collection = Depends(get_activity_logs_collection),
):
    """
    Continues an existing chat session.
    """
    user_id = str(current_user["_id"])

    # Verify session exists and belongs to the user (support ObjectId or string userId)
    try:
        match = {"_id": ObjectId(session_id), "$or": [{"userId": user_id}]}
        if ObjectId.is_valid(user_id):
            match["$or"].append({"userId": ObjectId(user_id)})
        session = await run_in_threadpool(sessions.find_one, match)
    except Exception:
        session = None
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    # Title auto-generation for first message in an empty/newly created session
    try:
        cur_title = (session.get("title") or "").strip()
        if not cur_title or cur_title.lower() == "new chat":
            new_title = _auto_title_from_first_message(request.message)
            await run_in_threadpool(
                sessions.update_one,
                {"_id": ObjectId(session_id)},
                {"$set": {"title": new_title, "updatedAt": datetime.utcnow()}},
            )
    except Exception:
        pass

    # First, attempt fast-path video handling or resolution of pending clarification
    try:
        # Text-based replay of last video
        t_low = (request.message or "").lower().strip()
        if any(k in t_low for k in ["play that again", "replay", "again", "repeat", "once more"]):
            try:
                cur = await redis_service.get_prefetched_data(f"session:video:{session_id}")
            except Exception:
                cur = None
            if cur and cur.get("videoId"):
                req_message = request.message if hasattr(request, 'message') else ""
                asyncio.create_task(log_message(session_id, "user", req_message, chat_log))
                asyncio.create_task(log_message(session_id, "assistant", "🔁 Replaying your requested video", chat_log))
                return ContinueChatResponse(response_text="🔁 Replaying your requested video", video={**cur, "autoplay": True})

        # If there's a pending clarification, merge with new message
        pending = await _get_pending_video(session_id)
        if pending and isinstance(pending, dict):
            inc = _extract_video_entities_and_confidence(request.message)
            if inc.get("detected"):
                merged = _merge_video_entities(pending, inc)
                if merged.get("confidence", 0.0) >= 0.75:
                    # Good enough to attempt playback now
                    best = await _youtube_best(merged.get("query") or request.message, entities=merged.get("entities"))
                    if best and best.get("videoId"):
                        title = best.get("title") or merged.get("query") or "the video"
                        channel = best.get("channelTitle") or ""
                        ch_str = f" · {channel}" if channel else ""
                        reply_txt = f"▶️ Playing {title}{ch_str}"
                        await _clear_pending_video(session_id)
                        # Log messages
                        req_message = request.message if hasattr(request, 'message') else ""
                        asyncio.create_task(log_message(session_id, "user", req_message, chat_log))
                        asyncio.create_task(log_message(session_id, "assistant", reply_txt, chat_log))
                        try:
                            await redis_service.set_prefetched_data(f"session:video:{session_id}", best, ttl_seconds=24 * 3600)
                        except Exception:
                            pass
                        return ContinueChatResponse(response_text=reply_txt, video={**best, "autoplay": True}, video_intent=merged)
            # If still unclear, keep asking—a more specific prompt
            ask = "🤔 I can play it—please confirm the exact song or language (e.g., 'Chaleya Hindi')."
            req_message = request.message if hasattr(request, 'message') else ""
            asyncio.create_task(log_message(session_id, "user", req_message, chat_log))
            asyncio.create_task(log_message(session_id, "assistant", ask, chat_log))
            return ContinueChatResponse(response_text=ask, video=None, video_intent=pending)

        # No pending—try normal fast path
        vi = await _maybe_handle_video_intent(request.message)
        if vi and vi.get("handled"):
            req_message = request.message if hasattr(request, 'message') else ""
            asyncio.create_task(log_message(session_id, "user", req_message, chat_log))
            asyncio.create_task(log_message(session_id, "assistant", vi.get("response_text") or "", chat_log))
            if vi.get("video") is None and vi.get("video_intent"):
                try:
                    await _store_pending_video(session_id, vi.get("video_intent"))
                except Exception:
                    pass
            else:
                try:
                    if vi.get("video") and vi["video"].get("videoId"):
                        await redis_service.set_prefetched_data(f"session:video:{session_id}", vi["video"], ttl_seconds=24 * 3600)
                except Exception:
                    pass
            # Log selection for learning
            try:
                vbest = vi.get("video") or {}
                if vbest.get("videoId"):
                    activity_logs.insert_one({
                        "type": "youtube_play",
                        "user_id": str(current_user.get("_id")),
                        "session_id": session_id,
                        "user_query": request.message,
                        "matched_title": vbest.get("title"),
                        "video_id": vbest.get("videoId"),
                        "timestamp": datetime.utcnow(),
                    })
            except Exception:
                pass
            return ContinueChatResponse(response_text=vi.get("response_text") or "", video={**(vi.get("video") or {}), "autoplay": True} if vi.get("video") else None, video_intent=vi.get("video_intent"))
    except Exception:
        pass

    response_text = await handle_chat_message(
        request.message, user_id, session_id, current_user, sessions, tasks, chat_log
    )
    
    # Update session's updatedAt and last message info so it bubbles to top; increment count
    try:
        await run_in_threadpool(
            sessions.update_one,
            {"_id": ObjectId(session_id)},
            {"$set": {"updatedAt": datetime.utcnow(), "lastMessageAt": datetime.utcnow(), "lastMessage": response_text[:200]}, "$inc": {"messageCount": 2}},
        )
    except Exception:
        pass

    # Attempt to fetch the last assistant message id
    ai_mid2 = None
    user_mid2 = None
    try:
        snap2 = await run_in_threadpool(
            sessions.find_one,
            {"_id": ObjectId(session_id)},
            {"messages": {"$slice": -2}},
        )
        msgs2 = (snap2 or {}).get("messages", [])
        for m in msgs2:
            if m.get("sender") == "assistant" and m.get("_id"):
                ai_mid2 = m.get("_id")
            if m.get("sender") == "user" and m.get("_id"):
                user_mid2 = m.get("_id")
    except Exception:
        pass
    return ContinueChatResponse(
        response_text=response_text,
        ai_message_id=str(ai_mid2) if ai_mid2 else None,
        user_message_id=str(user_mid2) if user_mid2 else None,
    )

# =====================================================
# 🔹 Start New Chat Session (Streaming)
# =====================================================
@router.post("/new/stream")
async def start_new_chat_stream(
    request: NewChatRequest,
    current_user: dict = Depends(get_current_active_user),
    sessions: Collection = Depends(get_sessions_collection),
    tasks: Collection = Depends(get_tasks_collection),
):
    """
    Starts a new chat session and streams the assistant's response token-by-token.
    Note: This uses a simulated streaming approach by chunking the final text.
    """
    user_id = str(current_user["_id"])
    await neo4j_service.create_user_node(user_id)

    # Inline deterministic extraction for immediate profile availability
    try:
        det = deterministic_extractor.extract(request.message, "")
        if det.get("profile_update"):
            profile_service.merge_update(user_id, **det["profile_update"])  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        pass

    context = await gather_memory_context(
        user_id=user_id,
        user_key=current_user.get("email", user_id),
        session_id="new",
        latest_user_message=request.message,
        recent_messages=[],
    )

    # Generate AI response in background thread
    ai_response_text = await ai_get_response(
        prompt=request.message,
        history=context.get("history"),
        state=context.get("state", "general_conversation"),
        pinecone_context=context.get("pinecone_context"),
        neo4j_facts=context.get("neo4j_facts"),
        profile=context.get("profile"),
        user_facts_semantic=context.get("user_facts_semantic"),
        persistent_memories=context.get("persistent_memories"),
    )

    # Save session in MongoDB
    user_message = {"_id": ObjectId(), "sender": "user", "text": request.message}
    ai_message = {
        "_id": ObjectId(),
        "sender": "assistant",
        "text": ai_response_text,
        "annotatedHtml": None,
        "highlights": [],
    }
    session_data = {
        "userId": current_user["_id"],
        "title": request.message[:50],
        "createdAt": datetime.utcnow(),
        "lastUpdatedAt": datetime.utcnow(),
        "isArchived": False,
        "messages": [user_message, ai_message],
    }
    result = await run_in_threadpool(sessions.insert_one, session_data)
    session_id = str(result.inserted_id)

    await redis_service.set_session_state(session_id, "general_conversation")

    # Post message update (after full response assembled for streaming)
    post_message_update(
        user_id=user_id,
        user_key=current_user.get("email", user_id),
        session_id=session_id,
        user_message=request.message,
        ai_message=ai_response_text,
        state="general_conversation",
    )
    try:
        from app.services import memory_service as _memsvc
        mem_ids = [m.get("id") for m in (context.get("persistent_memories") or []) if m.get("id")]
        if mem_ids:
            import asyncio as _asyncio
            _asyncio.create_task(_memsvc.touch_memory_access(user_id, mem_ids))
    except Exception:
        pass

    async def token_stream():
        # Simple word-chunk streaming; adjust chunking as needed
        words = ai_response_text.split()
        chunk = []
        for i, w in enumerate(words, 1):
            chunk.append(w)
            # Stream every ~10 words to reduce overhead
            if i % 10 == 0:
                yield " ".join(chunk) + " "
                chunk = []
                await asyncio.sleep(0)  # yield control
        if chunk:
            yield " ".join(chunk)

    # Try to create/schedule a reminder and append confirmation to the reply
    try:
        _res = await _maybe_create_and_schedule_reminder(request.message, current_user, tasks, context)
        if _res and _res.get("created"):
            eta = _res.get("eta_utc")
            profile = context.get("profile") or {}
            user_tz = (profile.get("timezone") if isinstance(profile, dict) else None) or "UTC"
            time_pref = (profile.get("time_format") if isinstance(profile, dict) else None)
            when = _format_eta_for_user(eta, user_tz, time_pref) if isinstance(eta, datetime) else str(eta)
            title = _res.get("title") or "Reminder"
            ai_response_text = (ai_response_text or "").strip()
            suffix = f"\n\nI've set a reminder: {title} at {when}. You'll get an email then."
            ai_response_text = (ai_response_text + suffix) if ai_response_text else suffix.strip()
    except Exception:
        pass

    headers = {"X-Session-Id": session_id}
    return StreamingResponse(token_stream(), media_type="text/plain", headers=headers)


# =====================================================
# 🔹 Continue Existing Chat Session
# =====================================================
@router.post("/{session_id}", response_model=ContinueChatResponse)
async def continue_chat(
    session_id: str,
    request: Union[ContinueChatRequest, ChatRequest],
    current_user: dict = Depends(get_current_active_user),
    sessions: Collection = Depends(get_sessions_collection),
    tasks: Collection = Depends(get_tasks_collection),
):
    """
    Continues a chat session with full context:
    - Short-term history (last 10 messages, fetched efficiently)
    - Redis session state
    - Pinecone & Neo4j memory (gathered concurrently)
    - Intent detection & proactive prefetch tasks
    - Background fact extraction
    """
    user_id = str(current_user["_id"])

    # Validate session ID
    try:
        session_obj_id = ObjectId(session_id)
    except errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid session ID format.")

    # Fetch session and only the last 10 messages for efficiency
    session = await run_in_threadpool(
        sessions.find_one,
        {"_id": session_obj_id, "userId": current_user["_id"]},
        {"messages": {"$slice": -10}},
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Gather state and context concurrently
    # Normalize request body (compat layer)
    req_message: str = request.message  # type: ignore[attr-defined]

    # Chain NLU → Brain + MemoryManager → (optional) NLG in background (non-blocking)
    try:
        async def _chain_brain():
            nlu_result = None
            try:
                nlu_result = await nlu.extract_intent_entities(message=req_message, session_id=session_id, user_id=user_id)
            except Exception:
                nlu_result = None
            return await _brain_nlg_pipeline(
                user_id=user_id,
                session_id=session_id,
                message=req_message,
                nlu_result=nlu_result,
                include_in_response=False,
            )
        asyncio.create_task(_chain_brain())
    except Exception:
        pass

    context = await gather_memory_context(
        user_id=user_id,
        user_key=current_user.get("email", user_id),
        session_id=session_id,
        latest_user_message=req_message,
        recent_messages=session.get("messages", []),
    )
    current_state = context.get("state", "general_conversation")
    recent_history = context.get("history", [])

    # If in planning mode, check Redis for prefetched data
    prefetched_context = None
    if current_state == "planning_trip":
        cache_key = f"prefetched_info:{session_id}"
        prefetched_data = await redis_service.get_prefetched_data(cache_key)
        if prefetched_data:
            prefetched_context = f"Here is some relevant information: {json.dumps(prefetched_data)}"

    # Merge contexts
    pinecone_context = context.get("pinecone_context")
    if prefetched_context:
        pinecone_context = f"{pinecone_context}\n{prefetched_context}"

    # Placeholder; may be overridden by fast-path or full model call below
    ai_response_text = ""

    # Rebuild with deterministic fast-path (override above placeholder logic)
    profile = context.get("profile") or {}
    text_lower = request.message.lower().strip()
    name = profile.get("name") if isinstance(profile, dict) else None
    fast_answer = None
    import re as _re

    # Name queries
    if any(p in text_lower for p in ["what's my name", "whats my name", "do you know my name", "do u know my name", "tell me my name"]):
        if name:
            fast_answer = ("Yes, your name is " + name + ".") if "do" in text_lower else f"Your name is {name}."
        else:
            fast_answer = "I don't have your name yet. You can tell me and I'll remember it."

    # Cuisine question
    if fast_answer is None and ("what cuisine do i like" in text_lower or "my favorite cuisine" in text_lower or "favorite cuisine" in text_lower):
        favs = profile.get("favorites") or {}
        cuisine = favs.get("cuisine") or favs.get("food")
        fast_answer = f"You enjoy {cuisine} cuisine." if cuisine else "You haven't told me your favorite cuisine yet."

    # General summary
    if fast_answer is None and any(p in text_lower for p in ["what do you know about me", "what do u know about me", "what do you know of me"]):
        favs = profile.get("favorites") or {}
        cuisine = favs.get("cuisine") or favs.get("food")
        hobbies = profile.get("hobbies") or []
        pieces = []
        if name and cuisine:
            pieces.append(f"I know your name is {name} and you enjoy {cuisine} cuisine")
        elif name:
            pieces.append(f"I know your name is {name}")
        elif cuisine:
            pieces.append(f"You enjoy {cuisine} cuisine")
        if hobbies:
            pieces.append("you like " + (", ".join(hobbies[:2]) if len(hobbies) > 1 else hobbies[0]))
        if not pieces:
            fast_answer = "I don't have personal details yet. You can share them and I'll remember."
        else:
            summary = " ".join(pieces).strip()
            if not summary.endswith('.'):
                summary += '.'
            fast_answer = summary

    # Favorites generic
    if fast_answer is None and any(p in text_lower for p in ["what are my favorites", "my favorites?", "do you know my favorites"]):
        favs = profile.get("favorites") or {}
        if favs:
            limited = list(favs.items())[:3]
            fav_str = ", ".join(f"{k}={v}" for k, v in limited)
            fast_answer = f"Your favorites I know: {fav_str}."
        else:
            fast_answer = "I don't have any favorites stored yet."

    if fast_answer is None:
        # Fall back to full model call
        ai_response_text = await ai_get_response(
            prompt=request.message,
            history=recent_history,
            state=current_state,
            pinecone_context=pinecone_context,
            neo4j_facts=context.get("neo4j_facts"),
            profile=profile,
            user_facts_semantic=context.get("user_facts_semantic"),
            persistent_memories=context.get("persistent_memories"),
        )
    else:
        ai_response_text = fast_answer

    # Intent detection
    detected = _detect_intent_and_entities(request.message)
    next_state = current_state
    if detected["intent"] == "PLAN_TRIP":
        next_state = "planning_trip"
        destination = detected["entities"].get("destination")
        if destination:
            celery_app.send_task(
                "prefetch_destination_info",
                args=[destination, session_id],
            )

    # Update Redis state
    # (state persisted via post_message_update below)

    # Video fast-path (after composing initial answer): attempt to play or ask clarification
    video_payload = None
    video_intent_meta = None
    try:
        vi = await _maybe_handle_video_intent(request.message)
        if vi and vi.get("handled"):
            ai_response_text = vi.get("response_text") or ai_response_text
            video_payload = vi.get("video")
            video_intent_meta = vi.get("video_intent")
    except Exception:
        pass

    # Save new messages with stable subdocument IDs and annotation fields
    user_message = {"_id": ObjectId(), "sender": "user", "text": request.message}
    ai_message = {
        "_id": ObjectId(),
        "sender": "assistant",
        "text": ai_response_text,
        # initialize annotation fields for highlighting feature
        "annotatedHtml": None,
        "highlights": [],
    }
    update_result = await run_in_threadpool(
        sessions.update_one,
        {"_id": session_obj_id},
        {
            "$push": {"messages": {"$each": [user_message, ai_message]}},
            "$set": {"lastUpdatedAt": datetime.utcnow()},
        },
    )

    # Unified post message update (embeddings + gated extraction + state)
    post_message_update(
        user_id=user_id,
        user_key=current_user.get("email", user_id),
        session_id=session_id,
        user_message=request.message,
        ai_message=ai_response_text,
        state=next_state,
    )

    # Try to create/schedule a reminder and append confirmation
    try:
        _res = await _maybe_create_and_schedule_reminder(request.message, current_user, tasks, context)
        if _res and _res.get("created"):
            eta = _res.get("eta_utc")
            profile = context.get("profile") or {}
            user_tz = (profile.get("timezone") if isinstance(profile, dict) else None) or "UTC"
            time_pref = (profile.get("time_format") if isinstance(profile, dict) else None)
            when = _format_eta_for_user(eta, user_tz, time_pref) if isinstance(eta, datetime) else str(eta)
            title = _res.get("title") or "Reminder"
            ai_response_text = (ai_response_text or "").strip()
            suffix = f"\n\nI've set a reminder: {title} at {when}. You'll get an email then."
            ai_response_text = (ai_response_text + suffix) if ai_response_text else suffix.strip()
    except Exception:
        pass

    try:
        from app.services import memory_service as _memsvc
        mem_ids = [m.get("id") for m in (context.get("persistent_memories") or []) if m.get("id")]
        if mem_ids:
            import asyncio as _asyncio
            _asyncio.create_task(_memsvc.touch_memory_access(user_id, mem_ids))
    except Exception:
        pass

    result = {"response_text": ai_response_text, "ai_message_id": str(ai_message["_id"]) }
    if video_payload is not None:
        result["video"] = video_payload
    if video_intent_meta is not None:
        result["video_intent"] = video_intent_meta
    # Remember current video for this session if provided by fast-path
    try:
        if video_payload and video_payload.get("videoId"):
            await redis_service.set_prefetched_data(f"session:video:{session_id}", video_payload, ttl_seconds=24 * 3600)
    except Exception:
        pass
    return result


# =====================================================
# 🔹 Continue Chat (Streaming)
# =====================================================
@router.post("/{session_id}/stream")
async def continue_chat_stream(
    session_id: str,
    request: Union[ContinueChatRequest, ChatRequest],
    current_user: dict = Depends(get_current_active_user),
    sessions: Collection = Depends(get_sessions_collection),
    tasks: Collection = Depends(get_tasks_collection),
):
    """
    Continues a chat session and streams the assistant's response token-by-token.
    Uses simulated streaming by chunking the generated text.
    """
    user_id = str(current_user["_id"])

    # Validate session ID
    try:
        session_obj_id = ObjectId(session_id)
    except errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid session ID format.")

    session = await run_in_threadpool(
        sessions.find_one,
        {"_id": session_obj_id, "userId": current_user["_id"]},
        {"messages": {"$slice": -10}},
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Normalize request body (compat layer)
    req_message: str = request.message  # type: ignore[attr-defined]

    # Chain NLU → Brain + MemoryManager → (optional) NLG in background (non-blocking)
    try:
        async def _chain_brain():
            nlu_result = None
            try:
                nlu_result = await nlu.extract_intent_entities(message=req_message, session_id=session_id, user_id=user_id)
            except Exception:
                nlu_result = None
            return await _brain_nlg_pipeline(
                user_id=user_id,
                session_id=session_id,
                message=req_message,
                nlu_result=nlu_result,
                include_in_response=False,
            )
        asyncio.create_task(_chain_brain())
    except Exception:
        pass

    context = await gather_memory_context(
        user_id=user_id,
        user_key=current_user.get("email", user_id),
        session_id=session_id,
        latest_user_message=req_message,
        recent_messages=session.get("messages", []),
    )
    current_state = context.get("state", "general_conversation")
    recent_history = context.get("history", [])

    prefetched_context = None
    if current_state == "planning_trip":
        cache_key = f"prefetched_info:{session_id}"
        prefetched_data = await redis_service.get_prefetched_data(cache_key)
        if prefetched_data:
            prefetched_context = f"Here is some relevant information: {json.dumps(prefetched_data)}"

    pinecone_context = context.get("pinecone_context")
    if prefetched_context:
        pinecone_context = f"{pinecone_context}\n{prefetched_context}"

    ai_response_text = await run_in_threadpool(
        ai_get_response,
        prompt=request.message,
        history=recent_history,
        state=current_state,
        pinecone_context=pinecone_context,
        neo4j_facts=context.get("neo4j_facts"),
        profile=context.get("profile"),
        user_facts_semantic=context.get("user_facts_semantic"),
    )

    # Intent detection and background tasks (same as non-streaming)
    detected = _detect_intent_and_entities(request.message)
    next_state = current_state
    if detected["intent"] == "PLAN_TRIP":
        next_state = "planning_trip"
        destination = detected["entities"].get("destination")
        if destination:
            celery_app.send_task(
                "prefetch_destination_info",
                args=[destination, session_id],
            )
    # (state persisted via post_message_update below)

    # Unified reminder creation/scheduling in streaming path with inline confirmation
    try:
        _res = await _maybe_create_and_schedule_reminder(request.message, current_user, tasks, context)
        if _res and _res.get("created"):
            eta = _res.get("eta_utc")
            profile = context.get("profile") or {}
            user_tz = (profile.get("timezone") if isinstance(profile, dict) else None) or "UTC"
            time_pref = (profile.get("time_format") if isinstance(profile, dict) else None)
            when = _format_eta_for_user(eta, user_tz, time_pref) if isinstance(eta, datetime) else str(eta)
            title = _res.get("title") or "Reminder"
            ai_response_text = (ai_response_text or "").strip()
            suffix = f"\n\nI've set a reminder: {title} at {when}. You'll get an email then."
            ai_response_text = (ai_response_text + suffix) if ai_response_text else suffix.strip()
    except Exception:
        pass

    # Save new messages
    user_message = {"sender": "user", "text": request.message}
    ai_message = {"sender": "assistant", "text": ai_response_text}
    await run_in_threadpool(
        sessions.update_one,
        {"_id": session_obj_id},
        {
            "$push": {"messages": {"$each": [user_message, ai_message]}},
            "$set": {"lastUpdatedAt": datetime.utcnow()},
        },
    )

    # Unified post message update
    post_message_update(
        user_id=user_id,
        user_key=current_user.get("email", user_id),
        session_id=session_id,
        user_message=request.message,
        ai_message=ai_response_text,
        state=next_state,
    )

    async def token_stream():
        words = ai_response_text.split()
        chunk = []
        for i, w in enumerate(words, 1):
            chunk.append(w)
            if i % 10 == 0:
                yield " ".join(chunk) + " "
                chunk = []
                await asyncio.sleep(0)
        if chunk:
            yield " ".join(chunk)

    return StreamingResponse(token_stream(), media_type="text/plain")

# =====================================================
# 🔹 Task Management Endpoints
# =====================================================
@router.post("/tasks")
async def create_task(
    task: TaskCreate,
    current_user: dict = Depends(get_current_active_user),
    tasks: Collection = Depends(get_tasks_collection),
):
    """Create a new pending task for the user."""
    new_task = {
        "email": current_user["email"],
        "content": task.content,
        "due_date_str": task.due_date,
        "status": "pending",
        "created_at": datetime.utcnow(),
    }
    result = tasks.insert_one(new_task)
    return {"status": "success", "task_id": str(result.inserted_id)}

@router.put("/tasks/{task_id}")
async def update_task(
    task_id: str,
    task: TaskUpdate,
    current_user: dict = Depends(get_current_active_user),
    tasks: Collection = Depends(get_tasks_collection),
):
    """Update task content or due date."""
    update_data = task.model_dump(exclude_unset=True)
    if "due_date" in update_data:
        update_data["due_date_str"] = update_data.pop("due_date")
    if not update_data:
        raise HTTPException(status_code=400, detail="No data provided to update.")
    try:
        result = tasks.update_one(
            {"_id": ObjectId(task_id), "email": current_user["email"]},
            {"$set": update_data},
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Task not found.")
        return {"status": "success"}
    except errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid task ID.")

@router.put("/tasks/{task_id}/done")
async def mark_task_done(
    task_id: str,
    current_user: dict = Depends(get_current_active_user),
    tasks: Collection = Depends(get_tasks_collection),
):
    """Mark a pending task as completed."""
    try:
        result = tasks.update_one(
            {"_id": ObjectId(task_id), "email": current_user["email"]},
            {"$set": {"status": "done"}},
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Task not found.")
        return {"status": "success"}
    except errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid task ID.")

@router.get("/tasks")
async def get_tasks(
    current_user: dict = Depends(get_current_active_user),
    tasks: Collection = Depends(get_tasks_collection),
):
    """Fetch all pending tasks for the user."""
    cursor = tasks.find({"email": current_user["email"], "status": "pending"}).sort(
        "created_at", -1
    )
    return [
        {"id": str(t["_id"]), "content": t["content"], "due_date": t.get("due_date_str")}
        for t in cursor
    ]

@router.get("/tasks/history")
async def get_task_history(
    current_user: dict = Depends(get_current_active_user),
    tasks: Collection = Depends(get_tasks_collection),
):
    """Fetch recently completed tasks (limit 10)."""
    cursor = tasks.find({"email": current_user["email"], "status": "done"}).sort(
        "created_at", -1
    ).limit(10)
    return [
        {"id": str(t["_id"]), "content": t["content"], "due_date": t.get("due_date_str")}
        for t in cursor
    ]

# =====================================================
# 🔹 Legacy Chat History Endpoints
# =====================================================
@router.get("/history")
async def get_chat_history(
    current_user: dict = Depends(get_current_active_user),
    chat_logs: Collection = Depends(get_chat_log_collection),
    limit: int = 50,
):
    """Retrieve the last `limit` chat messages from the user's legacy chat log."""
    cursor = chat_logs.find({"email": current_user["email"]}).sort("timestamp", -1).limit(limit)
    return [{"sender": m["sender"], "text": m["text"]} for m in cursor]

@router.delete("/history/clear")
async def clear_chat_history(
    current_user: dict = Depends(get_current_active_user),
    chat_logs: Collection = Depends(get_chat_log_collection),
):
    """Clear all legacy chat history for the user and delete Redis session state."""
    result = chat_logs.delete_many({"email": current_user["email"]})
    try:
        if hasattr(redis_service, 'redis_client') and redis_service.redis_client:
            await redis_service.redis_client.delete(current_user["email"])
    except Exception:
        pass
    return {
        "status": "success",
        "message": f"Deleted {result.deleted_count} messages from legacy chat log.",
    }

# =====================================================
# 🔹 Blueprint additions: per-session history & feedback
# =====================================================
@router.get("/{session_id}/history")
async def get_session_history(
    session_id: str,
    current_user: dict = Depends(get_current_active_user),
    sessions: Collection = Depends(get_sessions_collection),
    limit: int = 50,
):
    try:
        session_obj_id = ObjectId(session_id)
    except errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid session ID format.")
    doc = await run_in_threadpool(
        sessions.find_one,
        {"_id": session_obj_id, "userId": current_user["_id"]},
        {"messages": {"$slice": -limit}},
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Session not found.")
    return doc.get("messages", [])


@router.post("/{session_id}/feedback")
async def submit_session_feedback(
    session_id: str,
    body: FeedbackBody,
    current_user: dict = Depends(get_current_active_user),
):
    try:
        _ = ObjectId(session_id)
    except errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid session ID.")

    celery_app.send_task(
        "process_feedback_task",
        kwargs={"fact_id": body.fact_id, "correction": body.correction, "user_id": str(current_user["_id"])},
    )
    return {"status": "queued"}
