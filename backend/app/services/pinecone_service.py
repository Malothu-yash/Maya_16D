# backend/app/services/pinecone_service.py

import logging
from typing import Optional, Any, List, Dict, Tuple
from app.config import settings
from app.services.embedding_service import create_embedding

logger = logging.getLogger(__name__)

# =====================================================
# 🔹 Global Pinecone state
# =====================================================
pc: Optional[Any] = None
index = None
PINECONE_INDEX_NAME = settings.PINECONE_INDEX or "maya2-session-memory"
# This dimension MUST match your embedding model. Updated to 1536 for the new Pinecone index.
REQUIRED_DIMENSION = 1536

# =====================================================
# 🔹 Initialize Pinecone
# =====================================================
def initialize_pinecone():
    """
    Initializes the Pinecone client and index. Called once on app startup.
    It will automatically delete and recreate the index if the dimension is wrong.
    """
    global pc, index
    if not settings.PINECONE_API_KEY:
        logger.warning("⚠️ Pinecone API key not found. Pinecone service will be disabled.")
        return

    try:
        # Attempt modern v3 import first; if fails, inspect module for legacy shape
        try:
            from pinecone import Pinecone, ServerlessSpec  # type: ignore
            sdk_version = "v3"
            pc_obj = Pinecone(api_key=settings.PINECONE_API_KEY)
        except Exception as v3_err:  # noqa: BLE001
            import importlib
            try:
                pinecone_mod = importlib.import_module("pinecone")  # type: ignore
            except Exception as import_err:  # noqa: BLE001
                raise RuntimeError(
                    "Pinecone module import failed entirely. Install with 'pip install pinecone-client>=3'"
                    f" (v3 error: {v3_err}; import error: {import_err})"
                )

            # Identify legacy vs unexpected shapes
            has_init = hasattr(pinecone_mod, "init")
            has_index_attr = hasattr(pinecone_mod, "Index")
            if has_init and has_index_attr:
                try:
                    pinecone_mod.init(
                        api_key=settings.PINECONE_API_KEY,
                        environment=(settings.PINECONE_ENVIRONMENT or "us-east-1"),
                    )
                    sdk_version = "v2"
                    pc_obj = pinecone_mod
                except Exception as v2_err:  # noqa: BLE001
                    raise RuntimeError(
                        "Legacy Pinecone module present but initialization failed. Upgrade with 'pip install --upgrade pinecone-client'"
                        f" (v3 err: {v3_err}; v2 err: {v2_err})"
                    )
            else:
                exported = dir(pinecone_mod)
                raise RuntimeError(
                    "Unrecognized pinecone module shape. Expected v3 (Pinecone class) or v2 (init function). "
                    f"Found attributes: {exported[:25]}...  Install/upgrade with: pip install --upgrade 'pinecone-client>=3,<4'"
                )

        logger.info(f"Initializing Pinecone using {sdk_version} client path")
        create_new_index = False

        if sdk_version == "v3":
            existing = pc_obj.list_indexes()
            existing_names = existing.names() if hasattr(existing, "names") else [getattr(i, "name", None) for i in existing or []]
            if PINECONE_INDEX_NAME in existing_names:
                index_description = pc_obj.describe_index(PINECONE_INDEX_NAME)
                dim = getattr(index_description, "dimension", None) or (index_description.get("dimension") if isinstance(index_description, dict) else None)
                if dim != REQUIRED_DIMENSION:
                    logger.warning(f"Index '{PINECONE_INDEX_NAME}' wrong dimension {dim}; recreating")
                    pc_obj.delete_index(PINECONE_INDEX_NAME)
                    create_new_index = True
            else:
                create_new_index = True
            if create_new_index:
                pc_obj.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=REQUIRED_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_REGION),
                )
            bound_index = pc_obj.Index(PINECONE_INDEX_NAME)
        else:  # v2
            # v2 uses list_indexes() -> list, describe_index(index_name) returns dict with 'dimension'
            existing = pc_obj.list_indexes() or []
            if PINECONE_INDEX_NAME in existing:
                desc = pc_obj.describe_index(PINECONE_INDEX_NAME) or {}
                dim = desc.get("dimension")
                if dim != REQUIRED_DIMENSION:
                    logger.warning(f"Index '{PINECONE_INDEX_NAME}' wrong dimension {dim}; recreating")
                    pc_obj.delete_index(PINECONE_INDEX_NAME)
                    create_new_index = True
            else:
                create_new_index = True
            if create_new_index:
                pc_obj.create_index(PINECONE_INDEX_NAME, dimension=REQUIRED_DIMENSION, metric="cosine")
            bound_index = pc_obj.Index(PINECONE_INDEX_NAME)

        # Assign globals
        pc = pc_obj
        index = bound_index
        logger.info(f"✅ Pinecone index ready: '{PINECONE_INDEX_NAME}' (sdk {sdk_version})")
    except Exception as e:  # noqa: BLE001
        logger.error(f"❌ Error initializing Pinecone: {e}")
        pc = None
        index = None

# =====================================================
# 🔹 Internal Helper: Ensure Index Ready
# =====================================================
def _ensure_index_ready():
    """Internal helper to re-attempt initialization if the index is not ready."""
    if index is None:
        logger.warning("⚠️ Pinecone index was not initialized on startup, attempting again...")
        initialize_pinecone()
    return index is not None

# =====================================================
# 🔹 Upsert Session Summary
# =====================================================
def upsert_session_summary(session_id: str, summary: str):
    """Upserts a session summary embedding into Pinecone."""
    if not _ensure_index_ready():
        logger.error("❌ Pinecone index unavailable. Skipping upsert.")
        return
    try:
        embedding = create_embedding(summary)
        if embedding:
            index.upsert(vectors=[(session_id, embedding, {"summary": summary})])
            logger.info(f"✅ Upserted summary for session {session_id}.")
    except Exception as e:
        logger.error(f"❌ Failed to upsert summary: {e}")

# =====================================================
# 🔹 Query Relevant Summary
# =====================================================
def query_relevant_summary(text: str, top_k: int = 1) -> str | None:
    """Finds the most relevant summary for a given text."""
    if not _ensure_index_ready():
        logger.error("❌ Pinecone index unavailable. Cannot query.")
        return None
    try:
        embedding = create_embedding(text)
        if not embedding:
            logger.warning("⚠️ Failed to create embedding for query.")
            return None
        
        results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
        matches = (
            getattr(results, "matches", None)
            if not isinstance(results, dict)
            else results.get("matches", [])
        )
        if matches is None:
            matches = []
        if matches:
            best = matches[0]
            score = getattr(best, "score", None)
            md = getattr(best, "metadata", None)
            if score is None and isinstance(best, dict):
                score = best.get("score", 0)
                md = best.get("metadata")
            if (score or 0) > 0.75 and md:
                if isinstance(md, dict):
                    return md.get("summary")
                # Some SDKs may return metadata-like objects; fallback to str
                return str(md)
        return None
    except Exception as e:
        logger.error(f"❌ Query to Pinecone failed: {e}")
        return None

# =====================================================
# 🔹 Singleton-like Export for Compatibility
# =====================================================
class PineconeService:
    initialize_pinecone = staticmethod(initialize_pinecone)
    upsert_session_summary = staticmethod(upsert_session_summary)
    query_relevant_summary = staticmethod(query_relevant_summary)
    @staticmethod
    def is_ready() -> bool:
        return index is not None
    @staticmethod
    def get_index():
        return index

pinecone_service = PineconeService()

# =====================================================
# 🔹 New: Message-level Embeddings API
# =====================================================
def upsert_message_embedding(user_id: str, session_id: str, text: str, role: str, timestamp: str):
    """
    Upsert a single message embedding for later per-user semantic recall.
    - id format: f"{user_id}:{session_id}:{timestamp}:{role}"
    - metadata: includes user_id, session_id, role, timestamp, and text (for recall)
    """
    if not _ensure_index_ready():
        logger.debug("Pinecone index not ready; skipping message upsert.")
        return
    if not text:
        return

    try:
        emb = create_embedding(text)
        if not emb:
            return
        vid = f"{user_id}:{session_id}:{timestamp}:{role}"
        meta = {
            "user_id": user_id,
            "session_id": session_id,
            "role": role,
            "timestamp": timestamp,
            "text": text,
            "kind": "message",
        }
        # Use per-user namespace to keep embeddings isolated
        ns = f"user:{user_id}"
        try:
            index.upsert(vectors=[(vid, emb, meta)], namespace=ns)
        except TypeError:
            # Older SDKs may not accept namespace named arg; fall back to default
            index.upsert(vectors=[(vid, emb, meta)])
    except Exception as e:
        logger.debug(f"Pinecone message upsert failed: {e}")


def upsert_user_fact_embedding(user_id: str, fact_text: str, timestamp: str, category: str = "generic"):
    """Upsert a semantic embedding representing a stable user fact/preference.

    ID format: user:{user_id}:fact:{hash_prefix}  (hash = deterministic over text)
    Metadata includes scope/kind to allow filtered queries distinct from messages.
    """
    if not _ensure_index_ready() or not fact_text:
        return
    try:
        import hashlib
        emb = create_embedding(fact_text)
        if not emb:
            return
        h = hashlib.sha1(fact_text.encode("utf-8")).hexdigest()[:12]
        vid = f"user:{user_id}:fact:{h}"
        meta = {
            "user_id": user_id,
            "session_id": "",  # not session-bound
            "role": "fact",
            "timestamp": timestamp,
            "text": fact_text,
            "kind": "user_fact",
            "category": category,
        }
        ns = f"user:{user_id}"
        try:
            index.upsert(vectors=[(vid, emb, meta)], namespace=ns)
        except TypeError:
            index.upsert(vectors=[(vid, emb, meta)])
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Pinecone user_fact upsert failed: {e}")


def bulk_upsert(payloads: List[Dict[str, Any]]):
    """Bulk upsert heterogeneous payloads from the embedding queue.

    Supports kinds: message, user_fact
    Each payload must contain: user_id, text, timestamp, kind.
    """
    if not _ensure_index_ready():
        return
    vectors: List[Tuple[str, List[float], Dict[str, Any]]] = []
    try:
        import hashlib
        for item in payloads:
            text = item.get("text")
            if not text:
                continue
            emb = create_embedding(text)
            if not emb:
                continue
            kind = item.get("kind") or item.get("role") or "message"
            user_id = item.get("user_id", "")
            ts = item.get("timestamp", "")
            if kind == "user_fact":
                h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
                vid = f"user:{user_id}:fact:{h}"
                meta = {
                    "user_id": user_id,
                    "session_id": "",
                    "role": "fact",
                    "timestamp": ts,
                    "text": text,
                    "kind": "user_fact",
                    "category": item.get("category", "generic"),
                }
            else:
                session_id = item.get("session_id", "")
                role = item.get("role", "user")
                vid = f"{user_id}:{session_id}:{ts}:{role}"
                meta = {
                    "user_id": user_id,
                    "session_id": session_id,
                    "role": role,
                    "timestamp": ts,
                    "text": text,
                    "kind": "message",
                }
            vectors.append((vid, emb, meta))
        if vectors:
            # Assume batch belongs to same user; compute namespace from first item, else default
            first_user = next((it.get("user_id") for it in payloads if it.get("user_id")), "")
            ns = f"user:{first_user}" if first_user else None
            try:
                if ns:
                    index.upsert(vectors=vectors, namespace=ns)
                else:
                    index.upsert(vectors=vectors)
            except TypeError:
                index.upsert(vectors=vectors)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"bulk_upsert failed: {e}")

__all__ = [
    "pinecone_service",
    "upsert_message_embedding",
    "upsert_user_fact_embedding",
    "upsert_memory_embedding",
    "query_user_memories",
    "query_similar_texts",
    "query_user_facts",
    "bulk_upsert",
]


def query_similar_texts(user_id: str, text: str, top_k: int = 3) -> Optional[str]:
    """
    Query Pinecone for the most similar prior messages for this user and return
    a compact concatenated context string (limited to a few items).
    """
    if not _ensure_index_ready():
        return None
    if not text:
        return None
    try:
        emb = create_embedding(text)
        if not emb:
            return None
        # Restrict to this user's namespace and message-kind vectors
        ns = f"user:{user_id}"
        kwargs = {
            "vector": emb,
            "top_k": top_k,
            "include_metadata": True,
            "filter": {"user_id": {"$eq": user_id}, "kind": {"$eq": "message"}},
        }
        try:
            res = index.query(namespace=ns, **kwargs)
        except TypeError:
            res = index.query(**kwargs)
        matches = (
            getattr(res, "matches", None)
            if not isinstance(res, dict)
            else res.get("matches", [])
        ) or []
        snippets: list[str] = []
        for m in matches:
            md = getattr(m, "metadata", None)
            if md is None and isinstance(m, dict):
                md = m.get("metadata")
            md = md or {}
            snippet = md.get("text") if isinstance(md, dict) else None
            if snippet:
                snippets.append(snippet)
        return "\n---\n".join(snippets) if snippets else None
    except Exception as e:
        logger.debug(f"query_similar_texts failed: {e}")
        return None


def query_user_facts(user_id: str, hint_text: str, top_k: int = 5) -> List[str]:
    """Return top semantic user_fact snippets (kind=user_fact) for a user.

    hint_text guides the embedding query (can be last user message). We filter by kind=user_fact
    in metadata using supported Pinecone filter syntax.
    """
    if not _ensure_index_ready():
        return []
    try:
        emb = create_embedding(hint_text or user_id)
        if not emb:
            return []
        ns = f"user:{user_id}"
        kwargs = {
            "vector": emb,
            "top_k": top_k,
            "include_metadata": True,
            "filter": {"user_id": {"$eq": user_id}, "kind": {"$eq": "user_fact"}},
        }
        try:
            res = index.query(namespace=ns, **kwargs)
        except TypeError:
            res = index.query(**kwargs)
        matches = (
            getattr(res, "matches", None)
            if not isinstance(res, dict)
            else res.get("matches", [])
        ) or []
        out: List[str] = []
        for m in matches:
            md = getattr(m, "metadata", None)
            if md is None and isinstance(m, dict):
                md = m.get("metadata")
            if isinstance(md, dict):
                txt = md.get("text")
                if txt and txt not in out:
                    out.append(txt)
        return out
    except Exception as e:  # noqa: BLE001
        logger.debug(f"query_user_facts failed: {e}")
        return []


def upsert_memory_embedding(memory_id: str, user_id: str, text: str, lifecycle_state: str):
    """Add/replace embedding for a structured memory item.

    Vector id format: memory:{memory_id}
    Metadata distinguishes from messages/facts.
    """
    if not _ensure_index_ready() or not text:
        return
    try:
        emb = create_embedding(text)
        if not emb:
            return
        vid = f"memory:{memory_id}"
        meta = {
            "user_id": user_id,
            "kind": "memory",
            "memory_id": memory_id,
            "lifecycle_state": lifecycle_state,
            "text": text,
        }
        ns = f"user:{user_id}"
        try:
            index.upsert(vectors=[(vid, emb, meta)], namespace=ns)
        except TypeError:
            index.upsert(vectors=[(vid, emb, meta)])
    except Exception as e:  # noqa: BLE001
        logger.debug(f"upsert_memory_embedding failed: {e}")


def query_user_memories(user_id: str, query_text: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """Return top memory vectors with similarity score and metadata.

    Only active/candidate/distilled lifecycle memories are considered (filter on metadata lifecycle_state if present).
    """
    if not _ensure_index_ready() or not query_text:
        return []
    try:
        emb = create_embedding(query_text)
        if not emb:
            return []
        ns = f"user:{user_id}"
        kwargs = {
            "vector": emb,
            "top_k": top_k,
            "include_metadata": True,
            "filter": {"user_id": {"$eq": user_id}, "kind": {"$eq": "memory"}},
        }
        try:
            res = index.query(namespace=ns, **kwargs)
        except TypeError:
            res = index.query(**kwargs)
        matches = (
            getattr(res, "matches", None)
            if not isinstance(res, dict)
            else res.get("matches", [])
        ) or []
        out: List[Dict[str, Any]] = []
        for m in matches:
            score = getattr(m, "score", None)
            md = getattr(m, "metadata", None)
            if score is None and isinstance(m, dict):
                score = m.get("score")
                md = m.get("metadata")
            if isinstance(md, dict):
                # Filter by lifecycle if needed
                lc = md.get("lifecycle_state")
                if lc and lc not in ("active", "candidate", "distilled"):
                    continue
                out.append({
                    "memory_id": md.get("memory_id"),
                    "similarity": score,
                    "text": md.get("text"),
                    "lifecycle_state": lc,
                })
        return out
    except Exception as e:  # noqa: BLE001
        logger.debug(f"query_user_memories failed: {e}")
        return []


# =====================================================
# 🔹 Deletion Helpers (CRUD: Delete)
# =====================================================
def delete_vectors(vector_ids: List[str], namespace: Optional[str] = None) -> None:
    """Delete vectors by IDs, optionally within a namespace."""
    if not _ensure_index_ready() or not vector_ids:
        return
    try:
        if namespace:
            index.delete(ids=vector_ids, namespace=namespace)
        else:
            index.delete(ids=vector_ids)
    except TypeError:
        # Older SDK signatures
        index.delete(ids=vector_ids)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"delete_vectors failed: {e}")


def delete_user_memory_vectors(user_id: str, memory_id: str) -> None:
    """Delete vectors associated with a structured memory item."""
    try:
        delete_vectors([f"memory:{memory_id}"])
    except Exception:  # noqa: BLE001
        pass


def delete_user_namespace(user_id: str) -> None:
    """Delete an entire user's namespace (use with caution)."""
    if not _ensure_index_ready():
        return
    ns = f"user:{user_id}"
    try:
        index.delete(delete_all=True, namespace=ns)
    except TypeError:
        # Some SDKs may use different signature; fallback to best-effort by listing not available => skip
        logger.debug("Namespace delete not supported by this SDK; skipping")
    except Exception as e:  # noqa: BLE001
        logger.debug(f"delete_user_namespace failed: {e}")