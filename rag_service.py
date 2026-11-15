# rag_service.py
# Same two-pass pipeline as the original (translate query -> retrieve -> pass 1 -> anchor chunks ->
# window expansion -> streamed pass 2). Additions: metadata filters, [S#] citation tags verified on the
# server, "not covered" handling, logging, and incremental index sync.
import asyncio
import logging
import re
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field

import config
from filters import build_qdrant_filter, describe_filters, matches_filters
from lang_utils import (has_cjk, is_arabic, is_refusal, no_filter_match_message, normalize_text,
                        not_found_message, strip_cjk, strip_tags)
from logging_config import request_id_var

try:                                    # tracing is optional
    import logfire

    def _span(name, **attrs):
        return logfire.span(name, **attrs)
except ImportError:                     # pragma: no cover
    def _span(name, **attrs):
        return nullcontext(type("S", (), {"set_attribute": lambda *a, **k: None})())

log = logging.getLogger(__name__)

# Vector store state (filled by init_rag(); the original did this at import time)
client = None
vectorstore = None
chunks_by_id: dict = {}
llm = None
index_lock = threading.RLock()          # searches and index changes never run at the same time

_TAG_GROUP = re.compile(r"\[([^\]]*)\]")
_TAG_ID = re.compile(r"\bS(\d+)\b")
_TAG_ONLY = re.compile(r"\s*S\d+(?:\s*[,;،]\s*S\d+)*\s*")


@dataclass
class RagResult:
    answer: str = ""
    refused: bool = False
    tokens: int = 0
    sources: list = field(default_factory=list)


# ------------------------------------------------------------------ index lifecycle
def init_rag(timeout: float | None = None):
    """Open the index (waiting for Ollama if it is still starting) and sync it with ./docs."""
    global client, vectorstore, chunks_by_id
    from vector_store import initialize_vector_store
    deadline = time.monotonic() + (config.OLLAMA_WAIT_SECONDS if timeout is None else timeout)
    while True:
        try:
            with index_lock:
                client, vectorstore, chunks_by_id, report = initialize_vector_store(force_recreate=False)
            log.info("Startup index sync: %s", report.summary())
            return report
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise
            log.info("Waiting for Ollama/Qdrant (%s: %s)", type(exc).__name__, exc)
            time.sleep(3)


def sync_vector_store_service():
    """Incremental: only new/changed/deleted files are (re)embedded."""
    global chunks_by_id
    from vector_store import sync_index
    with index_lock:
        chunks_by_id, report = sync_index(
            client, vectorstore, data_path=config.DATA_PATH, collection=config.COLLECTION_NAME,
            embed_model=config.EMBED_MODEL, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    return report


def rebuild_vector_store_service():
    """Full rebuild (as in the original): drop the collection and re-embed everything."""
    global client, vectorstore, chunks_by_id
    from vector_store import initialize_vector_store
    with index_lock:
        try:
            client.close()
        except Exception as e:
            log.warning("Error while closing old client: %s", e)
        client, vectorstore, chunks_by_id, report = initialize_vector_store(force_recreate=True)
    return report


def delete_document_service(source_id: str) -> None:
    from vector_store import delete_source
    with index_lock:
        delete_source(client, config.COLLECTION_NAME, source_id)
        for k in [k for k, c in chunks_by_id.items() if c.metadata.get("source_id") == source_id]:
            del chunks_by_id[k]


def list_documents_service() -> list[dict]:
    docs: dict = {}
    for c in list(chunks_by_id.values()):
        md = c.metadata
        d = docs.setdefault(md["source_id"], {"source_id": md["source_id"], "file_name": md.get("file_name"),
                                              "file_type": md.get("file_type"), "chunks": 0,
                                              "modified_at": md.get("modified_at")})
        d["chunks"] += 1
    return sorted(docs.values(), key=lambda d: d["source_id"])


def close_vector_client():
    if client is not None:
        client.close()


def get_llm():
    global llm
    if llm is None:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0,
                         num_ctx=config.LLM_NUM_CTX)
    return llm


# ------------------------------------------------------------------ prompt / messages (original behaviour)
def build_messages(context: str, question: str, answer_lang: str, history: list | None = None):
    """System prompt (with context) + past turns + the new question."""
    msgs = [("system", config.prompt.replace("{context}", context))]
    for q, a in (history or []):
        msgs.append(("human", q))
        msgs.append(("ai", strip_tags(a)[: config.HISTORY_ANSWER_CHARS]))
    msgs.append(("human", f"{question}\n\n(Answer only in {answer_lang}. Do not use Chinese.)"))
    return msgs


def build_search_query(query: str, history: list | None) -> str:
    """Short follow-ups ("and the second one?") are searched together with the previous question."""
    if history and len(query.split()) <= config.FOLLOWUP_MAX_WORDS:
        return f"{history[-1][0]} {query}"
    return query


# ------------------------------------------------------------------ citations
def citation_label(doc) -> str:
    md = doc.metadata
    name = md.get("file_name") or "Unknown"
    if md.get("page_number"):
        return f"{name} (page {md['page_number']})"
    if md.get("section"):
        return f"{name} (section: {md['section']})"
    return name


def tag_documents(docs) -> list[tuple[str, object]]:
    return [(f"S{i}", d) for i, d in enumerate(docs, 1)]


def format_docs(tagged) -> str:
    return "\n\n".join(f"[{tag}] SOURCE: {citation_label(d)}\n{d.page_content}" for tag, d in tagged)


def parse_citations(answer: str, valid_count: int) -> tuple[list[int], list[int]]:
    """(valid tag numbers in order of first use, invalid tag numbers)."""
    valid, invalid = [], []
    for m in _TAG_GROUP.finditer(answer or ""):
        if not _TAG_ONLY.fullmatch(m.group(1)):
            continue
        for n in map(int, _TAG_ID.findall(m.group(1))):
            bucket = valid if 1 <= n <= valid_count else invalid
            if n not in bucket:
                bucket.append(n)
    return valid, invalid


def format_sources(tagged, cited: list[int], invalid: list[int]) -> str:
    """Sources come from stored metadata, never from the model's own words."""
    by_tag = dict(tagged)
    if cited:
        header = "Sources cited:"
        lines = [f"- [S{n}] {citation_label(by_tag[f'S{n}'])}" for n in cited]
    else:
        header = "Sources consulted (the answer did not cite them individually):"
        pages_by_file: dict = {}
        for _, d in tagged:
            md = d.metadata
            pages_by_file.setdefault(md.get("file_name") or "Unknown", set()).add(
                md.get("page_number") or md.get("section") or "-")
        lines = [f"- {n} ({', '.join(str(p) for p in sorted(pg, key=str))})" for n, pg in pages_by_file.items()]
    text = f"\n\n{header}\n" + "\n".join(lines)
    if invalid:
        text += "\n(Some citation markers did not match any retrieved passage and were ignored.)"
    return text


# ------------------------------------------------------------------ helpers
def _usage(msg) -> tuple[int, int]:
    """(prompt_tokens, completion_tokens) from a message or streamed chunk, if present."""
    meta = getattr(msg, "response_metadata", None) or {}
    p, c = meta.get("prompt_eval_count"), meta.get("eval_count")
    if p is None and c is None:
        um = getattr(msg, "usage_metadata", None) or {}
        return um.get("input_tokens", 0) or 0, um.get("output_tokens", 0) or 0
    return p or 0, c or 0


async def _safe_send(websocket, text: str):
    try:
        await websocket.send_text(text)
    except Exception:
        pass


def friendly_error(exc: Exception) -> str:
    name = f"{type(exc).__module__}.{type(exc).__name__}".lower()
    if isinstance(exc, (ConnectionError, TimeoutError)) or any(s in name for s in ("connect", "timeout", "ollama")):
        return "The language model service is unavailable right now. Please try again in a moment."
    return "Something went wrong while answering your question."


async def answer_with_guard(context: str, question: str, answer_lang: str, history: list | None = None):
    """Call the LLM; if CJK characters leak in, retry once with a stricter nudge."""
    msgs = build_messages(context, question, answer_lang, history)
    resp = await get_llm().ainvoke(msgs)
    if has_cjk(resp.content):
        msgs.append(("ai", resp.content))
        msgs.append(("human", f"Your answer contained Chinese. Rewrite the same answer ONLY in {answer_lang}."))
        retry = await get_llm().ainvoke(msgs)
        # sum tokens so the usage counter stays honest
        for k in ("prompt_eval_count", "eval_count"):
            retry.response_metadata[k] = retry.response_metadata.get(k, 0) + resp.response_metadata.get(k, 0)
        resp = retry
    return resp


# ------------------------------------------------------------------ real streaming
async def stream_answer(messages, websocket, answer_lang: str):
    """
    Streams tokens to the client as the model generates them.
    If Chinese/Japanese/Korean characters leak in, the partial answer is discarded
    (client gets ANSWER_RESET) and generated again with a stricter reminder; on the
    second attempt any stray CJK characters are simply stripped.
    Returns (answer_text, prompt_tokens, completion_tokens).
    """
    prompt_tokens = completion_tokens = 0
    parts = []
    await websocket.send_text("ANSWER_START:")
    try:
        for attempt in (1, 2):
            parts = []
            leaked = False
            p = c = 0
            stream = get_llm().astream(messages)
            try:
                async for chunk in stream:
                    piece = chunk.content or ""
                    cp, cc = _usage(chunk)
                    if cp or cc:
                        p, c = cp, cc
                    if piece and has_cjk(piece):
                        if attempt == 1:
                            leaked = True
                            break
                        piece = strip_cjk(piece)
                    if piece:
                        parts.append(piece)
                        await websocket.send_text(piece)
            finally:
                aclose = getattr(stream, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        pass
            prompt_tokens += p
            completion_tokens += c
            if not leaked:
                break
            await websocket.send_text("ANSWER_RESET:")
            messages = list(messages) + [(
                "human",
                f"Reminder: your previous answer contained Chinese. Answer again ONLY in {answer_lang}.",
            )]
    finally:
        await _safe_send(websocket, "ANSWER_END:")
    return "".join(parts), prompt_tokens, completion_tokens


async def translate_query(query: str):
    target = "English" if is_arabic(query) else "Arabic"
    messages = [
        ("system", f"Translate the user's text into {target}. Output only the translation, nothing else."),
        ("human", query),
    ]
    try:
        resp = await get_llm().ainvoke(messages)
        meta = resp.response_metadata
        tokens = meta.get("prompt_eval_count", 0) + meta.get("eval_count", 0)
        text = resp.content.strip()
        return (text or None), tokens
    except Exception as e:
        log.warning("Translation failed: %s", e)
        return None, 0


# ------------------------------------------------------------------ retrieval
def _filter(filters):
    return build_qdrant_filter(filters) if filters else None


def retrieve_multilingual(query: str, alt_query: str | None, filters: dict | None = None):
    """Original behaviour (k chunks per query, merged, de-duplicated) + metadata filter and best score."""
    flt = _filter(filters)
    seen, merged, best = set(), [], 0.0
    with index_lock:
        for q in (query, alt_query):
            if not q:
                continue
            for d, score in vectorstore.similarity_search_with_score(q, k=config.RETRIEVER_K, filter=flt):
                best = max(best, score)
                key = d.metadata.get("chunk_id")
                if key not in seen:
                    seen.add(key)
                    merged.append(d)
    return merged, best


def retrieve_anchors(text: str, filters: dict | None = None):
    """Chunks that match the pass-1 answer (the original `retriever_final`)."""
    with index_lock:
        hits = vectorstore.similarity_search_with_score(
            text, k=config.RETRIEVER_K, filter=_filter(filters), score_threshold=config.SCORE_THRESHOLD)
    return [d for d, _ in hits]


def expand_window(anchor_docs, filters: dict | None = None):
    """Original window expansion: each anchor plus the chunk before and the two after (same file)."""
    wanted = set()
    for d in anchor_docs:
        sid, idx = d.metadata.get("source_id"), d.metadata.get("chunk_index")
        if sid is None or idx is None:
            continue
        for n in (idx - 1, idx, idx + 1, idx + 2):
            nb = chunks_by_id.get(f"{sid}::{n}")
            if nb is not None and matches_filters(nb.metadata, filters):
                wanted.add((sid, n))
    return [chunks_by_id[f"{s}::{n}"] for s, n in sorted(wanted)]


# ------------------------------------------------------------------ pipeline
async def _finish(websocket, result: RagResult) -> RagResult:
    await websocket.send_text(f'METADATA:{{"tokens": {result.tokens}}}')
    return result


async def _refuse(websocket, text: str, tokens: int) -> RagResult:
    await websocket.send_text("\nAnswer:\n")
    await websocket.send_text(text)
    return await _finish(websocket, RagResult(answer=text, refused=True, tokens=tokens))


async def run_two_pass_rag(user_message: str, websocket, history: list | None = None,
                           filters: dict | None = None) -> RagResult:
    history, filters = history or [], filters or {}
    request_id_var.set(uuid.uuid4().hex[:8])
    query = normalize_text(user_message).strip()
    answer_lang = "Arabic" if is_arabic(query) else "English"
    log.info("Question received (lang=%s, filters=%s, chars=%d)", answer_lang, describe_filters(filters), len(query))
    await websocket.send_text(f"\n--- Running Two-Pass RAG Query: '{query}' ---\n")
    if filters:
        await websocket.send_text(f"[Filters] {describe_filters(filters)}\n")

    with _span("rag_agent", openinference_span_kind="AGENT") as span:
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Search text: the question, plus the previous question when this looks like a short follow-up
        search_query = build_search_query(query, history)
        if search_query != query:
            await websocket.send_text(f"\n[Search query with context] {search_query}\n")

        alt_query, translate_tokens = await translate_query(search_query)
        total_prompt_tokens += translate_tokens
        if alt_query:
            await websocket.send_text(f"\n[Translated query] {alt_query}\n")

        # PASS 1
        candidates, best = await asyncio.to_thread(retrieve_multilingual, search_query, alt_query, filters)
        log.info("Best retrieval score %.3f (%d candidates)", best, len(candidates))
        if not candidates:
            return await _refuse(websocket, no_filter_match_message(answer_lang) if filters
                                 else not_found_message(answer_lang), total_prompt_tokens)
        if best < config.RELEVANCE_MIN_SCORE:
            log.info("Not covered: best score %.3f < %.3f", best, config.RELEVANCE_MIN_SCORE)
            return await _refuse(websocket, not_found_message(answer_lang), total_prompt_tokens)

        pass1_response = await answer_with_guard(format_docs(tag_documents(candidates)), query, answer_lang, history)
        p1_meta = pass1_response.response_metadata
        p1_prompt = p1_meta.get("prompt_eval_count", 0)
        p1_completion = p1_meta.get("eval_count", 0)
        total_prompt_tokens += p1_prompt
        total_completion_tokens += p1_completion
        await websocket.send_text(f"\n[Pass 1 Token Counter] Prompt: {p1_prompt} | Completion: {p1_completion} | Subtotal: {p1_prompt + p1_completion}\n")

        if is_refusal(pass1_response.content):
            log.info("Pass 1 declined to answer: not covered")
            return await _refuse(websocket, not_found_message(answer_lang),
                                 total_prompt_tokens + total_completion_tokens)

        # ANCHOR SELECTION & WINDOW EXPANSION
        anchors = await asyncio.to_thread(retrieve_anchors, strip_tags(pass1_response.content), filters)
        if not anchors and candidates:
            anchors = [candidates[0]]
        expanded_chunks = expand_window(anchors, filters) or list(anchors)
        tagged = tag_documents(expanded_chunks)

        # PASS 2 (streamed to the client token by token)
        messages = build_messages(format_docs(tagged), query, answer_lang, history)
        await websocket.send_text("\nAnswer:\n")
        final_answer, p2_prompt, p2_completion = await stream_answer(messages, websocket, answer_lang)
        total_prompt_tokens += p2_prompt
        total_completion_tokens += p2_completion

        await websocket.send_text(f"\n[Pass 2 Token Counter] Prompt: {p2_prompt} | Completion: {p2_completion} | Subtotal: {p2_prompt + p2_completion}\n")
        await websocket.send_text(f"\n[Cumulative Total Used] Prompt: {total_prompt_tokens} | Completion: {total_completion_tokens} | Combined Total: {total_prompt_tokens + total_completion_tokens}\n")

        query_total_tokens = total_prompt_tokens + total_completion_tokens
        span.set_attribute("llm.token_count.total", query_total_tokens)
        span.set_attribute("output.value", final_answer)

        if is_refusal(final_answer):                      # "I dont know": no sources to show
            return await _finish(websocket, RagResult(final_answer, True, query_total_tokens))

        cited, invalid = parse_citations(final_answer, len(tagged))
        if invalid:
            log.warning("Answer cited unknown tags %s (valid: 1..%d)", invalid, len(tagged))
        await websocket.send_text(format_sources(tagged, cited, invalid) + "\n")
        by_tag = dict(tagged)
        return await _finish(websocket, RagResult(
            final_answer, False, query_total_tokens, [citation_label(by_tag[f"S{n}"]) for n in cited]))
