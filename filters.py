"""
Metadata filtering. One validated, canonical filter dict is used everywhere
(WebSocket JSON, the /filter chat command, retrieval, neighbour expansion).

Canonical keys:
  file_name: [str]     file_type: [pdf|docx]     language: [ar|en]
  page_from / page_to: int (PDF pages; DOCX chunks have no page number so a page filter excludes them)
  modified_after: epoch seconds (inclusive)      modified_before: epoch seconds (exclusive)
"""
import shlex
from datetime import datetime, timedelta, timezone

SUPPORTED_TYPES = {"pdf", "docx"}
_LANGS = {"ar": "ar", "arabic": "ar", "en": "en", "english": "en"}
_ALIASES = {
    "file": "file_name", "file_name": "file_name", "filename": "file_name",
    "type": "file_type", "file_type": "file_type",
    "lang": "language", "language": "language",
    "page": "page", "page_from": "page_from", "page_to": "page_to",
    "after": "modified_after", "modified_after": "modified_after",
    "before": "modified_before", "modified_before": "modified_before",
}
MAX_LIST = 20


class FilterError(ValueError):
    """The user supplied an invalid filter (safe to show to the user)."""


def _blank(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, (list, tuple)) and not v)


def _as_list(v) -> list[str]:
    items = v if isinstance(v, (list, tuple)) else str(v).split(",")
    out = []
    for i in items:
        s = str(i).strip()
        if s and s not in out:
            out.append(s)
    if len(out) > MAX_LIST:
        raise FilterError(f"At most {MAX_LIST} values per filter.")
    return out


def _page(name: str, v) -> int:
    try:
        n = int(str(v).strip())
    except ValueError:
        raise FilterError(f"'{name}' must be a whole number, got {v!r}.") from None
    if n < 1:
        raise FilterError(f"'{name}' must be 1 or greater.")
    return n


def _day(name: str, v) -> datetime:
    try:
        return datetime.strptime(str(v).strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise FilterError(f"'{name}' must be a date like 2025-01-31, got {v!r}.") from None


def normalize_filters(raw) -> dict:
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise FilterError("Filters must be an object.")
    out: dict = {}
    for key, value in raw.items():
        canon = _ALIASES.get(str(key).strip().lower())
        if canon is None:
            raise FilterError(f"Unknown filter '{key}'. Allowed: file, type, lang, page, after, before.")
        if _blank(value):
            continue
        if canon == "file_name":
            out["file_name"] = _as_list(value)
        elif canon == "file_type":
            types = [t.lower().lstrip(".") for t in _as_list(value)]
            bad = [t for t in types if t not in SUPPORTED_TYPES]
            if bad:
                raise FilterError(f"Unsupported file type {bad[0]!r}. Use: {', '.join(sorted(SUPPORTED_TYPES))}.")
            out["file_type"] = types
        elif canon == "language":
            langs = []
            for t in _as_list(value):
                if t.lower() not in _LANGS:
                    raise FilterError(f"Unsupported language {t!r}. Use: ar, en.")
                langs.append(_LANGS[t.lower()])
            out["language"] = list(dict.fromkeys(langs))
        elif canon == "page":
            lo, _, hi = str(value).partition("-")
            out["page_from"] = _page("page", lo)
            out["page_to"] = _page("page", hi) if hi.strip() else out["page_from"]
        elif canon in ("page_from", "page_to"):
            out[canon] = _page(canon, value)
        elif canon == "modified_after":
            out[canon] = int(_day(canon, value).timestamp())
        elif canon == "modified_before":   # inclusive of that whole day
            out[canon] = int((_day(canon, value) + timedelta(days=1)).timestamp())
    if "page_from" in out and "page_to" in out and out["page_from"] > out["page_to"]:
        raise FilterError("Page range is reversed (from is greater than to).")
    if "modified_after" in out and "modified_before" in out and out["modified_after"] >= out["modified_before"]:
        raise FilterError("'after' must be earlier than 'before'.")
    return out


def parse_filter_command(arg: str) -> dict:
    """'file=a.pdf,b.docx lang=ar page=2-5 after=2025-01-01' -> canonical filters."""
    try:
        tokens = shlex.split(arg)
    except ValueError:
        raise FilterError("Could not parse the filter (unbalanced quotes).") from None
    raw = {}
    for tok in tokens:
        key, sep, value = tok.partition("=")
        if not sep:
            raise FilterError(f"Expected key=value, got {tok!r}.")
        raw[key] = value
    return normalize_filters(raw)


def describe_filters(f: dict) -> str:
    if not f:
        return "none"
    parts = []
    if f.get("file_name"): parts.append("file=" + ",".join(f["file_name"]))
    if f.get("file_type"): parts.append("type=" + ",".join(f["file_type"]))
    if f.get("language"): parts.append("lang=" + ",".join(f["language"]))
    if "page_from" in f or "page_to" in f:
        parts.append(f"pages={f.get('page_from', '1')}-{f.get('page_to', 'end')}")
    day = lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    if "modified_after" in f: parts.append("after=" + day(f["modified_after"]))
    if "modified_before" in f: parts.append("before=" + day(f["modified_before"] - 86400))
    return ", ".join(parts)


def to_conditions(f: dict) -> list[tuple]:
    """Backend-neutral conditions: ('any', key, [values]) or ('range', key, {op: value})."""
    c: list[tuple] = []
    for key in ("file_name", "file_type", "language"):
        if f.get(key):
            c.append(("any", key, f[key]))
    page = {op: f[k] for op, k in (("gte", "page_from"), ("lte", "page_to")) if k in f}
    if page:
        c.append(("range", "page_number", page))
    mod = {op: f[k] for op, k in (("gte", "modified_after"), ("lt", "modified_before")) if k in f}
    if mod:
        c.append(("range", "modified_ts", mod))
    return c


def build_qdrant_filter(f: dict | None, extra_must: list | None = None):
    """Qdrant Filter for the chunk payload (langchain-style layout: metadata.<key>), or None."""
    from qdrant_client.http import models as qm
    must = list(extra_must or [])
    for kind, key, val in to_conditions(f or {}):
        if kind == "any":
            must.append(qm.FieldCondition(key=f"metadata.{key}", match=qm.MatchAny(any=val)))
        else:
            must.append(qm.FieldCondition(key=f"metadata.{key}", range=qm.Range(**val)))
    return qm.Filter(must=must) if must else None


def matches_filters(md: dict, f: dict | None) -> bool:
    """Python-side equivalent of the Qdrant filter; used for the neighbour chunks kept in memory."""
    if not f:
        return True
    for kind, key, val in to_conditions(f):
        v = md.get(key)
        if kind == "any":
            if v not in val:
                return False
        else:
            if v is None:
                return False
            if "gte" in val and v < val["gte"]: return False
            if "lte" in val and v > val["lte"]: return False
            if "lt" in val and v >= val["lt"]: return False
    return True
