import re
import unicodedata

_ARABIC_CHARS = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670]")


def normalize_text(text: str) -> str:
    """
    Cleans text extracted from PDFs.
    - NFKC converts Arabic presentation forms (common in PDF extraction)
      back to normal Arabic letters.
    - Removes diacritics (tashkeel) and tatweel so words match consistently.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _TASHKEEL.sub("", text)
    return text.replace("\u0640", "")

def is_arabic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    arabic = sum(1 for c in letters if _ARABIC_CHARS.match(c))
    return arabic / len(letters) > 0.3

_CJK_CHARS = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u30FF\uAC00-\uD7AF]")


def has_cjk(text: str) -> bool:
    return bool(_CJK_CHARS.search(text))


def strip_cjk(text: str) -> str:
    return _CJK_CHARS.sub("", text)


# ----------------------------------------------------------------- additions
_TAG = re.compile(r"\[\s*S\d+(?:\s*[,;،]\s*S\d+)*\s*\]")
NOT_FOUND = {
    "English": "I dont know the answer to that question",
    "Arabic": "لا أعرف إجابة هذا السؤال بناءً على المستندات المتاحة.",
}
NO_FILTER_MATCH = {
    "English": "No indexed document matches the selected filters. Clear or change the filters and try again.",
    "Arabic": "لا يوجد أي مستند مفهرس يطابق عوامل التصفية المحددة. عدّل عوامل التصفية أو أزلها وحاول مرة أخرى.",
}
_REFUSAL_STARTS = ("i don't know", "i dont know", "i do not know", "لا أعرف", "لا اعرف")


def strip_tags(text: str) -> str:
    """Remove [S1]-style citation tags."""
    return _TAG.sub("", text or "")


def not_found_message(lang: str) -> str:
    return NOT_FOUND.get(lang, NOT_FOUND["English"])


def no_filter_match_message(lang: str) -> str:
    return NO_FILTER_MATCH.get(lang, NO_FILTER_MATCH["English"])


def is_refusal(text: str) -> bool:
    """True when the whole answer is a "don't know". A partial answer that merely contains an
    "I don't know" for one sub-question does not count: only the start of the answer is checked."""
    t = strip_tags(text).strip().lower().replace("\u2019", "'").lstrip("\"'“”.:- ")
    if not t:
        return True
    return len(t) <= 160 and t.startswith(_REFUSAL_STARTS)
