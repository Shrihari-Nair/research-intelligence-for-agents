"""Text helpers shared by agents: keywords, whitespace-normalized verbatim quote search, passage chunking."""
from __future__ import annotations

import re

STOP = frozenset("""a an and are as at be but by can could do does for from give has have how i in into is it its me my of on or
please should show tell than that the their them these this to up was we what when where which who why will with would you your
find look about also any some more most very just like need want did were been being if""".split())
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\.\+#]*")
_WS = re.compile(r"\s+")


def keywords(text: str, n: int) -> list[str]:
    """First `n` unique non-stopword tokens (lowercased, order preserved)."""
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        tok = tok.strip(".-")
        if tok and tok not in STOP and tok not in out:
            out.append(tok)
        if len(out) == n:
            break
    return out


def normalize_ws(text: str) -> str:
    """Collapse all whitespace (incl. non-breaking spaces) to single spaces and strip."""
    return _WS.sub(" ", text.replace(" ", " ")).strip()


def find_quote(quote: str, content: str) -> tuple[int, int] | None:
    """Span in `content` of a verbatim quote, ignoring only whitespace differences (case and wording must match)."""
    q = normalize_ws(quote)
    if not q:
        return None
    # regex: quote tokens separated by any whitespace, so offsets map to the ORIGINAL content
    pattern = r"\s+".join(re.escape(tok) for tok in q.split(" "))
    m = re.search(pattern.replace(re.escape(" "), " "), content.replace(" ", " "))
    return (m.start(), m.end()) if m else None


def chunk_passages(text: str, target_chars: int) -> list[dict]:
    """Split text into sentence-aligned windows of about `target_chars`, keeping offsets into `text`."""
    if not text.strip():
        return []
    sentences = [(m.start(), m.end()) for m in re.finditer(r"[^.!?\n]+[.!?]*\s*", text) if m.group(0).strip()]
    chunks, cur_start, cur_end = [], None, None
    for s, e in sentences:
        if cur_start is None:
            cur_start, cur_end = s, e
        elif e - cur_start > target_chars:
            chunks.append((cur_start, cur_end))
            cur_start, cur_end = s, e
        else:
            cur_end = e
    if cur_start is not None:
        chunks.append((cur_start, cur_end))
    out = []
    for s, e in chunks:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        for a in range(s, e, target_chars + target_chars // 4 or 1):  # hard-split an oversized sentence
            b = min(e, a + target_chars + target_chars // 4)
            out.append({"start": a, "end": b, "text": text[a:b]})
    return out


class ShortIds:
    """Short sequential ids (c1, c2, ...) shown to an LLM in place of long hashes it could mis-copy; mapped back in code."""

    def __init__(self, ids, prefix: str):
        self._fwd: dict[str, str] = {}
        for i in ids:
            if i not in self._fwd:
                self._fwd[i] = f"{prefix}{len(self._fwd) + 1}"
        self._back = {v.lower(): k for k, v in self._fwd.items()}

    def short(self, real: str) -> str:
        """Short id for a real id (KeyError if the id was not registered)."""
        return self._fwd[real]

    def real(self, short: str) -> str | None:
        """Real id for a short id (case/whitespace tolerant); None if unknown."""
        return self._back.get(str(short).strip().lower())

    def __len__(self) -> int:
        return len(self._fwd)
