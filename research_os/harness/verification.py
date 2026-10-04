"""Deterministic verification layer: grounding checks, overreach checks, contradiction detection, gate logic."""
from __future__ import annotations

import re

from research_os.agents.text import find_quote, keywords
from research_os.contracts import Source
from research_os.router.entities import EntityExtractor

_COMPARATIVE = re.compile(r"\b(faster|slower|better|worse|best|worst|fastest|slowest|outperform\w*|superior|inferior|cheaper|cheapest|"
                          r"than|most|least)\b", re.I)
_UNIVERSAL = re.compile(r"\b(always|never|every|everyone|all|none)\b", re.I)
_CAUSAL = re.compile(r"\b(because|due to|causes?|caused|leads? to|results? in|therefore)\b", re.I)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_NEGATION = re.compile(r"\b(no|not|never|without|lacks?|cannot|can't|doesn't|does not|isn't|is not|has no|have no|n't)\b", re.I)
_GENERIC = frozenset("""support supports supported has have had is are was were be release released uses use used using provide provides
provided offer offers library framework through under via based built build builds allow allows enable enables include includes
contain contains run runs work works make makes can will""".split())
_ENTITIES = EntityExtractor.from_config()


def _stem(tok: str) -> str:
    return tok[:6]


def deterministic_checks(claim: dict, sources: dict[str, Source], wf: dict) -> list[str]:
    """Reasons a claim is rejected before any LLM judging (empty list = passes)."""
    reasons: list[str] = []
    src = sources.get(claim.get("source_id", ""))
    if not claim.get("source_id") or src is None:
        return ["claim has no source in the run's source store"]
    quote, text = claim.get("evidence_quote", ""), claim.get("claim", "")
    if not quote or find_quote(quote, src.content) is None:
        return ["evidence quote is not a verbatim substring of the stored source"]
    allowed = wf.get("allowed_source_types") or []
    if allowed and src.source_type not in allowed:
        reasons.append(f"source type {src.source_type!r} is not allowed for this workflow")
    if wf.get("require_dates") and src.published_at is None:
        reasons.append("source has no publication date but the workflow requires dated sources")
    if _COMPARATIVE.search(text) and not (_COMPARATIVE.search(quote) or _NUMBER.search(quote)):
        reasons.append("comparative/superlative claim is not backed by a comparative or quantitative quote")
    qnums = {n.replace(",", "") for n in _NUMBER.findall(quote)}
    missing = [n for n in (x.replace(",", "") for x in _NUMBER.findall(text)) if n not in qnums]
    if missing and not qnums:  # if the quote has other numbers, the judge decides (it may be CONTRADICTED)
        reasons.append(f"number(s) {missing} in the claim do not appear in the quote")
    if _UNIVERSAL.search(text) and not _UNIVERSAL.search(quote):
        reasons.append("universal statement (always/never/all/every) is not present in the quote")
    if _CAUSAL.search(text) and not _CAUSAL.search(quote):
        reasons.append("causal language in the claim is not present in the quote")
    return reasons


def ungrounded_numbers(text: str, evidence_texts: list[str]) -> list[str]:
    """Numbers in `text` that appear in none of `evidence_texts` (thousands separators ignored)."""
    have = {n.replace(",", "") for t in evidence_texts for n in _NUMBER.findall(t)}
    return [n for n in dict.fromkeys(x.replace(",", "") for x in _NUMBER.findall(text)) if n not in have]


def _entities_of(text: str) -> list[str]:
    e = _ENTITIES.extract(text)
    return [*e["technologies"], *e["companies"]]


def detect_conflicts(claims: list[dict], sources: dict[str, Source]) -> list[dict]:
    """Groups of claims about the same entity and attribute whose polarity (asserted vs negated) differs."""
    info = []
    for c in claims:
        ents = _entities_of(c["claim"])
        if not ents:
            continue
        ent_words = {w for e in ents for w in e.lower().split()}
        attrs = {_stem(w) for w in keywords(c["claim"], 14) if w not in ent_words and w not in _GENERIC and not _NEGATION.fullmatch(w)
                 and len(w) > 3}
        info.append((c, set(ents), attrs, bool(_NEGATION.search(c["claim"]))))
    groups: dict[tuple[str, str], dict[bool, list[dict]]] = {}
    for i, (ci, ei, ai, ni) in enumerate(info):
        for cj, ej, aj, nj in info[i + 1:]:
            if ni == nj or ci["source_id"] == cj["source_id"]:      # a conflict is two sources disagreeing
                continue
            for ent in ei & ej:
                for attr in ai & aj:
                    g = groups.setdefault((ent, attr), {True: [], False: []})
                    for c, neg in ((ci, ni), (cj, nj)):
                        if c not in g[neg]:
                            g[neg].append(c)
    out = []
    for (ent, attr), sides in groups.items():
        def side(stance: str, cl: list[dict]) -> dict:
            return {"stance": stance, "claims": [{"claim_id": c["claim_id"], "claim": c["claim"], "source_id": c["source_id"],
                                                  "source_url": sources[c["source_id"]].url, "evidence_quote": c["evidence_quote"]}
                                                 for c in cl]}
        out.append({"subject": f"{ent} / {attr}", "sides": [side("asserts", sides[False]), side("negates", sides[True])]})
    return out


def gate(counts: dict[str, int], supported_source_types: list[str], wf: dict, n_conflicts: int = 0, n_conflict_claims: int | None = None) -> dict:
    """PASS/FAIL decision from verdict counts and workflow thresholds (FAIL hands off to recovery)."""
    supported, unsupported = counts.get("SUPPORTED", 0), counts.get("UNSUPPORTED", 0)
    total = sum(counts.values())
    rate = unsupported / total if total else 1.0
    reason, note = None, None
    if n_conflicts and counts.get("CONTRADICTED", 0):
        # a detected conflict IS evidence (the report surfaces both sides): count it toward the minimum and skip the required-type check
        note = f"{n_conflicts} conflict(s) between sources; surfaced in the report instead of failing the run"
        # only claims the conflict detector flagged count; claims the judge found contradicted by their own quote do not
        supported += counts["CONTRADICTED"] if n_conflict_claims is None else min(n_conflict_claims, counts["CONTRADICTED"])
    if total == 0:
        reason = "no claims to verify"
    elif supported < wf["min_supported_claims"]:
        reason = f"supported claims ({supported}) below min_supported_claims ({wf['min_supported_claims']})"
    elif rate > wf["max_unsupported_rate"]:
        reason = f"unsupported rate {rate:.2f} exceeds max_unsupported_rate {wf['max_unsupported_rate']}"
    elif wf.get("required_source_types") and not note and not (set(supported_source_types) & set(wf["required_source_types"])):
        reason = f"no supported claim from required source types {wf['required_source_types']}"
    return {"result": "FAIL" if reason else "PASS", "reason": reason, "note": note, "counts": counts, "unsupported_rate": rate,
            "failure_type": "NO_USEFUL_SOURCES" if reason else None}
