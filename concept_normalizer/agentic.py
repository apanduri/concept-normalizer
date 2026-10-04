"""Two-agent LLM-driven normalization — ported from CP, with the target abstracted.

The mechanism is Abdulnazar-shaped: Agent 1 turns messy free text into a handful
of candidate clinical phrases at different abstraction levels, each phrase is
embedded and retrieved against the SELECTED target vocabulary, then Agent 2
picks the best among the retrieved concepts (or returns -1 for "none fit").

What changed from CP's version:

  * `target: TargetProfile` replaces hardcoded SNOMED. Prompts are templated on
    the vocabulary's name and style — "SNOMED CT-style concept labels" becomes
    "<target>-style concept labels" — so LOINC, OMOP and BSO-AD inherit the
    behaviour without a separate orchestrator each.
  * `Retriever` (concept_normalizer.retriever) replaces `SnomedMapper`. The
    retrieval mechanism is identical (SapBERT + FAISS + Levenshtein); only the
    index being searched changes per target.
  * The output drops SNOMED-specific keys (`snomed_id`, hierarchy ancestors) in
    favour of the package's existing Normalization type.

Deliberately NOT changed:

  * Agent 2's abstraction-level criterion. "Prefer the most specific concept
    that still covers the full meaning" is vocabulary-independent and is the
    reason this pipeline doesn't resolve "Type 2 Diabetes" to "Clinical finding".
  * The Agent-1 → embedding → Agent-2 → fallback order. Each fallback has a
    reason (LLM unavailable / returned nonsense / no candidates matched) and the
    behaviour under each is what CP's existing SNOMED path produces, so a
    comparable test case produces a comparable answer.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from .retriever import RetrievedConcept, Retriever

log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class TargetProfile:
    """What the agent prompts need to know about one target vocabulary.

    Instances are small and declarative — no behaviour, just the strings and
    examples that calibrate the LLMs for a particular vocabulary. CP's hardcoded
    SNOMED prompts become one profile; adding a new target is adding another
    profile and an index.
    """

    name: str                   # "SNOMED CT", "LOINC", "BSO-AD", "OMOP standard concepts"
    style_hint: str             # e.g. "SNOMED CT-style concept labels"
    purpose: str                # what the mapping is used for downstream, in one line
    examples: list[tuple[str, list[str]]]  # (raw input, example candidates) pairs — few-shot
    index_dir: str              # on-disk path to this target's FAISS index + metadata


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Agent 1 — candidate generation.
#
# The task is unchanged from CP's prompt: produce 3-5 phrases at different
# abstraction levels. What's templated out is the TARGET-specific phrasing so a
# LOINC or BSO-AD target gets a prompt that calibrates for the right style.
_AGENT1_PROMPT = """\
You are a clinical terminology and ontology assistant.

PURPOSE:
Your output will be used to map text to {target_name} concepts for:
{purpose}

Each candidate must resemble a valid {style_hint} at the correct level of abstraction.

--------------------------------------------------
TASK:

Given a raw input phrase, generate 3 to 5 candidate concept phrases suitable for
{target_name} mapping.

Each candidate should represent a DIFFERENT interpretation or abstraction level
of the same input:
- At least one SPECIFIC concept-level term
- At least one BROADER parent-style term
- Optionally a related clinical finding or disorder term

--------------------------------------------------
RULES:

1. Remove non-clinical and contextual words such as:
   identification, management, classification, study, proposal, based on, in EHRs,
   organization names, methods, workflows, or explanatory text

2. Keep only the core clinical meaning.

3. If the input contains abbreviations, expand them in at least one candidate.

4. If the input contains multiple concepts joined by "and" or "or":
   - Include the shared parent concept as one candidate
   - Include the individual concepts as separate candidates

5. Each candidate must be short (typically 2-5 words).

6. Do NOT include explanations, numbering, or labels — just the terms.

--------------------------------------------------
EXAMPLES:
{examples}

--------------------------------------------------
OUTPUT FORMAT:

Return ONLY a JSON array of strings. No markdown, no explanation.

--------------------------------------------------
INPUT:
{raw_input}"""


# Agent 2 — best-match selection.
#
# The decision criteria are vocabulary-independent and come straight from CP's
# prompt. Semantic correctness over score; abstraction level over score; score
# only as a tiebreaker. Returning -1 for "none acceptable" is important — it is
# what lets the pipeline abstain rather than ship a wrong concept.
_AGENT2_PROMPT = """\
You are a clinical ontology expert evaluating {target_name} concept mappings.

TASK:
An input phrase was normalized into several candidate terms. Each candidate was
mapped to a {target_name} concept using SapBERT + FAISS. You must pick the BEST
match.

--------------------------------------------------
ORIGINAL INPUT:
{raw_input}

--------------------------------------------------
CANDIDATES AND THEIR {target_name_upper} MAPPINGS:

{candidates_table}

--------------------------------------------------
DECISION CRITERIA (in priority order):

1. SEMANTIC CORRECTNESS — Does the mapped concept actually represent what the
   input is about? Reject mappings where the concept is clinically unrelated to
   the input.

2. ABSTRACTION LEVEL — Prefer the most specific concept that still covers the
   full meaning. Too broad (e.g., "Clinical finding" for a diabetes input) is
   bad. Too narrow (e.g., "Type 2 diabetes with renal complication" for a
   general diabetes input) is also bad.

3. SCORE — Higher is better, but only as a tiebreaker between otherwise equal
   candidates.

--------------------------------------------------
OUTPUT FORMAT:

Return ONLY the index number (0-based) of the best candidate.
If none of the candidates are acceptable, return -1.
Return nothing else — just the number."""


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

# The LLM is held behind a callable so tests can inject a fake without needing a
# real OpenRouter key. CP's code hardcoded ChatOpenAI; we accept a factory so
# that assumption doesn't lock downstream users into one provider.
LLMFactory = Callable[[], object]


def _default_llm_factory() -> object:
    """Default: OpenRouter-backed ChatOpenAI, matching CP's existing setup.

    Falls back to the env var when no key is passed, same as CP. Changing model
    across the group becomes an env var (`CONCEPT_NORMALIZER_MODEL`) rather than
    a code edit here.
    """
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No OPENROUTER_API_KEY (or OPENAI_API_KEY) in the environment, and no "
            "llm_factory was passed. Agentic normalization needs an LLM."
        )
    return ChatOpenAI(
        model=os.environ.get("CONCEPT_NORMALIZER_MODEL", "openai/gpt-5.4-mini"),
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
        temperature=0,
        max_retries=2,
    )


# ---------------------------------------------------------------------------
# Agent 1
# ---------------------------------------------------------------------------

def _format_examples(examples: list[tuple[str, list[str]]]) -> str:
    """Few-shot examples the agent uses to pick the right style per target.

    SNOMED's examples come from CP; BSO-AD's would use ontology labels; LOINC's
    would use LOINC long-common-names. The profile owns these so the orchestrator
    stays target-agnostic.
    """
    out = []
    for raw, cands in examples:
        out.append(f"Input: {raw}\nOutput: {json.dumps(cands, ensure_ascii=False)}")
    return "\n\n".join(out)


def generate_candidates(
    raw_input: str,
    target: TargetProfile,
    llm_factory: LLMFactory = _default_llm_factory,
) -> list[str]:
    """Agent 1 — ask the LLM for 3-5 candidate phrases.

    On any failure returns [raw_input] rather than raising. CP's version does
    the same, and the reasoning is right: a candidate-generation failure must
    not block the downstream retrieval, because the raw input is itself a valid
    (if lower-recall) candidate.
    """
    llm = llm_factory()
    prompt = _AGENT1_PROMPT.format(
        target_name=target.name,
        purpose=target.purpose,
        style_hint=target.style_hint,
        examples=_format_examples(target.examples),
        raw_input=raw_input,
    )
    try:
        response = llm.invoke(prompt).content.strip()
    except Exception as exc:  # noqa: BLE001 — the LLM can fail in many ways
        log.warning("agent1: LLM call failed (%s); falling back to raw input", exc)
        return [raw_input]

    # Markdown fences happen in practice even with the explicit "no markdown"
    # instruction, so strip them rather than letting json.loads blow up.
    if response.startswith("```"):
        response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        log.warning("agent1: non-JSON response (%r); falling back to raw input", response[:200])
        return [raw_input]

    if not isinstance(parsed, list) or not parsed:
        log.warning("agent1: empty candidate list; falling back to raw input")
        return [raw_input]

    cleaned = [str(c).strip() for c in parsed if str(c).strip()]
    return cleaned or [raw_input]


# ---------------------------------------------------------------------------
# Agent 2
# ---------------------------------------------------------------------------

def _candidates_table(scored: list[dict]) -> str:
    """Human-readable table for Agent 2's prompt."""
    lines = []
    for i, c in enumerate(scored):
        hit: RetrievedConcept | None = c["hit"]
        if hit is None:
            lines.append(f"[{i}] Candidate: {c['candidate']!r} -> No match found")
        else:
            lines.append(
                f"[{i}] Candidate: {c['candidate']!r} -> "
                f"{hit.code} - {hit.name!r} "
                f"(semantic: {hit.semantic_score:.4f}, syntactic: {hit.syntactic_score:.0f})"
            )
    return "\n".join(lines)


def select_best(
    raw_input: str,
    scored: list[dict],
    target: TargetProfile,
    llm_factory: LLMFactory = _default_llm_factory,
) -> int:
    """Agent 2 — ask the LLM for the index of the best candidate (or -1).

    Returns -1 on any failure, which the caller treats as "fall back to the
    highest-scoring candidate". That mirrors CP's behaviour and keeps the
    pipeline degradable rather than brittle.
    """
    llm = llm_factory()
    prompt = _AGENT2_PROMPT.format(
        target_name=target.name,
        target_name_upper=target.name.upper(),
        raw_input=raw_input,
        candidates_table=_candidates_table(scored),
    )
    try:
        response = llm.invoke(prompt).content.strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("agent2: LLM call failed (%s); returning -1", exc)
        return -1

    try:
        return int(response)
    except ValueError:
        # Models occasionally decorate the index with surrounding text despite
        # the instruction. Fish the first signed integer out of the response
        # rather than giving up entirely.
        match = re.search(r"-?\d+", response)
        return int(match.group()) if match else -1


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AgenticResult:
    """What the orchestrator returns.

    Deliberately generic — no vocabulary-specific keys. The caller already has
    the target profile, so target identity is a property of the call, not of
    the result.
    """

    code: str | None
    name: str | None
    semantic_score: float | None
    syntactic_score: float | None
    candidate_used: str | None
    all_candidates: list[dict]
    chosen_via: str   # "agent2" | "fallback_highest_score" | "no_matches"


def normalize(
    raw_input: str,
    target: TargetProfile,
    retriever: Retriever | None = None,
    llm_factory: LLMFactory = _default_llm_factory,
) -> AgenticResult:
    """Full two-agent pipeline. Target-agnostic by construction.

    `retriever` is injectable for tests. In normal use it is resolved from
    target.index_dir via the module's singleton cache.
    """
    if retriever is None:
        retriever = Retriever(target.index_dir)

    # Agent 1 — generate candidate phrases
    candidates = generate_candidates(raw_input, target, llm_factory=llm_factory)
    log.info("agent1: produced %d candidates: %s", len(candidates), candidates)

    # Retrieve against the target index for each candidate
    scored: list[dict] = []
    for candidate in candidates:
        hits = retriever.search(candidate)
        best = hits[0] if hits else None
        scored.append({"candidate": candidate, "hit": best})
        if best is not None:
            log.info(
                "retriever: %r -> %s %r (sem=%.4f syn=%.0f)",
                candidate, best.code, best.name, best.semantic_score, best.syntactic_score,
            )
        else:
            log.info("retriever: %r -> no match", candidate)

    matched = [s for s in scored if s["hit"] is not None]
    if not matched:
        return AgenticResult(
            code=None, name=None, semantic_score=None, syntactic_score=None,
            candidate_used=None,
            all_candidates=[_candidate_record(s) for s in scored],
            chosen_via="no_matches",
        )

    # Agent 2 — pick the best
    chosen_idx = select_best(raw_input, scored, target, llm_factory=llm_factory)
    log.info("agent2: selected candidate index %d", chosen_idx)

    if 0 <= chosen_idx < len(scored) and scored[chosen_idx]["hit"] is not None:
        best = scored[chosen_idx]
        via = "agent2"
    else:
        # Agent 2 abstained or returned nonsense. Fall back to the highest
        # semantic-score candidate that got a match — CP's behaviour.
        best = max(matched, key=lambda s: s["hit"].semantic_score)
        via = "fallback_highest_score"

    hit: RetrievedConcept = best["hit"]
    return AgenticResult(
        code=hit.code,
        name=hit.name,
        semantic_score=hit.semantic_score,
        syntactic_score=hit.syntactic_score,
        candidate_used=best["candidate"],
        all_candidates=[_candidate_record(s) for s in scored],
        chosen_via=via,
    )


def _candidate_record(s: dict) -> dict:
    hit: RetrievedConcept | None = s["hit"]
    return {
        "candidate": s["candidate"],
        "code": hit.code if hit else None,
        "name": hit.name if hit else None,
        "semantic_score": hit.semantic_score if hit else None,
        "syntactic_score": hit.syntactic_score if hit else None,
    }
