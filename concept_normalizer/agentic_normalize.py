"""The composed normalize() with CP's two-agent pipeline in the middle.

Resolution order:

  1. Reviewed alias table (optional)
     A signed-off decision beats a model decision. Short-circuits before any LLM
     call, so the cheap path stays cheap.

  2. Two-agent retrieval (CP-style, target-abstracted)
     Candidate generation -> SapBERT+FAISS retrieval -> LLM selection.
     This is the primary path for free text.

  3. Exact-name search (existing exact matcher)
     Final fallback when the agentic path is unavailable (no API key, no index,
     test environment). Lets the module stay importable and partially useful
     without the heavy dependencies.

This file sits beside `normalize.py` rather than replacing it, so the exact-name
path and the ACTS alias flow that already exist keep working unchanged. Downstream
code (the writer) calls whichever composition it wants.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from .agentic import AgenticResult, TargetProfile, normalize as agentic_normalize
from .aliases import AliasTable
from .normalize import Normalization, Status
from .retriever import Retriever
from .target import Concept, OmopVocabulary, TargetVocabulary

log = logging.getLogger(__name__)


LLMFactory = Callable[[], object]


def normalize(
    text: str,
    *,
    target_profile: TargetProfile | None = None,
    exact_target: TargetVocabulary | None = None,
    aliases: AliasTable | dict | None = None,
    retriever: Retriever | None = None,
    llm_factory: LLMFactory | None = None,
) -> Normalization:
    """Resolve `text` through the composed pipeline.

    Either `target_profile` (for the agentic path) or `exact_target` (for the
    exact-name fallback) must be given; typically both are, so the pipeline can
    degrade cleanly when the heavy deps are missing.

    `aliases`, `retriever` and `llm_factory` are injectable to keep the function
    testable without touching real models or network.
    """
    target_name = (
        target_profile.name if target_profile is not None
        else getattr(exact_target, "name", "target")
    )

    if not text or not text.strip():
        return Normalization(
            text=text, target=target_name, status=Status.UNMAPPED,
            detail="empty text",
        )

    # 1. reviewed alias — short-circuit on a human decision
    if aliases is not None:
        hit = _check_aliases(text, aliases, exact_target, target_name)
        if hit is not None:
            return hit

    # 2. agentic retrieval — the primary free-text path
    if target_profile is not None:
        try:
            agentic = agentic_normalize(
                text, target_profile,
                retriever=retriever,
                llm_factory=llm_factory or _default_llm_factory,
            )
        except (FileNotFoundError, ImportError, RuntimeError) as exc:
            # Index missing, torch/faiss unavailable, or no LLM key. Degrade to
            # exact-name fallback rather than raise — the caller gets a clear
            # UNMAPPED with an explanation.
            log.info("agentic path unavailable (%s); falling back to exact-name", exc)
        else:
            if agentic.code is not None:
                return _agentic_to_normalization(agentic, target_name)

    # 3. exact-name fallback — the existing matcher
    if exact_target is not None:
        from .normalize import normalize as _exact_normalize
        return _exact_normalize(text, exact_target, aliases=aliases)

    return Normalization(
        text=text, target=target_name, status=Status.UNMAPPED,
        detail="no target configured (need target_profile or exact_target)",
    )


def _check_aliases(
    text: str,
    aliases: AliasTable | dict,
    exact_target: TargetVocabulary | None,
    target_name: str,
) -> Normalization | None:
    """Alias layer — unchanged from the existing normalize() path.

    Factored out so this file does not have to reimplement the AliasTable
    semantics; it just reuses them. Returns None when the aliases don't apply so
    the caller can proceed to the next stage.
    """
    if isinstance(aliases, AliasTable):
        alias = aliases.get(text, target=target_name)
        if alias is None:
            return None
        if alias.is_deliberate_nonmapping:
            return Normalization(
                text=text, target=target_name, status=Status.NOT_IN_TARGET,
                detail=alias.note or "reviewed: no suitable concept in this target",
            )
        if exact_target is not None and hasattr(exact_target, "by_concept_id"):
            concept = exact_target.by_concept_id(alias.concept_id)
            if concept is not None:
                return Normalization(
                    text=text, target=target_name, status=Status.MAPPED,
                    concept=concept,
                    detail=f"reviewed alias ({aliases.name})"
                           + (f": {alias.note}" if alias.note else ""),
                )
        # Can't resolve the alias to a Concept object — still a hit.
        return Normalization(
            text=text, target=target_name, status=Status.MAPPED,
            concept=Concept(
                code=str(alias.concept_id), name="", vocabulary_id=target_name,
                concept_id=alias.concept_id,
            ),
            detail=f"reviewed alias ({aliases.name})",
        )
    # Plain dict path for the simpler case.
    code = aliases.get(text) if isinstance(aliases, dict) else None
    if code and exact_target is not None:
        concept = exact_target.by_code(code)
        if concept is not None:
            return Normalization(
                text=text, target=target_name, status=Status.MAPPED,
                concept=concept, detail="reviewed alias (dict)",
            )
    return None


def _agentic_to_normalization(result: AgenticResult, target_name: str) -> Normalization:
    """Translate the agentic pipeline's output into the package's Normalization.

    Keeping two types rather than merging is deliberate: `AgenticResult` carries
    pipeline-specific signals (chosen_via, all_candidates) that `Normalization`
    shouldn't need to know about. The translation drops the pipeline detail into
    `detail` so a caller logging the result still has it, without widening the
    canonical result shape for everyone else.
    """
    concept = Concept(
        code=result.code,
        name=result.name or "",
        vocabulary_id=target_name,
        concept_id=_coerce_int(result.code),
    )
    detail = (
        f"agentic ({result.chosen_via}): candidate={result.candidate_used!r}, "
        f"semantic={result.semantic_score:.4f}, "
        f"syntactic={result.syntactic_score:.0f}"
    )
    return Normalization(
        text=result.candidate_used or "",
        target=target_name,
        status=Status.MAPPED,
        concept=concept,
        detail=detail,
    )


def _coerce_int(code: str | None) -> int | None:
    """OMOP concept ids are ints; SNOMED and LOINC are string codes.

    Returning None rather than raising keeps the Normalization usable for both
    vocabulary styles — the caller already handles the "no concept_id" case.
    """
    if code is None:
        return None
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def _default_llm_factory():
    """Deferred import so this module stays importable without langchain_openai."""
    from .agentic import _default_llm_factory as factory
    return factory()
