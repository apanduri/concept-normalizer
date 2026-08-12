"""The contract: extracted free text + a target vocabulary -> a concept.

    normalize("MMSE", target)  ->  Normalization(concept=42869861 ..., status=MAPPED)

Deliberately conservative.  A normalizer that guesses is worse than one that
abstains, because a wrong concept becomes a clinical fact nobody can trace back
to a decision.  Measured over a real ontology, partial name matching produced the
wrong concept more often than the right one:

    "Diet"          -> "tolerating diet"                  (post-op feeding)
    "Substance Use" -> "substance use disorder severity"   (a rating scale)
    "Treadmill"     -> "Treadmill"                         (a physical object)

The last is the instructive one: an *exact* match, and still wrong for the
meaning intended.  So exact matches are reported as MAPPED but partial ones are
never auto-accepted — they come back as AMBIGUOUS with candidates attached, for a
human or a downstream reviewer to settle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from .aliases import AliasTable
from .target import Candidate, Concept, TargetVocabulary


class Status(str, Enum):
    MAPPED = "mapped"              # exactly one confident concept
    AMBIGUOUS = "ambiguous"        # several plausible; caller/human decides
    UNMAPPED = "unmapped"          # nothing plausible in this target
    PREMAPPED = "premapped"        # the input already carried a concept
    NOT_IN_TARGET = "not_in_target"  # reviewed and deliberately unmapped


@dataclass(slots=True)
class Normalization:
    """What the pipeline returns for one piece of extracted text."""

    text: str
    target: str
    status: Status
    concept: Concept | None = None
    candidates: list[Candidate] = field(default_factory=list)
    detail: str | None = None

    @property
    def is_resolved(self) -> bool:
        return self.concept is not None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "target": self.target,
            "status": self.status.value,
            "concept": None if self.concept is None else {
                "concept_id": self.concept.concept_id,
                "code": self.concept.code,
                "name": self.concept.name,
                "vocabulary_id": self.concept.vocabulary_id,
                "domain_id": self.concept.domain_id,
            },
            "candidates": [
                {
                    "concept_id": c.concept.concept_id,
                    "code": c.concept.code,
                    "name": c.concept.name,
                    "domain_id": c.concept.domain_id,
                    "score": c.score,
                    "match_kind": c.match_kind,
                }
                for c in self.candidates
            ],
            "detail": self.detail,
        }


def normalize(
    text: str,
    target: TargetVocabulary,
    *,
    aliases: dict[str, str] | AliasTable | None = None,
) -> Normalization:
    """Normalize one piece of extracted text against one target vocabulary.

    Two paths, in this order:

      1. a reviewed alias table, when the caller has one for its known terms
      2. searching the target vocabulary

    Aliases are consulted first because a reviewed decision must beat a search
    result — that is what makes a correction stick instead of being re-litigated
    every run.  Callers with no alias table (any other project reusing this) fall
    straight through to search, which is why the table is optional.

    `aliases` accepts an AliasTable or a plain {term: code} dict.
    """
    tname = getattr(target, "name", "target")
    if not text or not text.strip():
        return Normalization(text=text, target=tname, status=Status.UNMAPPED,
                             detail="empty text")

    if isinstance(aliases, AliasTable):
        alias = aliases.get(text, target=tname)
        if alias is not None:
            if alias.is_deliberate_nonmapping:
                # Someone checked and found nothing suitable. Distinct from "not
                # looked at yet", and it must stop the search fallback — otherwise
                # a reviewed "no" gets quietly overridden by a bad guess.
                return Normalization(
                    text=text, target=tname, status=Status.NOT_IN_TARGET,
                    detail=alias.note or "reviewed: no suitable concept in this target",
                )
            concept = _by_concept_id(target, alias.concept_id)
            if concept is not None:
                return Normalization(
                    text=text, target=tname, status=Status.MAPPED, concept=concept,
                    detail=f"reviewed alias ({aliases.name})"
                           + (f": {alias.note}" if alias.note else ""),
                )
            return Normalization(
                text=text, target=tname, status=Status.UNMAPPED,
                detail=f"alias points at concept_id {alias.concept_id}, "
                       f"absent from this target",
            )
    elif aliases:
        code = aliases.get(text) or aliases.get(text.strip().lower())
        if code:
            concept = target.by_code(code)
            if concept is not None:
                return Normalization(
                    text=text, target=tname, status=Status.MAPPED, concept=concept,
                    detail="reviewed alias",
                )
            return Normalization(
                text=text, target=tname, status=Status.UNMAPPED,
                detail=f"alias points at code {code!r}, absent from this target",
            )

    candidates = target.search(text)
    if not candidates:
        return Normalization(text=text, target=tname, status=Status.UNMAPPED,
                             detail="no exact name match in target")

    exact = [c for c in candidates if c.is_exact]
    if len(exact) == 1:
        return Normalization(text=text, target=tname, status=Status.MAPPED,
                             concept=exact[0].concept, candidates=candidates)

    if len(exact) > 1:
        # Same name, several concepts — often different domains, which would send
        # the fact to different tables.  Not ours to pick.
        domains = sorted({c.concept.domain_id or "?" for c in exact})
        return Normalization(
            text=text, target=tname, status=Status.AMBIGUOUS,
            candidates=candidates,
            detail=f"{len(exact)} exact matches across domains {domains}",
        )

    return Normalization(text=text, target=tname, status=Status.AMBIGUOUS,
                         candidates=candidates, detail="no exact match")


def _by_concept_id(target: TargetVocabulary, concept_id: int | None) -> Concept | None:
    """Aliases record concept_ids; not every target can look one up."""
    if concept_id is None:
        return None
    getter = getattr(target, "by_concept_id", None)
    return getter(concept_id) if getter else None


def normalize_all(
    texts: Iterable[str],
    target: TargetVocabulary,
    *,
    aliases: dict[str, str] | AliasTable | None = None,
) -> list[Normalization]:
    return [normalize(t, target, aliases=aliases) for t in texts]
