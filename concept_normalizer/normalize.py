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

from .target import Candidate, Concept, TargetVocabulary


class Status(str, Enum):
    MAPPED = "mapped"              # exactly one confident concept
    AMBIGUOUS = "ambiguous"        # several plausible; caller/human decides
    UNMAPPED = "unmapped"          # nothing plausible in this target
    PREMAPPED = "premapped"        # the input already carried a concept


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
    aliases: dict[str, str] | None = None,
) -> Normalization:
    """Normalize one piece of extracted text against one target vocabulary.

    `aliases` maps input text to a code in the target, for mappings a human has
    already decided.  Checked first, because a reviewed decision must always beat
    a search result — that is what makes corrections stick.
    """
    tname = getattr(target, "name", "target")
    if not text or not text.strip():
        return Normalization(text=text, target=tname, status=Status.UNMAPPED,
                             detail="empty text")

    if aliases:
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


def normalize_all(
    texts: Iterable[str],
    target: TargetVocabulary,
    *,
    aliases: dict[str, str] | None = None,
) -> list[Normalization]:
    return [normalize(t, target, aliases=aliases) for t in texts]
