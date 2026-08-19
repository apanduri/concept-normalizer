"""Target vocabularies — what text gets normalized *to*.

Per Hongyu (2026-08-08): the pipeline "should support a user-selected target
vocabulary, such as OMOP, SNOMED CT, an OWL ontology, or a user-defined
vocabulary.  Its input should be the extracted free text together with the
selected target vocabulary, and its output should be the corresponding
normalized concept or code."

So the target is a runtime argument, never a build-time assumption.  Nothing in
this package knows about the OMOP CDM, patients, or insertion — those belong to
whatever consumes the output.  That separation is what lets a project with no CDM
at all use this.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


@dataclass(slots=True, frozen=True)
class Concept:
    """A concept in some target vocabulary.

    `code` is what the vocabulary itself calls it (a LOINC code, a SNOMED id, an
    ontology label).  `concept_id` is populated only for vocabularies that have
    OMOP surrogate keys; a plain OWL ontology or a spreadsheet list will not.
    """

    code: str
    name: str
    vocabulary_id: str
    concept_id: int | None = None
    domain_id: str | None = None
    is_standard: bool = False

    def __str__(self) -> str:
        ident = self.concept_id if self.concept_id is not None else self.code
        return f"{ident} {self.name!r} ({self.vocabulary_id})"


@dataclass(slots=True)
class Candidate:
    """One possible normalization, with why it was proposed."""

    concept: Concept
    score: float
    match_kind: str

    @property
    def is_exact(self) -> bool:
        return self.match_kind == "exact_name"


class TargetVocabulary(Protocol):
    """What every target must be able to do: look itself up by name."""

    name: str

    def search(self, text: str, limit: int = 5) -> list[Candidate]:
        """Candidates for this text, best first. Empty when nothing plausible."""
        ...

    def by_code(self, code: str) -> Concept | None:
        ...


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Fold surface differences that never carry meaning.

    Ontology labels use underscores ("Exercise_equipment"); clinical text uses
    spaces, punctuation and inconsistent case.  Anything beyond this — synonyms,
    abbreviations, word order — is a real matching problem and must NOT be hidden
    here, because silently-clever normalisation is how wrong matches get made.
    """
    text = text.replace("_", " ").lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text)


# ---------------------------------------------------------------------------
# OMOP-backed targets
# ---------------------------------------------------------------------------

# Domains a clinical fact can legitimately occupy.  Used to rank candidates, not
# to reject them: the caller decides what to do with an odd domain.
CLINICAL_DOMAINS = ("Observation", "Measurement", "Condition", "Drug",
                    "Procedure", "Device", "Specimen")


class OmopVocabulary:
    """Target = OMOP standard concepts, read from a CONCEPT table.

    Schema-compatible with computable_phenotype_library's concept.db, so the same
    file serves both projects.

    Also serves SNOMED / LOINC / RxNorm targets: those live *inside* the OMOP
    vocabulary, so restricting `vocabulary_ids` is all that differs.  That is why
    "target = SNOMED CT" needs no separate implementation.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        vocabulary_ids: Iterable[str] | None = None,
        standard_only: bool = True,
        domains: Iterable[str] | None = CLINICAL_DOMAINS,
    ):
        if not db_path.exists():
            raise FileNotFoundError(
                f"vocabulary not found at {db_path}. Point --vocab at an OMOP "
                f"concept.db (see scripts/build_omop_sqlite.py in "
                f"computable_phenotype_library)."
            )
        self.db_path = db_path
        self.vocabulary_ids = tuple(vocabulary_ids) if vocabulary_ids else None
        self.standard_only = standard_only
        self.domains = tuple(domains) if domains else None
        self.name = (
            "OMOP" if not self.vocabulary_ids else "+".join(self.vocabulary_ids)
        )
        self._conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self._conn.row_factory = sqlite3.Row
        self._index: dict[str, list[Concept]] | None = None

    # -- filters ---------------------------------------------------------

    def _where(self) -> tuple[str, list[object]]:
        clauses, params = [], []
        if self.standard_only:
            clauses.append("standard_concept = 'S'")
        if self.vocabulary_ids:
            clauses.append(
                f"vocabulary_id IN ({','.join('?' * len(self.vocabulary_ids))})"
            )
            params.extend(self.vocabulary_ids)
        if self.domains:
            clauses.append(f"domain_id IN ({','.join('?' * len(self.domains))})")
            params.extend(self.domains)
        return (" AND ".join(clauses) or "1=1"), params

    def _row_to_concept(self, row: sqlite3.Row) -> Concept:
        return Concept(
            code=(row["concept_code"] or "").strip(),
            name=(row["concept_name"] or "").strip(),
            vocabulary_id=(row["vocabulary_id"] or "").strip(),
            concept_id=int(row["concept_id"]),
            domain_id=(row["domain_id"] or "").strip() or None,
            is_standard=row["standard_concept"] == "S",
        )

    # -- lookup ----------------------------------------------------------

    def _build_index(self) -> dict[str, list[Concept]]:
        """Name -> concepts, built once.

        6.4M rows means a LIKE scan per query is unusable (minutes each); an
        in-memory index over the filtered subset answers in microseconds.
        """
        where, params = self._where()
        rows = self._conn.execute(
            f"""SELECT concept_id, concept_name, domain_id, vocabulary_id,
                       standard_concept, concept_code
                  FROM concept WHERE {where}""",
            params,
        ).fetchall()
        index: dict[str, list[Concept]] = {}
        for row in rows:
            concept = self._row_to_concept(row)
            index.setdefault(normalize_text(concept.name), []).append(concept)
        return index

    @property
    def index(self) -> dict[str, list[Concept]]:
        if self._index is None:
            self._index = self._build_index()
        return self._index

    def search(self, text: str, limit: int = 5) -> list[Candidate]:
        key = normalize_text(text)
        if not key:
            return []
        out = [
            Candidate(concept=c, score=1.0, match_kind="exact_name")
            for c in self.index.get(key, ())
        ]
        # Exact hits only.  Substring matching is deliberately NOT done here:
        # measured over a real ontology it produced the wrong concept more often
        # than the right one ("Diet" -> "tolerating diet", "Treadmill" -> the
        # physical object).  Partial matching belongs in a reviewed suggestion
        # workflow, not in an automatic answer.
        return out[:limit]

    def by_code(self, code: str) -> Concept | None:
        where, params = self._where()
        row = self._conn.execute(
            f"""SELECT concept_id, concept_name, domain_id, vocabulary_id,
                       standard_concept, concept_code
                  FROM concept WHERE concept_code = ? AND {where} LIMIT 1""",
            [code, *params],
        ).fetchone()
        return self._row_to_concept(row) if row else None

    def by_concept_id(self, concept_id: int) -> Concept | None:
        row = self._conn.execute(
            """SELECT concept_id, concept_name, domain_id, vocabulary_id,
                      standard_concept, concept_code
                 FROM concept WHERE concept_id = ? LIMIT 1""",
            (concept_id,),
        ).fetchone()
        return self._row_to_concept(row) if row else None

    def close(self) -> None:
        self._conn.close()


def snomed(db_path: Path, **kwargs) -> OmopVocabulary:
    """Target = SNOMED CT (as the CP Library used)."""
    return OmopVocabulary(db_path, vocabulary_ids=("SNOMED",), **kwargs)


def loinc(db_path: Path, **kwargs) -> OmopVocabulary:
    """Target = LOINC — where most instrument scores live."""
    return OmopVocabulary(db_path, vocabulary_ids=("LOINC",), **kwargs)


# Domains that hold ANSWERS rather than events.  OMOP separates the two: the
# concept for "smoking status" is an Observation, while the concept for "Former
# smoker" lives in 'Meas Value' — that is what value_as_concept_id points at.
# Searching for a value in the event domains finds nothing, and searching for an
# event in Meas Value finds the wrong kind of thing, so they need separate targets.
VALUE_DOMAINS = ("Meas Value", "Observation")

# Where units live, for unit_concept_id.
UNIT_DOMAINS = ("Unit",)


def value_target(db_path: Path, **kwargs) -> OmopVocabulary:
    """Target for normalizing a VALUE (an answer), not an event.

        value_target(db) -> "Former smoker" -> 45883458 (Meas Value / LOINC)
    """
    kwargs.setdefault("domains", VALUE_DOMAINS)
    return OmopVocabulary(db_path, **kwargs)


def unit_target(db_path: Path, **kwargs) -> OmopVocabulary:
    """Target for normalizing a UNIT, for unit_concept_id.

        unit_target(db) -> "year" -> 9448 (UCUM)
    """
    kwargs.setdefault("domains", UNIT_DOMAINS)
    return OmopVocabulary(db_path, **kwargs)


# ---------------------------------------------------------------------------
# list-backed target
# ---------------------------------------------------------------------------


class ListVocabulary:
    """Target = a user-defined list ("an ontology or a simple flat list").

    No concept_ids, because a spreadsheet has none.  Output is a code and a name,
    which is exactly what Hongyu's contract asks for.
    """

    def __init__(self, concepts: Iterable[Concept], name: str = "custom"):
        self.name = name
        self._concepts = list(concepts)
        self._index: dict[str, list[Concept]] = {}
        for c in self._concepts:
            self._index.setdefault(normalize_text(c.name), []).append(c)
        self._by_code = {c.code: c for c in self._concepts}

    def search(self, text: str, limit: int = 5) -> list[Candidate]:
        key = normalize_text(text)
        return [
            Candidate(concept=c, score=1.0, match_kind="exact_name")
            for c in self._index.get(key, ())
        ][:limit]

    def by_code(self, code: str) -> Concept | None:
        return self._by_code.get(code)

    def close(self) -> None:
        pass
