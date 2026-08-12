"""Reviewed alias tables — a human decision, recorded and versioned.

An alias says "this input text means this concept in this target", decided once by
someone who knows the vocabulary.  `normalize()` consults aliases before any
search, so a reviewed decision always wins and stays fixed.

Why this exists as a first-class input rather than a hardcoded dict:

  * Different sources have different known terms.  A rubric-based extractor emits
    field ids it defined itself ("mmse_score"); nothing can be inferred from that
    string, but the mapping is knowable once.  Other callers have no such table
    and need search instead — so alias tables must be optional and per-source.
  * Automatic matching is unreliable in a specific, measurable way: an exact name
    match can be the wrong sense of a word (OMOP's "Treadmill" is a physical
    object; a behavioural ontology means a person exercising).  A reviewed table
    is how that gets corrected permanently instead of re-litigated per run.
  * A wrong concept becomes a clinical fact that raises no error.  Storing the
    decision alongside who made it and why is the only way it stays auditable.

File format (CSV), one row per alias:

    source_term,concept_id,target,note,reviewed_by
    mmse_score,4169175,OMOP,"SNOMED Measurement — scored instrument",xai

`concept_id` may be blank to record a *deliberate* non-mapping — "we looked, there
is nothing suitable" — which is different from "nobody has checked yet", and stops
the search fallback from quietly producing a bad answer.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

BUILTIN_DIR = Path(__file__).parent / "alias_tables"


@dataclass(slots=True, frozen=True)
class Alias:
    source_term: str
    concept_id: int | None
    target: str
    note: str = ""
    reviewed_by: str = ""

    @property
    def is_deliberate_nonmapping(self) -> bool:
        """Recorded as 'checked, nothing suitable' rather than 'not yet looked at'."""
        return self.concept_id is None


class AliasTable:
    """Aliases for one source, keyed by term (case- and underscore-insensitive)."""

    def __init__(self, aliases: list[Alias], name: str = "aliases"):
        self.name = name
        self.aliases = aliases
        self._by_term: dict[str, Alias] = {}
        for a in aliases:
            self._by_term[_key(a.source_term)] = a

    def get(self, term: str, target: str | None = None) -> Alias | None:
        alias = self._by_term.get(_key(term))
        if alias is None:
            return None
        if target and alias.target and alias.target.upper() != target.upper():
            # An alias decided for one target says nothing about another: the same
            # term maps to different concepts in SNOMED and LOINC.
            return None
        return alias

    def concept_ids(self, target: str | None = None) -> dict[str, int]:
        """Term -> concept_id, skipping deliberate non-mappings."""
        out = {}
        for a in self.aliases:
            if a.concept_id is None:
                continue
            if target and a.target and a.target.upper() != target.upper():
                continue
            out[a.source_term] = a.concept_id
        return out

    def __len__(self) -> int:
        return len(self.aliases)

    def __repr__(self) -> str:
        mapped = sum(1 for a in self.aliases if a.concept_id is not None)
        return (f"AliasTable({self.name!r}, {mapped} mapped, "
                f"{len(self.aliases) - mapped} deliberate non-mappings)")


def _key(term: str) -> str:
    return term.replace("_", " ").replace("-", " ").strip().lower()


def load(path: Path, *, name: str | None = None) -> AliasTable:
    """Load an alias CSV.

    Recognised columns: source_term (or field_id / term / name), concept_id,
    target, note, reviewed_by.
    """
    # Leading '#' lines carry review status and rationale — worth having in a
    # curated table, so they are stripped before the header is read rather than
    # being mistaken for it.
    lines = [
        line for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"{path}: no rows (only comments?)")

    reader = csv.DictReader(lines)
    if not reader.fieldnames:
        raise ValueError(f"{path}: no header row")
    lower = {c.lower().strip(): c for c in reader.fieldnames}

    def col(*names: str) -> str | None:
        for n in names:
            if n in lower:
                return lower[n]
        return None

    term_col = col("source_term", "field_id", "term", "name")
    if term_col is None:
        raise ValueError(
            f"{path}: needs a source_term column (or field_id/term/name); "
            f"found {reader.fieldnames}"
        )
    id_col = col("concept_id", "reviewed_concept_id")
    if id_col is None:
        raise ValueError(f"{path}: needs a concept_id column")

    target_col = col("target", "target_vocabulary")
    note_col = col("note", "reviewer_note", "comment")
    by_col = col("reviewed_by", "reviewer")

    aliases: list[Alias] = []
    for i, row in enumerate(reader, start=2):
        term = (row.get(term_col) or "").strip()
        if not term:
            continue
        raw_id = (row.get(id_col) or "").strip()
        concept_id: int | None = None
        if raw_id:
            if not raw_id.lstrip("-").isdigit():
                raise ValueError(
                    f"{path}:{i}: concept_id {raw_id!r} is not an integer. Leave it "
                    f"blank to record a deliberate non-mapping."
                )
            concept_id = int(raw_id)
        aliases.append(
            Alias(
                source_term=term,
                concept_id=concept_id,
                target=(row.get(target_col) or "OMOP").strip() if target_col else "OMOP",
                note=(row.get(note_col) or "").strip() if note_col else "",
                reviewed_by=(row.get(by_col) or "").strip() if by_col else "",
            )
        )
    if not aliases:
        raise ValueError(f"{path}: no alias rows found")
    return AliasTable(aliases, name=name or path.stem)


def load_builtin(name: str) -> AliasTable:
    """Load a table shipped with the package, e.g. load_builtin("acts")."""
    path = BUILTIN_DIR / f"{name}.csv"
    if not path.exists():
        available = sorted(p.stem for p in BUILTIN_DIR.glob("*.csv"))
        raise FileNotFoundError(
            f"no built-in alias table {name!r}. Available: {available or 'none'}"
        )
    return load(path, name=name)


def available_builtin() -> list[str]:
    return sorted(p.stem for p in BUILTIN_DIR.glob("*.csv"))
