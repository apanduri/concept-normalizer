"""Loading a source ontology, whatever shape it arrives in.

A target can be "an ontology or a simple flat list", so both are supported and
reduced to the same internal form: a concept with a name, an optional parent, and
a subtree it belongs to.

Identity is (subtree, name), never the source's own id.  Real ontologies reuse
ids across subtrees — BSO-AD has 40 such collisions across its 660 concepts —
while the pair has been verified unique.  Getting this wrong silently merges two
different concepts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .registry import SourceConcept


def load_nested_json(path: Path) -> list[SourceConcept]:
    """Ontology as nested JSON with {id, label, parent_label, depth} nodes.

    This is BSO-AD's `concepts.json` shape: top-level keys are subtrees, and
    concept nodes appear at any depth beneath them.  A `_meta` key is skipped.
    """
    doc = json.loads(path.read_text())
    out: list[SourceConcept] = []

    def walk(node: object, subtree: str) -> None:
        if isinstance(node, dict):
            if "id" in node and "label" in node:
                out.append(
                    SourceConcept(
                        entity_type=subtree,
                        concept_name=str(node["label"]),
                        source_id=str(node["id"]),
                        parent_name=(
                            str(node["parent_label"]) if node.get("parent_label") else None
                        ),
                        depth=node.get("depth"),
                    )
                )
            for value in node.values():
                walk(value, subtree)
        elif isinstance(node, list):
            for item in node:
                walk(item, subtree)

    for key, value in doc.items():
        if key == "_meta":
            continue
        walk(value, key)
    return _check_unique(out, path)


def load_flat_csv(path: Path, *, subtree: str = "custom") -> list[SourceConcept]:
    """Ontology as a flat list — "a simple flat list" in the requirement.

    Recognised columns (all optional except a name):
        name / label / concept_name   the term
        parent / parent_label         optional hierarchy
        id / code                     the source's own identifier
        subtree / entity_type / group which branch it belongs to
    """
    out: list[SourceConcept] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"{path}: no header row")
        lower = {c.lower(): c for c in reader.fieldnames}

        def col(*names: str) -> str | None:
            for n in names:
                if n in lower:
                    return lower[n]
            return None

        name_col = col("name", "label", "concept_name", "term")
        if name_col is None:
            raise ValueError(
                f"{path}: need a name column (one of name/label/concept_name/term); "
                f"found {reader.fieldnames}"
            )
        parent_col = col("parent", "parent_label", "parent_name")
        id_col = col("id", "code", "source_id")
        subtree_col = col("subtree", "entity_type", "group", "category")

        for row in reader:
            name = (row.get(name_col) or "").strip()
            if not name:
                continue
            out.append(
                SourceConcept(
                    entity_type=(row.get(subtree_col) or subtree).strip()
                    if subtree_col else subtree,
                    concept_name=name,
                    source_id=(row.get(id_col) or "").strip() or None if id_col else None,
                    parent_name=(row.get(parent_col) or "").strip() or None
                    if parent_col else None,
                    depth=None,
                )
            )
    return _check_unique(out, path)


def load(path: Path, **kwargs) -> list[SourceConcept]:
    """Dispatch on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_nested_json(path)
    if suffix in (".csv", ".tsv", ".txt"):
        return load_flat_csv(path, **kwargs)
    raise ValueError(
        f"{path}: unsupported ontology format {suffix!r}. Supported: .json (nested), "
        f".csv (flat list). OWL is not implemented — convert to either shape first."
    )


def _check_unique(concepts: list[SourceConcept], path: Path) -> list[SourceConcept]:
    seen: set[str] = set()
    for c in concepts:
        if c.key in seen:
            raise ValueError(
                f"{path}: key collision on {c.key!r} — (subtree, name) must be unique, "
                f"since it is the identity used for stable concept ids"
            )
        seen.add(c.key)
    if not concepts:
        raise ValueError(f"{path}: no concepts found")
    return concepts
