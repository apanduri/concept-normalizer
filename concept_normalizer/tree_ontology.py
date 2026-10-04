"""Tree-ontology target — ported from chart-review's bso-ad-sdk.

Different mechanism from agentic.py (SapBERT retrieval): here the LLM gets the
WHOLE ontology as an ASCII tree and picks a concept_name from it. That works
better than embedding retrieval for small, deliberately-structured ontologies
where the right answer depends on where a leaf sits in the tree, not just its
name — "Food_Insecurity" under Food is a different thing from "Food_Insecurity"
under Economic_Stability, and a vector match would not catch the distinction.

Ported from:
    vendor/bso-ad-sdk/pipeline/workbench.py        (api_entity_types, api_subtree)
    vendor/bso-ad-sdk/claude_agent/ner_runner.py   (the agent prompt that drives it)
    vendor/bso-ad-sdk/ontology/concepts.json        (the data shape this reads)

Why keep it separate from agentic.py:

  * Tree navigation needs no FAISS index, so a tiny ontology (BSO-AD is ~660
    concepts) does not pay the index-build cost.
  * The agent sees STRUCTURE — depth, siblings, parent context — which the
    retrieval path cannot show. For "pick the right concept at the right depth"
    questions this is what the BSO-AD SDK is designed around.
  * No embedding model to load, so this path runs on a laptop without GPU.

What stays shared with agentic.py:

  * TargetProfile — same shape, so a project can list its targets uniformly.
  * The LLM factory — the same env var and the same ChatOpenAI configuration.
  * Alias tables remain the override layer, consulted BEFORE this engine runs.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ontology loading
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class ConceptNode:
    """One node in a tree ontology.

    Fields map 1:1 to BSO-AD's concepts.json. Keeping the names identical means
    a BSO-AD ontology file loads with no translation layer, which is also what
    chart-review's workbench.py reads directly.
    """

    label: str
    parent_label: str | None
    entity_type: str          # the top-level subtree this node belongs to
    depth: int = 0


@dataclass(slots=True)
class TreeOntology:
    """A tree ontology, loaded from BSO-AD-shaped JSON.

    One top-level key per entity_type, each wrapping {concepts: [{label,
    parent_label}, ...]}. Keys starting with `_` are metadata (BSO-AD stores
    version info under `_meta`) and are skipped — same behaviour as the
    workbench's `k.startswith('_')` filter.
    """

    name: str
    by_entity_type: dict[str, list[ConceptNode]] = field(default_factory=dict)

    def entity_types(self) -> list[str]:
        return list(self.by_entity_type.keys())

    def nodes_by_type(self, entity_type: str) -> list[ConceptNode]:
        """Raises on an unknown entity_type rather than returning [] — the agent
        asking for a subtree it was told about is a bug, not an empty answer."""
        if entity_type not in self.by_entity_type:
            raise KeyError(
                f"unknown entity_type {entity_type!r}. "
                f"Known: {sorted(self.by_entity_type)}"
            )
        return self.by_entity_type[entity_type]

    def ascii_tree(self, entity_type: str) -> str:
        """Render a subtree the way chart-review's workbench does.

        Lines look like:
            Element_Relevant_to_Food
            ├── Food_Insecurity
            │   └── Mild_Food_Insecurity
            └── Food_Access

        The agent reads this; the format is reproduced from workbench.py so a
        BSO-AD-trained prompt port behaves the same.
        """
        nodes = self.nodes_by_type(entity_type)
        children_of: dict[str | None, list[ConceptNode]] = defaultdict(list)
        for n in nodes:
            children_of[n.parent_label].append(n)

        lines = [entity_type]

        def walk(parent: str, prefix: str = "") -> None:
            kids = children_of.get(parent, [])
            for i, c in enumerate(kids):
                last = i == len(kids) - 1
                lines.append(f"{prefix}{'└── ' if last else '├── '}{c.label}")
                walk(c.label, prefix + ("    " if last else "│   "))

        walk(entity_type)
        return "\n".join(lines)

    def labels(self) -> set[str]:
        """Every valid concept_name, flattened. Used to validate the LLM's
        answer: if the agent returns a label that is not in this set, it is
        treated as a `novel_candidate` rather than a mapping — same policy as
        chart-review's SDK."""
        return {n.label for et in self.by_entity_type.values() for n in et}


def load_tree(path: Path, *, name: str | None = None) -> TreeOntology:
    """Load a BSO-AD-shaped `concepts.json`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(data).__name__}")

    ontology = TreeOntology(name=name or Path(path).stem)
    for key, block in data.items():
        if key.startswith("_"):
            continue
        if not isinstance(block, dict):
            raise ValueError(f"{path}: entity_type {key!r} is not an object")
        concepts = block.get("concepts", [])
        if not isinstance(concepts, list):
            raise ValueError(f"{path}: entity_type {key!r} .concepts is not a list")

        # Compute depth by walking from each node up to the entity_type root.
        # BSO-AD's own files include depth, but recomputing lets us accept a
        # trimmed or hand-edited copy without silently believing a stale field.
        parent_lookup = {c["label"]: c.get("parent_label") for c in concepts if "label" in c}

        def depth_of(label: str, seen: frozenset = frozenset()) -> int:
            if label in seen:
                return 0  # cycle guard
            parent = parent_lookup.get(label)
            if parent is None:
                return 0
            return 1 + depth_of(parent, seen | {label})

        nodes = [
            ConceptNode(
                label=c["label"],
                parent_label=c.get("parent_label"),
                entity_type=key,
                depth=depth_of(c["label"]),
            )
            for c in concepts
            if "label" in c
        ]
        ontology.by_entity_type[key] = nodes
    return ontology


# ---------------------------------------------------------------------------
# Agent prompt
# ---------------------------------------------------------------------------

# The prompt is a simplified descendent of chart-review's NER prompt. The real
# SDK uses tool-calling (list_entity_types, get_subtree, normalize_to_ontology)
# across a conversation; a single-shot prompt that shows the tree and asks for a
# label fits our shape better and keeps the port target-agnostic.
_TREE_PROMPT = """\
You are a clinical and social-determinants ontology assistant.

TASK:
Given an input phrase, pick the single best matching concept_name from a tree
ontology. The concept you pick must exist in the ontology; if nothing in the
ontology fits well, return the exact string "novel_candidate".

--------------------------------------------------
ONTOLOGY: {ontology_name}

Each subtree is one entity_type. The LEAF (deepest matching) concept is almost
always preferred over its parent, except when the input is genuinely about the
broader category — then return the parent.

{trees}

--------------------------------------------------
INPUT:
{raw_input}

--------------------------------------------------
OUTPUT FORMAT:

Return ONLY the chosen concept_name (one label from the ontology above), or the
literal string "novel_candidate" when nothing fits.
Do not include quotes, explanation, or any other text.
"""


# ---------------------------------------------------------------------------
# Target profile
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class TreeTargetProfile:
    """A tree-navigation target. One ontology, one prompt, one LLM call per query.

    Mirror of agentic.TargetProfile but for the tree engine, with the ontology
    file replacing the embedding index. Kept as its own type so a caller mixing
    both engines across targets can be explicit about which it wants.
    """

    name: str
    ontology_path: Path
    # Optional subset of entity_types to show the agent. Useful when an ontology
    # has non-clinical sections (BSO-AD's `_meta`, say) that are already filtered
    # by load_tree but a project might also want to narrow to just the social-
    # determinants branches.
    only_entity_types: tuple[str, ...] | None = None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TreeResult:
    """What the tree engine returns.

    `label` is None when the LLM returned `novel_candidate` or something not in
    the ontology — a legitimate outcome, not a failure. The caller distinguishes
    with `is_novel`.
    """

    label: str | None
    entity_type: str | None
    raw_response: str
    is_novel: bool

    @property
    def code(self) -> str | None:
        """Alias so this result slots into the same accessor patterns the
        AgenticResult uses (`.code` is what downstream code reads)."""
        return self.label


LLMFactory = Callable[[], object]


def _default_llm_factory():
    """Deferred import to keep this module importable without langchain_openai.
    Uses the same env vars as agentic.py so projects configure one LLM, not two."""
    from .agentic import _default_llm_factory as factory
    return factory()


def normalize_tree(
    raw_input: str,
    target: TreeTargetProfile,
    ontology: TreeOntology | None = None,
    llm_factory: LLMFactory = _default_llm_factory,
) -> TreeResult:
    """Pick a concept_name from the tree ontology for `raw_input`.

    `ontology` is injectable so tests can supply an in-memory TreeOntology
    without touching the filesystem. In normal use it is loaded from
    target.ontology_path.
    """
    if ontology is None:
        ontology = load_tree(target.ontology_path, name=target.name)

    entity_types = (
        list(target.only_entity_types) if target.only_entity_types
        else ontology.entity_types()
    )
    trees = "\n\n".join(ontology.ascii_tree(et) for et in entity_types)
    prompt = _TREE_PROMPT.format(
        ontology_name=ontology.name,
        trees=trees,
        raw_input=raw_input,
    )

    llm = llm_factory()
    try:
        response = llm.invoke(prompt).content.strip()
    except Exception as exc:  # noqa: BLE001 — the LLM can fail in many ways
        log.warning("tree: LLM call failed (%s); returning novel_candidate", exc)
        return TreeResult(label=None, entity_type=None,
                          raw_response="", is_novel=True)

    # Strip quotes the model sometimes adds despite the instruction.
    cleaned = response.strip().strip('"\'')

    if cleaned.lower() == "novel_candidate":
        return TreeResult(label=None, entity_type=None,
                          raw_response=response, is_novel=True)

    # The response must match an actual ontology label — otherwise the agent
    # hallucinated. This is also chart-review's policy: anything not in the
    # concepts.json becomes novel_candidate rather than being accepted.
    valid_labels = ontology.labels()
    if cleaned in valid_labels:
        entity_type = _entity_type_for(ontology, cleaned)
        return TreeResult(label=cleaned, entity_type=entity_type,
                          raw_response=response, is_novel=False)

    # Give the model one more chance: it may have returned the label with
    # whitespace or minor punctuation differences. Match case-insensitively,
    # but still require exact character content modulo spaces and dashes.
    normalized_response = _collapse(cleaned)
    for label in valid_labels:
        if _collapse(label) == normalized_response:
            entity_type = _entity_type_for(ontology, label)
            return TreeResult(label=label, entity_type=entity_type,
                              raw_response=response, is_novel=False)

    log.info("tree: response %r is not in the ontology; treating as novel", cleaned)
    return TreeResult(label=None, entity_type=None,
                      raw_response=response, is_novel=True)


def _entity_type_for(ontology: TreeOntology, label: str) -> str | None:
    """Find which subtree a label lives in.

    The same label CAN appear in two subtrees (BSO-AD does this deliberately for
    cross-cutting concepts), so this returns the first match — which matches the
    workbench's own behaviour, since it serves subtrees one at a time anyway.
    """
    for et, nodes in ontology.by_entity_type.items():
        for n in nodes:
            if n.label == label:
                return et
    return None


def _collapse(text: str) -> str:
    """Collapse to lowercase alphanumerics for fuzzy label-equality checks.

    Deliberately strict: we only want to match "Food_Insecurity" to
    "food insecurity" and "food-insecurity" — not to synonyms or paraphrases.
    Semantic matching is a different pipeline (agentic.py).
    """
    return re.sub(r"[^a-z0-9]", "", text.lower())
