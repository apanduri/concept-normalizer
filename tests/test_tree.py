"""Tests for the tree-navigation engine.

The ontology shape, the ASCII tree format, and the novel_candidate fallback
are all behaviours documented in chart-review's bso-ad-sdk. These tests fix
them so a later edit does not accidentally diverge the port from the SDK.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from concept_normalizer.tree_ontology import (  # noqa: E402
    ConceptNode,
    TreeOntology,
    TreeTargetProfile,
    load_tree,
    normalize_tree,
)
from tests.test_agentic import FakeLLM  # noqa: E402


# ---------------------------------------------------------------------------
# BSO-AD-shaped sample ontology for the loader + walker tests
# ---------------------------------------------------------------------------

BSO_SAMPLE = {
    "_meta": {"version": "test"},            # must be ignored
    "Element_Relevant_to_Food": {
        "concepts": [
            {"label": "Food", "parent_label": None, "depth": 0},
            {"label": "Food_Insecurity", "parent_label": "Food", "depth": 1},
            {"label": "Mild_Food_Insecurity", "parent_label": "Food_Insecurity", "depth": 2},
            {"label": "Food_Access", "parent_label": "Food", "depth": 1},
        ],
    },
    "Economic_Stability": {
        "concepts": [
            {"label": "Employment", "parent_label": None, "depth": 0},
            {"label": "Unemployment", "parent_label": "Employment", "depth": 1},
        ],
    },
}


def write_ontology(path: Path) -> Path:
    path.write_text(json.dumps(BSO_SAMPLE))
    return path


# ---------------------------------------------------------------------------

class TestLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = write_ontology(Path(self.tmp.name) / "concepts.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_meta_keys_are_skipped(self):
        ontology = load_tree(self.path)
        self.assertNotIn("_meta", ontology.entity_types())

    def test_two_entity_types_are_separate_subtrees(self):
        """A concept that appears in two subtrees must stay in both — not get
        merged, which was the specific bug BSO-AD's workbench serves subtrees
        one at a time to avoid."""
        ontology = load_tree(self.path)
        self.assertEqual(set(ontology.entity_types()),
                         {"Element_Relevant_to_Food", "Economic_Stability"})

    def test_depth_is_recomputed_not_trusted_from_the_file(self):
        """The file might carry stale depth. Recomputing from parent_label is
        the only honest source — a hand-edited copy with a wrong depth would
        otherwise ship."""
        corrupt = dict(BSO_SAMPLE)
        corrupt["Element_Relevant_to_Food"] = {
            "concepts": [
                {"label": "Food", "parent_label": None, "depth": 99},
                {"label": "Food_Insecurity", "parent_label": "Food", "depth": 99},
            ],
        }
        path = Path(self.tmp.name) / "corrupt.json"
        path.write_text(json.dumps(corrupt))
        ontology = load_tree(path)
        depths = {n.label: n.depth for n in ontology.nodes_by_type("Element_Relevant_to_Food")}
        self.assertEqual(depths["Food"], 0)
        self.assertEqual(depths["Food_Insecurity"], 1)

    def test_rejects_a_non_object_entity_type(self):
        bad = Path(self.tmp.name) / "bad.json"
        bad.write_text(json.dumps({"A": ["not a dict"]}))
        with self.assertRaises(ValueError):
            load_tree(bad)


class TestAsciiTree(unittest.TestCase):
    def setUp(self) -> None:
        self.ontology = TreeOntology(name="test")
        self.ontology.by_entity_type["Food"] = [
            ConceptNode("Food", None, "Food", 0),
            ConceptNode("Food_Insecurity", "Food", "Food", 1),
            ConceptNode("Mild_Food_Insecurity", "Food_Insecurity", "Food", 2),
            ConceptNode("Food_Access", "Food", "Food", 1),
        ]

    def test_root_appears_first_as_a_bare_line(self):
        """Chart-review's workbench prints the root unindented so the agent
        can see the subtree name."""
        lines = self.ontology.ascii_tree("Food").splitlines()
        self.assertEqual(lines[0], "Food")

    def test_last_sibling_uses_the_corner_glyph(self):
        """└── for the last child, ├── for the rest — matches workbench format."""
        tree = self.ontology.ascii_tree("Food")
        self.assertIn("├── Food_Insecurity", tree)
        self.assertIn("└── Food_Access", tree)

    def test_descendants_of_a_non_last_sibling_use_the_vertical_bar(self):
        """`│` carries down past non-last siblings; `' '` past last ones."""
        tree = self.ontology.ascii_tree("Food")
        self.assertIn("│   └── Mild_Food_Insecurity", tree)

    def test_unknown_entity_type_raises(self):
        with self.assertRaises(KeyError):
            self.ontology.ascii_tree("Economic_Stability")


# ---------------------------------------------------------------------------

class TestNormalizeTree(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = write_ontology(Path(self.tmp.name) / "concepts.json")
        self.ontology = load_tree(self.path, name="BSO-AD-test")
        self.target = TreeTargetProfile(name="BSO-AD-test",
                                        ontology_path=self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_valid_label_is_returned_with_its_entity_type(self):
        llm = FakeLLM(["Food_Insecurity"])
        result = normalize_tree("not enough food this month", self.target,
                                ontology=self.ontology,
                                llm_factory=lambda: llm)
        self.assertEqual(result.label, "Food_Insecurity")
        self.assertEqual(result.entity_type, "Element_Relevant_to_Food")
        self.assertFalse(result.is_novel)
        self.assertEqual(result.code, result.label)   # alias accessor

    def test_novel_candidate_is_a_legitimate_outcome(self):
        """Not a failure. The caller distinguishes with is_novel; `code` is None."""
        llm = FakeLLM(["novel_candidate"])
        result = normalize_tree("obscure thing", self.target,
                                ontology=self.ontology,
                                llm_factory=lambda: llm)
        self.assertTrue(result.is_novel)
        self.assertIsNone(result.label)
        self.assertIsNone(result.code)

    def test_label_the_model_invented_becomes_novel_not_a_mapping(self):
        """BSO-AD policy: anything not in concepts.json is novel_candidate.

        This is the whole reason the loader keeps the label set and the engine
        validates against it — otherwise the model could silently coin a
        concept that no downstream project knows about.
        """
        llm = FakeLLM(["Totally_Invented_Concept"])
        result = normalize_tree("whatever", self.target,
                                ontology=self.ontology,
                                llm_factory=lambda: llm)
        self.assertTrue(result.is_novel)

    def test_whitespace_and_dash_variants_still_match(self):
        """`Food_Insecurity` and `food insecurity` are the same label stylistically;
        the engine permits that one small forgiveness because models occasionally
        drop the underscores."""
        llm = FakeLLM(["food insecurity"])
        result = normalize_tree("hunger", self.target,
                                ontology=self.ontology,
                                llm_factory=lambda: llm)
        self.assertEqual(result.label, "Food_Insecurity")
        self.assertFalse(result.is_novel)

    def test_quoted_response_is_tolerated(self):
        """Models sometimes add surrounding quotes despite the instruction."""
        llm = FakeLLM(['"Unemployment"'])
        result = normalize_tree("lost job", self.target,
                                ontology=self.ontology,
                                llm_factory=lambda: llm)
        self.assertEqual(result.label, "Unemployment")

    def test_llm_failure_degrades_to_novel_not_an_exception(self):
        """A caller logging mappings for a corpus must not crash on one bad
        LLM response. CP's agentic path has the same behaviour."""

        class BrokenLLM:
            def invoke(self, _):
                raise RuntimeError("API down")

        result = normalize_tree("anything", self.target,
                                ontology=self.ontology,
                                llm_factory=lambda: BrokenLLM())
        self.assertTrue(result.is_novel)

    def test_only_entity_types_narrows_the_trees_shown(self):
        """A project with one subtree worth of concepts should not have to show
        the agent the whole BSO-AD tree. The filter also avoids tempting the LLM
        with labels in a subtree that the project does not use."""
        narrow = TreeTargetProfile(
            name="BSO-AD-test",
            ontology_path=self.path,
            only_entity_types=("Economic_Stability",),
        )

        class RecordingLLM:
            def __init__(self):
                self.last_prompt = None

            def invoke(self, prompt: str):
                self.last_prompt = prompt
                return type("R", (), {"content": "Unemployment"})()

        llm = RecordingLLM()
        normalize_tree("lost job", narrow,
                       ontology=self.ontology,
                       llm_factory=lambda: llm)
        self.assertIn("Economic_Stability", llm.last_prompt)
        self.assertNotIn("Element_Relevant_to_Food", llm.last_prompt)


if __name__ == "__main__":
    unittest.main()
