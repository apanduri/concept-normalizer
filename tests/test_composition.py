"""Tests for the layered normalize() in agentic_normalize.py.

Alias layer -> agentic retrieval -> exact-name fallback. Each layer short-
circuits on a hit, which is what makes the pipeline cheap when a decision is
already known.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from concept_normalizer.agentic import TargetProfile  # noqa: E402
from concept_normalizer.agentic_normalize import normalize as composed_normalize  # noqa: E402
from concept_normalizer.aliases import Alias, AliasTable  # noqa: E402
from concept_normalizer.normalize import Status  # noqa: E402
from concept_normalizer.retriever import RetrievedConcept  # noqa: E402


# Reuse the fakes from the other test file so behaviour stays consistent.
from tests.test_agentic import FakeLLM, FakeRetriever, hit  # noqa: E402


def profile() -> TargetProfile:
    return TargetProfile(
        name="SNOMED CT", style_hint="SNOMED-style", purpose="test",
        examples=[], index_dir="/does/not/exist",
    )


class TestLayerOrder(unittest.TestCase):
    def test_reviewed_alias_short_circuits_before_the_llm(self):
        """A signed-off decision must not be re-adjudicated by a model."""
        table = AliasTable(
            [Alias(source_term="t2dm", concept_id=73211009, target="SNOMED CT")],
            name="test",
        )
        llm = FakeLLM([])  # Would raise if called — proving the LLM is skipped.
        retriever = FakeRetriever({})
        result = composed_normalize(
            "t2dm", target_profile=profile(),
            exact_target=None, aliases=table,
            retriever=retriever, llm_factory=lambda: llm,
        )
        self.assertEqual(result.status, Status.MAPPED)
        self.assertEqual(result.concept.concept_id, 73211009)
        self.assertIn("reviewed alias", result.detail)
        self.assertEqual(llm.calls, [])

    def test_reviewed_nonmapping_blocks_even_a_good_retrieval(self):
        """'Checked, nothing suitable' must stay a reviewed no, not get overridden
        by a model that finds a plausible hit."""
        table = AliasTable(
            [Alias(source_term="undefined concept",
                   concept_id=None, target="SNOMED CT",
                   note="checked - nothing suitable")],
            name="test",
        )
        llm = FakeLLM(['["something"]', "0"])
        retriever = FakeRetriever({"something": [hit("99", "Something")]})
        result = composed_normalize(
            "undefined concept", target_profile=profile(),
            exact_target=None, aliases=table,
            retriever=retriever, llm_factory=lambda: llm,
        )
        self.assertEqual(result.status, Status.NOT_IN_TARGET)
        self.assertIsNone(result.concept)
        self.assertEqual(llm.calls, [])

    def test_agentic_path_runs_when_no_alias_hits(self):
        llm = FakeLLM(['["Hemoglobin A1c"]', "0"])
        retriever = FakeRetriever({"hemoglobin": [hit("4548-4", "Hemoglobin A1c")]})
        result = composed_normalize(
            "A1c result", target_profile=profile(),
            exact_target=None, aliases=None,
            retriever=retriever, llm_factory=lambda: llm,
        )
        self.assertEqual(result.status, Status.MAPPED)
        self.assertEqual(result.concept.code, "4548-4")
        self.assertIn("agentic", result.detail)

    def test_empty_text_short_circuits_at_the_top(self):
        """No reason to spin up an LLM for whitespace."""
        llm = FakeLLM([])
        result = composed_normalize(
            "   ", target_profile=profile(), exact_target=None,
            aliases=None, retriever=FakeRetriever({}),
            llm_factory=lambda: llm,
        )
        self.assertEqual(result.status, Status.UNMAPPED)
        self.assertEqual(llm.calls, [])

    def test_pipeline_degrades_when_no_target_is_configured(self):
        """A caller that forgot both target_profile and exact_target gets a
        clear UNMAPPED with a reason, not a crash."""
        result = composed_normalize(
            "diabetes", target_profile=None, exact_target=None,
            aliases=None, retriever=None, llm_factory=lambda: FakeLLM([]),
        )
        self.assertEqual(result.status, Status.UNMAPPED)
        self.assertIn("no target configured", result.detail)


if __name__ == "__main__":
    unittest.main()
