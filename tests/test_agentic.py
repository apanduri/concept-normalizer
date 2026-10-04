"""Tests for the generalized two-agent pipeline.

Written against CP's `snomed_normalizer.py` behaviour as the reference: anything
observable about the SNOMED pipeline today — fallbacks, abstention, score
tiebreaks — should produce the same result through the generalized version.

Fakes are used throughout because the real retriever needs SapBERT + FAISS +
torch, and the real LLM needs an OpenRouter key. Both are heavyweight; a tiny
behavioural shim exercises exactly the orchestration logic we care about.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from concept_normalizer.agentic import (  # noqa: E402
    TargetProfile,
    generate_candidates,
    normalize as agentic_normalize,
    select_best,
)
from concept_normalizer.retriever import RetrievedConcept  # noqa: E402


# --- fakes ---------------------------------------------------------------

class FakeLLM:
    """Deterministic LLM that returns pre-set strings for each call."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[str] = []

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        if not self._responses:
            raise AssertionError("FakeLLM ran out of responses")
        return _FakeResponse(self._responses.pop(0))


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeRetriever:
    """Keyword-indexed retriever that mimics the real one's shape."""

    def __init__(self, mapping: dict[str, list[RetrievedConcept]]):
        self.mapping = mapping

    def search(self, query: str, top_k: int = 10) -> list[RetrievedConcept]:
        for key, hits in self.mapping.items():
            if key.lower() in query.lower():
                return hits[:top_k]
        return []


def profile() -> TargetProfile:
    """A minimal SNOMED-shaped profile for testing. Real content doesn't matter;
    only the orchestration behaviour does."""
    return TargetProfile(
        name="SNOMED CT",
        style_hint="SNOMED-style",
        purpose="testing",
        examples=[("Type 2 Diabetes", ["Type 2 diabetes mellitus"])],
        index_dir="/does/not/exist",
    )


def hit(code: str, name: str, sem: float = 0.95, syn: float = 90.0) -> RetrievedConcept:
    return RetrievedConcept(code=code, name=name,
                            semantic_score=sem, syntactic_score=syn)


# --- Agent 1 -------------------------------------------------------------

class TestAgent1(unittest.TestCase):
    def test_parses_a_clean_json_array(self):
        llm = FakeLLM(['["Type 2 diabetes mellitus", "Diabetes mellitus"]'])
        got = generate_candidates("T2DM", profile(), llm_factory=lambda: llm)
        self.assertEqual(got, ["Type 2 diabetes mellitus", "Diabetes mellitus"])

    def test_strips_markdown_fences_the_model_adds_anyway(self):
        llm = FakeLLM(['```json\n["Diabetes mellitus"]\n```'])
        got = generate_candidates("diabetes", profile(), llm_factory=lambda: llm)
        self.assertEqual(got, ["Diabetes mellitus"])

    def test_falls_back_to_raw_input_on_non_json(self):
        """The downstream retrieval must still get something to work with."""
        llm = FakeLLM(["I think the best candidate is diabetes"])
        got = generate_candidates("diabetes mellitus", profile(),
                                  llm_factory=lambda: llm)
        self.assertEqual(got, ["diabetes mellitus"])

    def test_falls_back_to_raw_input_on_empty_list(self):
        llm = FakeLLM(["[]"])
        got = generate_candidates("diabetes", profile(), llm_factory=lambda: llm)
        self.assertEqual(got, ["diabetes"])

    def test_falls_back_to_raw_input_on_llm_exception(self):
        class BrokenLLM:
            def invoke(self, _):
                raise RuntimeError("API down")
        got = generate_candidates("diabetes", profile(), llm_factory=lambda: BrokenLLM())
        self.assertEqual(got, ["diabetes"])


# --- Agent 2 -------------------------------------------------------------

class TestAgent2(unittest.TestCase):
    def _scored(self):
        return [
            {"candidate": "diabetes", "hit": hit("44054006", "Type 2 diabetes mellitus")},
            {"candidate": "diabetes mellitus", "hit": hit("73211009", "Diabetes mellitus")},
        ]

    def test_returns_the_integer_the_llm_returned(self):
        llm = FakeLLM(["1"])
        got = select_best("T2DM", self._scored(), profile(), llm_factory=lambda: llm)
        self.assertEqual(got, 1)

    def test_extracts_the_integer_from_a_wrapped_response(self):
        llm = FakeLLM(["The best is index 0."])
        got = select_best("T2DM", self._scored(), profile(), llm_factory=lambda: llm)
        self.assertEqual(got, 0)

    def test_returns_minus_one_when_the_llm_abstains(self):
        """-1 is CP's signal for 'none acceptable' and must survive."""
        llm = FakeLLM(["-1"])
        got = select_best("random text", self._scored(), profile(), llm_factory=lambda: llm)
        self.assertEqual(got, -1)

    def test_returns_minus_one_on_llm_failure_not_a_crash(self):
        class BrokenLLM:
            def invoke(self, _):
                raise RuntimeError("timeout")
        got = select_best("diabetes", self._scored(), profile(),
                          llm_factory=lambda: BrokenLLM())
        self.assertEqual(got, -1)


# --- Orchestrator --------------------------------------------------------

class TestOrchestrator(unittest.TestCase):
    def test_target_is_not_hardcoded_to_snomed(self):
        """A non-SNOMED profile must flow through unchanged.

        The whole point of this port: swap target_profile.name and the pipeline
        continues to work, calling the same retriever and the same LLMs.
        """
        loinc = TargetProfile(name="LOINC", style_hint="LOINC-style",
                              purpose="labs", examples=[], index_dir="/x")
        llm = FakeLLM(['["Hemoglobin A1c"]', "0"])
        retriever = FakeRetriever({"hemoglobin": [hit("4548-4", "Hemoglobin A1c")]})
        result = agentic_normalize("A1c result", loinc,
                                    retriever=retriever, llm_factory=lambda: llm)
        self.assertEqual(result.code, "4548-4")
        self.assertEqual(result.name, "Hemoglobin A1c")

    def test_agent2_picks_the_abstraction_level_not_the_top_faiss_hit(self):
        """CP's design: Agent 2 can override the top-ranked candidate when a
        broader one is semantically more appropriate. Each Agent-1 candidate maps
        to a distinct retriever hit (mirroring the real pipeline), so the index
        Agent 2 returns identifies which candidate/hit pair wins.
        """
        retriever = FakeRetriever({
            "type 2":   [hit("44054006", "Type 2 diabetes mellitus", sem=0.99, syn=95)],
            "mellitus": [hit("73211009", "Diabetes mellitus",         sem=0.90, syn=85)],
        })
        llm = FakeLLM([
            '["type 2 diabetes", "diabetes mellitus"]',   # Agent 1
            "1",                                             # Agent 2 picks the broader concept
        ])
        result = agentic_normalize("diabetes", profile(),
                                    retriever=retriever, llm_factory=lambda: llm)
        self.assertEqual(result.code, "73211009")
        self.assertEqual(result.chosen_via, "agent2")

    def test_falls_back_to_highest_score_when_agent2_returns_minus_one(self):
        """CP's behaviour: Agent 2 abstaining does not mean no answer; it means
        'you pick', which translates to the highest-scoring valid candidate."""
        retriever = FakeRetriever({
            "diabetes": [hit("44054006", "Type 2 diabetes mellitus", sem=0.99)],
            "mellitus": [hit("73211009", "Diabetes mellitus", sem=0.90)],
        })
        llm = FakeLLM(['["diabetes", "mellitus"]', "-1"])
        result = agentic_normalize("sugar", profile(),
                                    retriever=retriever, llm_factory=lambda: llm)
        # Fallback picks the hit with higher semantic_score — the first one.
        self.assertEqual(result.code, "44054006")
        self.assertEqual(result.chosen_via, "fallback_highest_score")

    def test_no_matches_produces_an_explicit_result_not_an_exception(self):
        """A caller logging must be able to say what happened; raising buries it."""
        retriever = FakeRetriever({})
        llm = FakeLLM(['["totally unmappable"]'])
        result = agentic_normalize("random text", profile(),
                                    retriever=retriever, llm_factory=lambda: llm)
        self.assertIsNone(result.code)
        self.assertEqual(result.chosen_via, "no_matches")

    def test_all_candidates_is_preserved_for_auditing(self):
        """The all_candidates list is what makes a wrong answer explainable
        later. The orchestrator must preserve it even when Agent 2 picked a
        specific candidate."""
        retriever = FakeRetriever({
            "diabetes": [hit("44054006", "Type 2 diabetes")],
            "mellitus": [hit("73211009", "Diabetes mellitus")],
        })
        llm = FakeLLM(['["diabetes", "mellitus"]', "0"])
        result = agentic_normalize("diabetes", profile(),
                                    retriever=retriever, llm_factory=lambda: llm)
        self.assertEqual(len(result.all_candidates), 2)
        self.assertEqual(result.all_candidates[1]["code"], "73211009")


if __name__ == "__main__":
    unittest.main()
