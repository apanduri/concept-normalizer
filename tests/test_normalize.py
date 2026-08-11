"""The contract: text + target -> concept."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from concept_normalizer import (  # noqa: E402
    Concept,
    ListVocabulary,
    OmopVocabulary,
    Status,
    loinc,
    normalize,
    normalize_text,
    snomed,
)

# A miniature CONCEPT table covering the cases that matter.
CONCEPTS = [
    # id,        name,                            domain,        vocab,   class,  std, code
    (4169175, "Mini-mental state examination", "Measurement", "SNOMED", "Proc", "S", "MMSE1"),
    (42869861, "Mini-Mental State Examination", "Observation", "LOINC", "Survey", "S", "72106-8"),
    (4004884, "Treadmill", "Device", "SNOMED", "Object", "S", "TM1"),
    (4086506, "Frailty", "Condition", "SNOMED", "Finding", "S", "FR1"),
    (4151768, "Pack years", "Measurement", "SNOMED", "Finding", "S", "PY1"),
    (999001, "Deprecated thing", "Condition", "SNOMED", "Finding", None, "DEP1"),
    (999002, "Ethnicity", "Metadata", "SNOMED", "Finding", "S", "ETH1"),
]


def build_vocab(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE concept (
               concept_id INTEGER PRIMARY KEY, concept_name TEXT, domain_id TEXT,
               vocabulary_id TEXT, concept_class_id TEXT, standard_concept TEXT,
               concept_code TEXT, invalid_reason TEXT)"""
    )
    conn.executemany(
        "INSERT INTO concept VALUES (?,?,?,?,?,?,?,NULL)", CONCEPTS
    )
    conn.commit()
    conn.close()
    return path


class TestNormalizeText(unittest.TestCase):
    def test_folds_underscores_case_and_punctuation(self) -> None:
        self.assertEqual(normalize_text("Exercise_Equipment"), "exercise equipment")
        self.assertEqual(normalize_text("  Mini-Mental  State "), "mini mental state")

    def test_does_not_invent_synonyms(self) -> None:
        """Only surface folding — anything semantic must stay a real match problem."""
        self.assertNotEqual(normalize_text("MMSE"), normalize_text("Mini mental state"))


class TargetTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.vocab_path = build_vocab(Path(self.tmp.name) / "concept.db")

    def tearDown(self) -> None:
        self.tmp.cleanup()


class TestOmopTarget(TargetTestBase):
    def test_exact_match_is_mapped(self) -> None:
        t = OmopVocabulary(self.vocab_path, vocabulary_ids=("SNOMED",))
        r = normalize("Frailty", t)
        self.assertIs(r.status, Status.MAPPED)
        self.assertEqual(r.concept.concept_id, 4086506)
        self.assertEqual(r.concept.domain_id, "Condition")
        t.close()

    def test_case_and_underscores_do_not_prevent_a_match(self) -> None:
        t = snomed(self.vocab_path)
        self.assertIs(normalize("pack_years", t).status, Status.MAPPED)
        self.assertIs(normalize("PACK YEARS", t).status, Status.MAPPED)
        t.close()

    def test_unknown_term_is_unmapped_not_guessed(self) -> None:
        t = OmopVocabulary(self.vocab_path)
        r = normalize("Something nobody has heard of", t)
        self.assertIs(r.status, Status.UNMAPPED)
        self.assertIsNone(r.concept)
        t.close()

    def test_partial_overlap_is_not_a_match(self) -> None:
        """'Treadmill speed' must not silently resolve to 'Treadmill'."""
        t = OmopVocabulary(self.vocab_path)
        self.assertIs(normalize("Treadmill speed achieved", t).status, Status.UNMAPPED)
        t.close()

    def test_two_exact_matches_are_ambiguous_not_arbitrary(self) -> None:
        """MMSE exists as SNOMED/Measurement and LOINC/Observation.

        Picking one silently would decide which CDM table the fact lands in.
        """
        t = OmopVocabulary(self.vocab_path)
        r = normalize("Mini-mental state examination", t)
        self.assertIs(r.status, Status.AMBIGUOUS)
        self.assertEqual(len(r.candidates), 2)
        self.assertIn("Measurement", r.detail)
        self.assertIn("Observation", r.detail)
        t.close()

    def test_restricting_the_target_resolves_that_ambiguity(self) -> None:
        """Which is what 'user-selected target vocabulary' is for."""
        r = normalize("Mini-mental state examination", loinc(self.vocab_path))
        self.assertIs(r.status, Status.MAPPED)
        self.assertEqual(r.concept.concept_id, 42869861)
        self.assertEqual(r.concept.vocabulary_id, "LOINC")

    def test_non_standard_concepts_are_excluded_by_default(self) -> None:
        t = OmopVocabulary(self.vocab_path)
        self.assertIs(normalize("Deprecated thing", t).status, Status.UNMAPPED)
        t.close()

    def test_non_clinical_domains_are_excluded_by_default(self) -> None:
        """A Metadata concept is not something a clinical fact can be."""
        t = OmopVocabulary(self.vocab_path)
        self.assertIs(normalize("Ethnicity", t).status, Status.UNMAPPED)
        t.close()

    def test_device_match_is_reported_not_hidden(self) -> None:
        """The writer decides what to do with an odd domain; we report honestly."""
        t = OmopVocabulary(self.vocab_path)
        r = normalize("Treadmill", t)
        self.assertIs(r.status, Status.MAPPED)
        self.assertEqual(r.concept.domain_id, "Device")
        t.close()

    def test_empty_text_is_unmapped_not_an_error(self) -> None:
        t = OmopVocabulary(self.vocab_path)
        for text in ("", "   "):
            self.assertIs(normalize(text, t).status, Status.UNMAPPED)
        t.close()


class TestAliases(TargetTestBase):
    def test_reviewed_alias_wins_over_search(self) -> None:
        """A human correction must stick, not be re-litigated by the matcher."""
        t = OmopVocabulary(self.vocab_path)
        r = normalize("Treadmill", t, aliases={"Treadmill": "FR1"})
        self.assertIs(r.status, Status.MAPPED)
        self.assertEqual(r.concept.concept_id, 4086506)   # not the Device
        self.assertEqual(r.detail, "reviewed alias")
        t.close()

    def test_alias_to_a_code_absent_from_the_target_is_reported(self) -> None:
        t = OmopVocabulary(self.vocab_path)
        r = normalize("Whatever", t, aliases={"Whatever": "NOPE"})
        self.assertIs(r.status, Status.UNMAPPED)
        self.assertIn("absent from this target", r.detail)
        t.close()


class TestListTarget(unittest.TestCase):
    """'an ontology or a simple flat list' — a list has no concept_ids."""

    def setUp(self) -> None:
        self.target = ListVocabulary(
            [
                Concept(code="S1", name="Smoking status", vocabulary_id="MYLIST"),
                Concept(code="S2", name="Housing instability", vocabulary_id="MYLIST"),
            ],
            name="MYLIST",
        )

    def test_matches_by_name(self) -> None:
        r = normalize("housing instability", self.target)
        self.assertIs(r.status, Status.MAPPED)
        self.assertEqual(r.concept.code, "S2")

    def test_output_has_a_code_without_a_concept_id(self) -> None:
        r = normalize("Smoking status", self.target)
        self.assertEqual(r.concept.code, "S1")
        self.assertIsNone(r.concept.concept_id)

    def test_serialization_survives_a_missing_concept_id(self) -> None:
        d = normalize("Smoking status", self.target).to_dict()
        self.assertEqual(d["concept"]["code"], "S1")
        self.assertIsNone(d["concept"]["concept_id"])
        self.assertEqual(d["target"], "MYLIST")


if __name__ == "__main__":
    unittest.main(verbosity=2)
