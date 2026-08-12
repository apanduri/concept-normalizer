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


class TestAliasTables(TargetTestBase):
    """Option A (reviewed table) first, option B (search) as fallback."""

    def table(self, rows: str) -> "object":
        from concept_normalizer import aliases as alias_mod
        p = Path(self.tmp.name) / "t.csv"
        p.write_text("source_term,concept_id,target,note,reviewed_by\n" + rows)
        return alias_mod.load(p, name="t")

    def test_reviewed_alias_beats_search(self) -> None:
        """The table says Frailty; search would say the Device. Table wins."""
        t = OmopVocabulary(self.vocab_path)
        table = self.table("Treadmill,4086506,OMOP,reviewed,xai\n")
        r = normalize("Treadmill", t, aliases=table)
        self.assertIs(r.status, Status.MAPPED)
        self.assertEqual(r.concept.concept_id, 4086506)
        self.assertIn("reviewed alias", r.detail)
        t.close()

    def test_term_not_in_table_falls_through_to_search(self) -> None:
        """A caller with no entry for a term must still get an answer."""
        t = OmopVocabulary(self.vocab_path)
        table = self.table("something_else,4086506,OMOP,,\n")
        r = normalize("Pack years", t, aliases=table)
        self.assertIs(r.status, Status.MAPPED)
        self.assertEqual(r.concept.concept_id, 4151768)
        t.close()

    def test_blank_concept_id_is_a_reviewed_no_not_a_gap(self) -> None:
        """A reviewed 'nothing suitable' must NOT be overridden by search."""
        t = OmopVocabulary(self.vocab_path)
        table = self.table("Frailty,,OMOP,checked - nothing suitable,xai\n")
        r = normalize("Frailty", t, aliases=table)
        self.assertIs(r.status, Status.NOT_IN_TARGET)
        self.assertIsNone(r.concept)          # search WOULD have found 4086506
        self.assertIn("nothing suitable", r.detail)
        t.close()

    def test_matching_is_case_and_underscore_insensitive(self) -> None:
        t = OmopVocabulary(self.vocab_path)
        table = self.table("mmse_score,4169175,OMOP,,\n")
        for term in ("mmse_score", "MMSE Score", "mmse-score"):
            self.assertEqual(
                normalize(term, t, aliases=table).concept.concept_id, 4169175, term
            )
        t.close()

    def test_alias_for_another_target_is_ignored(self) -> None:
        """The same term maps to different concepts in SNOMED and LOINC."""
        table = self.table("Frailty,4086506,SNOMED,,\n")
        r = normalize("Frailty", loinc(self.vocab_path), aliases=table)
        self.assertIsNot(r.status, Status.MAPPED)

    def test_alias_pointing_at_a_missing_concept_is_reported(self) -> None:
        t = OmopVocabulary(self.vocab_path)
        table = self.table("whatever,999999999,OMOP,,\n")
        r = normalize("whatever", t, aliases=table)
        self.assertIs(r.status, Status.UNMAPPED)
        self.assertIn("absent from this target", r.detail)
        t.close()

    def test_non_integer_concept_id_is_rejected_at_load(self) -> None:
        from concept_normalizer import aliases as alias_mod
        p = Path(self.tmp.name) / "bad.csv"
        p.write_text("source_term,concept_id\nx,not-a-number\n")
        with self.assertRaises(ValueError) as ctx:
            alias_mod.load(p)
        self.assertIn("not an integer", str(ctx.exception))

    def test_comment_lines_are_not_mistaken_for_the_header(self) -> None:
        from concept_normalizer import aliases as alias_mod
        p = Path(self.tmp.name) / "c.csv"
        p.write_text("# status: draft\n# reviewed by nobody yet\n"
                     "source_term,concept_id\nmmse_score,4169175\n")
        table = alias_mod.load(p)
        self.assertEqual(len(table), 1)


class TestShippedActsTable(unittest.TestCase):
    def test_acts_table_loads_and_is_split_between_mapped_and_reviewed_no(self) -> None:
        from concept_normalizer import aliases as alias_mod
        self.assertIn("acts", alias_mod.available_builtin())
        table = alias_mod.load_builtin("acts")
        mapped = table.concept_ids()
        self.assertEqual(len(mapped), 13)
        self.assertEqual(mapped["mmse_score"], 4169175)
        self.assertTrue(table.get("mattis_drs").is_deliberate_nonmapping)

    def test_every_non_mapping_records_why(self) -> None:
        """A blank concept_id with no explanation is indistinguishable from neglect."""
        from concept_normalizer import aliases as alias_mod
        for alias in alias_mod.load_builtin("acts").aliases:
            if alias.is_deliberate_nonmapping:
                self.assertTrue(alias.note, f"{alias.source_term} has no note")
