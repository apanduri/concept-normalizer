#!/usr/bin/env python3
"""Live reference check — generalized retriever against expected SNOMED answers.

Goal: prove the port is behaviour-preserving on the one target CP has a
reference answer for (SNOMED). The two-agent pipeline needs an LLM key for
Agent 1 / Agent 2; the retriever does not, and the retriever is the generalized
replacement for CP's SnomedMapper.map(). If the generalized retriever surfaces
each expected concept in its top-K for the inputs CP's prompt is designed around,
the only remaining variable is the LLM selector — which is already behaviour-
tested with fakes in tests/test_agentic.py.

So the check we run here answers: "does our retriever behave like CP's mapper?"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from concept_normalizer.retriever import Retriever  # noqa: E402


@dataclass(frozen=True)
class Case:
    """One (input, expected concept name, how many hits before we call it a miss)."""

    raw_input: str
    expected_name_contains: str
    max_rank: int
    note: str = ""


# Inputs are either lifted from CP's prompt examples or are typical phrasing a
# clinician would use. The expected string is a case-insensitive substring that
# must appear in some hit within max_rank; this tolerates minor name variations
# ("Type 2 diabetes mellitus (disorder)" vs "Type 2 diabetes mellitus") without
# false negatives.
CASES = [
    Case("Type 2 Diabetes Mellitus", "Type 2 diabetes mellitus", 3,
         "CP's own prompt example"),
    Case("Diabetes mellitus", "Diabetes mellitus", 3,
         "exact match should rank at the top"),
    Case("Transgender and Gender Nonconforming (TGNC) Identification in EHRs",
         "Gender identity", 5,
         "messy real-world phrasing from CP's prompt"),
    Case("Atherosclerotic Cardiovascular Disease (ASCVD)",
         "Atherosclero", 5,
         "abbreviation + parenthetical — retrieval must see past it"),
    Case("T2DM", "diabetes", 10,
         "abbreviation CP's Agent 1 would expand before retrieval"),
    Case("A1c result", "Hemoglobin", 10,
         "lab shorthand"),
    Case("quit smoking", "Smoking cessation", 10,
         "social-determinant — SNOMED has Smoking cessation as the standard\n         # observation; Current/Former smoker live in LOINC"),
    Case("MMSE score", "Mini-mental state examination", 10,
         "cognitive assessment — note abbreviation in input, full name expected"),
]


def main() -> int:
    index_dir = Path(__file__).resolve().parent.parent / "indexes" / "snomed"
    print(f"[check] loading retriever from {index_dir}")
    retriever = Retriever(index_dir)
    print(f"[check] index has {retriever.index.ntotal:,} vectors")
    print()

    passed = 0
    misses: list[tuple[Case, str]] = []
    for case in CASES:
        hits = retriever.search(case.raw_input, top_k=case.max_rank)
        rank = next(
            (i for i, h in enumerate(hits)
             if case.expected_name_contains.lower() in h.name.lower()),
            None,
        )
        if rank is None:
            top = ", ".join(f"{h.name!r}" for h in hits[:3])
            misses.append((case, f"top hits: {top or '(none)'}"))
            print(f"  MISS  {case.raw_input!r}  ({case.note})")
            print(f"        expected a hit containing {case.expected_name_contains!r} "
                  f"in top {case.max_rank}; top: {top or '(none)'}")
            continue
        passed += 1
        top = hits[0]
        print(f"  OK    {case.raw_input!r}  ({case.note})")
        print(f"        expected {case.expected_name_contains!r} found at rank {rank}")
        print(f"        top hit: {top.code} {top.name!r}  "
              f"(sem={top.semantic_score:.3f} syn={top.syntactic_score:.0f})")

    print()
    print(f"{passed}/{len(CASES)} cases retrieved the expected concept within top-K")
    return 0 if not misses else 1


if __name__ == "__main__":
    sys.exit(main())
