#!/usr/bin/env python3
"""Export a reproducible SNOMED sample for the live reference check.

The full OMOP vocabulary has 349k standard SNOMED concepts in the clinical
domains; indexing all of them on CPU takes hours. For verification we only need
a sample that is (a) big enough to be a real retrieval problem and (b)
guaranteed to contain the specific concepts CP's prompt is designed around, so
a comparison against CP's expected answers is meaningful.

Composition:
  * every REFERENCE_NAMES concept (exact match, guarantees comparability)
  * every concept whose name mentions a reference-domain keyword (distractors)
  * a stratified sample of each clinical domain (noise so retrieval has to work)

The output is deterministic for a given vocabulary — same CSV every time.
"""

from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

# Reference answers taken from CP's own prompt examples (snomed_normalizer.py).
# A reproducibility floor: if an expected answer is NOT in SNOMED standard, the
# whole pipeline is excused for that case — but it must be in the index if we're
# claiming to verify it.
REFERENCE_NAMES = {
    "Gender identity finding",
    "Gender identity",
    "Transgender identity",
    "Type 2 diabetes mellitus",
    "Diabetes mellitus",
    "Non-insulin dependent diabetes",
    "Atherosclerotic cardiovascular disease",
    "Atherosclerosis",
    "Cardiovascular disease",
    "Diabetes classification",
    "Endocrine disorder",
    "Mini-mental state examination",
    "Hemoglobin A1c",
    "Current smoker",
    "Former smoker",
}


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        print("\nUsage: export_snomed_sample.py <concept.db> <out.csv>",
              file=sys.stderr)
        return 2
    db_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])

    conn = sqlite3.connect(db_path)
    rows: list[tuple[int, str]] = []
    seen: set[int] = set()

    def add(sql: str, params: tuple = ()) -> None:
        for cid, name in conn.execute(sql, params):
            if cid in seen or not name:
                continue
            seen.add(cid)
            rows.append((cid, name.strip()))

    # (a) exact reference names — the comparability floor
    for name in REFERENCE_NAMES:
        add("""SELECT concept_id, concept_name FROM concept
                WHERE vocabulary_id='SNOMED' AND standard_concept='S'
                  AND concept_name=? LIMIT 5""", (name,))

    # (b) distractors — forces retrieval to pick between competing hits
    for pattern in ("%diabetes%", "%gender%", "%atherosclerosis%",
                    "%cardiovascular%", "%mental state examination%",
                    "%hemoglobin%", "%tobacco%", "%smoker%"):
        add("""SELECT concept_id, concept_name FROM concept
                WHERE vocabulary_id='SNOMED' AND standard_concept='S'
                  AND LOWER(concept_name) LIKE ?
                  AND domain_id IN ('Condition','Observation','Measurement','Procedure')
                LIMIT 500""", (pattern,))

    # (c) stratified noise, 5k per clinical domain
    for domain in ("Condition", "Observation", "Measurement", "Procedure"):
        add("""SELECT concept_id, concept_name FROM concept
                WHERE vocabulary_id='SNOMED' AND standard_concept='S'
                  AND domain_id=? ORDER BY concept_id LIMIT 5000""", (domain,))

    conn.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "name"])
        for cid, name in rows:
            w.writerow([cid, name])
    print(f"wrote {len(rows):,} rows to {out_path}")

    # Report which reference names are present vs genuinely missing from SNOMED
    # standard — a reader of the live check output needs this context.
    names_in_sample = {n for _, n in rows}
    missing = REFERENCE_NAMES - names_in_sample
    if missing:
        print(f"note: {len(missing)} reference names not in standard SNOMED and "
              f"therefore cannot be found by any pipeline: {sorted(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
