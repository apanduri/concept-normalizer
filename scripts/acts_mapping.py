#!/usr/bin/env python3
"""Draft an ACTS field -> OMOP concept mapping, for human review.

ACTS is a chart-review rubric: 29 typed fields (MMSE score, APOE genotype,
smoking status, ...) rather than free-text spans.  Each field needs one OMOP
concept before its value can be written to a CDM domain table.

This proposes candidates and shows the evidence for each.  It does NOT decide —
the output is a review sheet.  Two reasons that matters:

  * An exact name match can be the wrong sense of a word.  OMOP's "Treadmill"
    (4004884) is a physical object, not a person exercising.
  * Where a field matches concepts in two domains, the choice determines which
    CDM table the value lands in.  MMSE exists as SNOMED/Measurement and
    LOINC/Observation; those are different tables and not interchangeable.

Search terms come from each field's own prompt text, plus hand-written aliases
for instruments whose OMOP name differs from how the rubric phrases it.

    python3 scripts/acts_mapping.py --vocab vocab/concept.db \
        --fields /path/to/acts_fields.json --out build/acts_mapping.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from concept_normalizer.target import OmopVocabulary, normalize_text  # noqa: E402

# Search terms per field.  The first hit wins, so order is most-specific-first.
# Instrument names are spelled as OMOP spells them, which is often not how the
# rubric phrases the question.
SEARCH_TERMS: dict[str, list[str]] = {
    "mmse_score": ["Mini-mental state examination", "Mini-Mental State Examination"],
    "mmse_severity": ["Mini-mental state examination"],
    "moca_score": ["Montreal cognitive assessment"],
    "moca_severity": ["Montreal cognitive assessment"],
    "cdr_global": ["Clinical dementia rating scale", "Clinical Dementia Rating"],
    "cdr_severity": ["Clinical dementia rating scale"],
    "hachinski_score": ["Hachinski ischemia score"],
    "mattis_drs": ["Mattis Dementia Rating Scale", "Dementia rating scale"],
    "tics_score": ["Telephone interview for cognitive status"],
    "npi_total": ["Neuropsychiatric Inventory Questionnaire", "Neuropsychiatric inventory"],
    "gds_depression_score": ["Geriatric depression scale"],
    "cornell_csdd": ["Cornell scale for depression in dementia", "Cornell scale"],
    "gds_stage": ["Global deterioration scale", "Reisberg global deterioration scale"],
    "impaired_cognition": ["Cognitive impairment", "Mild cognitive impairment"],
    "apoe_genotype": ["Apolipoprotein E genotype", "Apolipoprotein E gene"],
    "apoe2": ["Apolipoprotein E2", "Apolipoprotein E allele"],
    "apoe3": ["Apolipoprotein E3", "Apolipoprotein E allele"],
    "apoe4": ["Apolipoprotein E4", "Apolipoprotein E allele"],
    "education_years": ["Highest level of education", "Years of education", "Education"],
    "smoking_status": ["Smoking status", "Tobacco smoking status"],
    "smoking_duration": ["Duration of smoking", "Years smoked"],
    "pack_year": ["Pack years", "Cigarette pack-years"],
    "pack_per_day": ["Cigarettes smoked per day", "Number of cigarettes smoked per day"],
    "quit_time": ["Date quit smoking", "Year quit smoking"],
    "postmenopause": ["Postmenopausal state", "Postmenopause"],
    "lmp_date": ["Date of last menstrual period", "Last menstrual period"],
    "allergen": ["Allergy to substance", "Allergen"],
    "vaccine_name": ["Vaccination", "Immunization"],
    "vaccine_category": ["Vaccination", "Immunization"],
}

# What a broader substring search turned up for the fields with no exact match,
# and why it does not settle them.  Recorded so a reviewer does not repeat the
# search: these are mostly the wrong SENSE of the words, which is the same trap as
# OMOP's "Treadmill" being a physical object.
NO_MATCH_FINDINGS: dict[str, str] = {
    "cornell_csdd": "closest is 37168624 'Cornell medical index-health questionnaire' "
                    "— a DIFFERENT instrument. CSDD appears absent; custom concept likely.",
    "mattis_drs": "nothing containing 'Mattis' or 'dementia rating scale'. Absent.",
    "tics_score": "nothing containing 'telephone interview'. Absent.",
    "smoking_duration": "nothing for smoking duration in years. Absent — though it may "
                        "be derivable from quit year and start year rather than stored.",
    "impaired_cognition": "only severity-specific conditions exist (439795 Minimal, "
                          "45765899 Moderate, 45765900 Severe cognitive impairment). "
                          "ACTS asks a yes/no, so this needs a MODELLING decision: one "
                          "general concept, or map onto a severity concept?",
    "allergen": "OMOP models allergy as 'Allergy to <substance>' (439224 Allergy to "
                "drug, 4306169 Allergy to rubber). The allergen is part of the concept, "
                "not a value — so one concept per allergen, not one field concept. "
                "MODELLING decision.",
    "vaccine_name": "vaccines are Drug-domain concepts (35894915 COVID-19 vaccine); "
                    "'Vaccine CVX code' hits are code-system references, not vaccines. "
                    "Likely DRUG_EXPOSURE per vaccine rather than one field concept.",
    "pack_per_day": "hits are qualified variants ('per day --during pregnancy') or "
                    "lifetime totals. A plain 'cigarettes per day' concept was not found.",
    "quit_time": "hits are PROMIS attitude items ('If I quit smoking I will breathe "
                 "easier') — questionnaire beliefs, not a quit date. Wrong sense.",
}

# Fields the rubric computes from another field rather than reading from a note.
# They carry no independent clinical fact, so they should NOT be inserted — the
# raw field already is.
DERIVED_FIELDS = {"mmse_severity", "moca_severity", "cdr_severity",
                  "apoe2", "apoe3", "apoe4", "vaccine_category"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--fields", type=Path, required=True,
                    help="JSON list of {field_id, type, group, prompt}")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "acts_mapping.csv")
    args = ap.parse_args()

    fields = json.loads(args.fields.read_text())
    target = OmopVocabulary(args.vocab)
    print(f"[acts] {len(fields)} fields vs {len(target.index):,} indexed concept names\n")

    rows = []
    for f in fields:
        fid = f["field_id"]
        terms = SEARCH_TERMS.get(fid, [re.sub(r"_", " ", fid)])
        hits, used_term = [], ""
        for term in terms:
            hits = target.index.get(normalize_text(term), [])
            if hits:
                used_term = term
                break

        derived = fid in DERIVED_FIELDS
        if derived:
            status, note = "derived_do_not_insert", "computed from another field"
        elif not hits:
            status = "no_match"
            note = NO_MATCH_FINDINGS.get(fid, f"tried: {'; '.join(terms)}")
        elif len(hits) == 1:
            status, note = "single", ""
        else:
            domains = sorted({h.domain_id or "?" for h in hits})
            status = "choose_domain" if len(domains) > 1 else "choose_one"
            note = f"{len(hits)} candidates, domains {domains}"

        best = hits[0] if hits else None
        rows.append({
            "field_id": fid,
            "group": f.get("group", ""),
            "value_type": f.get("type", ""),
            "status": status,
            "concept_id": best.concept_id if best else "",
            "concept_name": best.name if best else "",
            "domain_id": best.domain_id if best else "",
            "vocabulary_id": best.vocabulary_id if best else "",
            "n_candidates": len(hits),
            "matched_term": used_term,
            "all_candidates": " | ".join(
                f"{h.concept_id}:{h.name}[{h.domain_id}/{h.vocabulary_id}]" for h in hits[:5]
            ),
            "note": note,
            "REVIEWED_concept_id": "",   # a human fills this in
            "REVIEWER_NOTE": "",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    order = {"single": 0, "choose_domain": 1, "choose_one": 2,
             "no_match": 3, "derived_do_not_insert": 4}
    print(f"{'field_id':<22} {'status':<22} candidate")
    for r in sorted(rows, key=lambda r: (order[r["status"]], r["field_id"])):
        cand = (f"{r['concept_id']} {r['concept_name'][:36]} "
                f"[{r['domain_id']}/{r['vocabulary_id']}]") if r["concept_id"] else r["note"][:52]
        print(f"  {r['field_id']:<22} {r['status']:<22} {cand}")

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"\nReview sheet: {args.out}")
    print("Fill REVIEWED_concept_id for anything not 'single', and sanity-check the")
    print("'single' rows too — an exact name match can still be the wrong sense.")
    target.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
