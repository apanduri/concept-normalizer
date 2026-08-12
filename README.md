# concept-normalizer

Normalize extracted clinical text to concepts in a **user-selected target
vocabulary** — OMOP standard concepts, SNOMED CT, LOINC, or a custom list.

```
input:   extracted free text  +  selected target vocabulary
output:  the corresponding normalized concept or code
```

An independent pipeline. Nothing here knows about the OMOP CDM, patients, or data
insertion — a project with no database at all can use it for normalization alone.

## Quickstart

```bash
python3 -m concept_normalizer normalize --vocab /path/to/concept.db \
    "Mini-mental state examination" "Montreal cognitive assessment" "Pack years"
```

```
target: OMOP   (2,448,354 distinct names indexed)

  MAPPED     'Mini-mental state examination'
             -> 4169175 'Mini-mental state examination' (SNOMED)  domain=Measurement
  MAPPED     'Montreal cognitive assessment'
             -> 44808666 'Montreal cognitive assessment' (SNOMED)  domain=Measurement
  MAPPED     'Pack years'
             -> 4151768 'Pack years' (SNOMED)  domain=Measurement
```

Switch target with one flag — same code, no reconfiguration:

```bash
--target OMOP          # any standard concept
--target SNOMED        # SNOMED CT only
--target LOINC         # LOINC only
--target SNOMED,LOINC  # either
```

SNOMED, LOINC and RxNorm all live *inside* the OMOP vocabulary, so selecting one
is a filter rather than a different implementation.

## As a library

```python
from pathlib import Path
from concept_normalizer import OmopVocabulary, normalize, Status

target = OmopVocabulary(Path("concept.db"))
r = normalize("Hachinski ischemia score", target)

r.status                  # Status.MAPPED
r.concept.concept_id      # 4164973
r.concept.domain_id       # 'Measurement'
```

Five outcomes, always explicit:

| Status | Meaning |
|---|---|
| `MAPPED` | exactly one confident concept |
| `AMBIGUOUS` | several plausible — candidates attached, caller or human decides |
| `UNMAPPED` | nothing plausible in this target |
| `NOT_IN_TARGET` | reviewed, and deliberately not mapped |
| `PREMAPPED` | the input already carried a concept |

## Two resolution paths

A caller with known terms gets a reviewed answer; a caller without one still gets
an answer.

```
1. reviewed alias table   "mmse_score" -> 4169175        deterministic, signed off
2. search the target      "Pack years" -> 4151768        for anything not in the table
```

Aliases are consulted first, because a reviewed decision must beat a search result
— that is what makes a correction stick instead of being re-litigated every run.
Callers with no table fall straight through to search, which is why the table is
optional and per-source.

```bash
python3 -m concept_normalizer normalize --vocab concept.db --aliases acts \
    mmse_score mattis_drs "Pack years"
```

```
  MAPPED      'mmse_score'  -> 4169175 'Mini-mental state examination'   (reviewed alias)
  REVIEWED-NO 'mattis_drs'  (nothing containing Mattis — absent from OMOP)
  MAPPED      'Pack years'  -> 4151768 'Pack years'                      (search)
```

A blank `concept_id` in a table means **checked, nothing suitable** — a decision,
not an omission. It returns `NOT_IN_TARGET` and deliberately stops the search
fallback, so a reviewed "no" cannot be overridden by a guess.

Shipped tables live in `concept_normalizer/alias_tables/`; `--aliases` also takes
a path to your own CSV:

```csv
source_term,concept_id,target,note,reviewed_by
mmse_score,4169175,OMOP,"SNOMED Measurement — scored instrument",xai
mattis_drs,,OMOP,"absent from OMOP",xai
```

## Commands

| Command | What it does |
|---|---|
| `normalize` | text → concept, for one term or a file of them |
| `coverage` | how much of a source ontology exists in the target |
| `register` | register a source ontology as OMOP custom concepts |

## Design decisions worth knowing

**Exact name matches only.** Partial matching is deliberately not automatic.
Measured over a real ontology it produced the wrong concept more often than the
right one:

```
"Diet"           ->  "tolerating diet"                  (post-operative feeding)
"Substance Use"  ->  "substance use disorder severity"   (a rating scale)
"Treadmill"      ->  "Treadmill"                         (a physical object)
```

The last is the instructive one — an *exact* match that is still the wrong sense
of the word. So even exact matches into `Device` or `Procedure` are held for
review rather than auto-accepted when registering mappings.

**Reviewed decisions beat search results.** `normalize(..., aliases=...)` is
checked before any lookup, so a human correction always wins and stays fixed.

**Abstaining is better than guessing.** A wrong concept becomes a clinical fact
that nobody can trace back to a decision, and no error is ever raised. `UNMAPPED`
is a legitimate answer.

**Identity is (subtree, name), never the source's own id.** Real ontologies reuse
ids across subtrees — BSO-AD has 40 such collisions in 660 concepts — so keying on
the source id silently merges different concepts. Loading rejects a duplicate key
rather than accepting it.

## Registering an ontology as custom concepts

For source concepts with no equivalent in the target, the OHDSI-supported answer
is local concepts:

```bash
python3 -m concept_normalizer coverage --vocab concept.db \
    --ontology concepts.json --out build/coverage.csv
# review build/coverage.csv, then:
python3 -m concept_normalizer register --vocab concept.db \
    --ontology concepts.json --mappings build/coverage.csv --commit
```

This writes three things, not one:

- **CONCEPT** — one row per concept, ids in the OHDSI-reserved `>= 2,000,000,000`
  local range, hashed from the identity so they stay stable as the ontology grows
- **CONCEPT_ANCESTOR** — generated from the ontology's own parent/child tree,
  including each concept's self-row. Without these, a custom concept exists but is
  invisible to cohort SQL, which expands concept sets by descendant
- **CONCEPT_RELATIONSHIP** — `Maps to` rows where a standard equivalent exists

Domains are inherited: from the concept's own mapping first, then the nearest
ancestor with one, then `Observation`. Inheritance beats deciding each concept
individually because it is consistent by construction — siblings land in the same
domain. Every concept records **which rule decided it**, so a wrong one is
findable rather than silently baked in.

Re-running is incremental: existing concepts keep their ids, only new ones are
added.

## Ontology formats

| Format | Shape |
|---|---|
| `.json` | nested, with `{id, label, parent_label, depth}` nodes under subtree keys |
| `.csv` | flat list; recognises `name`/`label`, optional `parent`, `id`, `subtree` |

OWL is not implemented — convert to either shape first.

## Vocabulary file

Needs an OMOP `CONCEPT` table as SQLite, schema-compatible with
computable_phenotype_library's `concept.db` (built by its
`backend/build_omop_sqlite.py`), so the same file serves both projects. Not
included here — it is ~1.1 GB.

## Tests

```bash
python3 -m unittest discover -s tests
```

Stdlib only, no install step.
