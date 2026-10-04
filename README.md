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
| `agentic` | normalize with the two-agent LLM pipeline against a target index |

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

## Vocabulary registration (moved)

The CONCEPT / CONCEPT_ANCESTOR / `Maps to` writer that used to live here
moved to [omop-nlp-writer](https://github.com/apanduri/omop-nlp-writer),
because registering concepts into a CDM is a writer concern — not a
normalization one. A project with no CDM should be able to use this
package for normalization alone, and the normalizer's dependency surface
stays narrower as a result.

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

---

## Two-agent LLM-driven normalization (ported from CP)

A free-text phrase ("Transgender and Gender Nonconforming (TGNC) Identification
in EHRs") does not match any concept name exactly, so the exact-name path returns
nothing. The agentic path solves this with a three-stage pipeline:

```
Agent 1 (LLM)       raw text → 3-5 candidate phrases at different abstraction levels
SapBERT + FAISS     each candidate → nearest concepts in the SELECTED target
Agent 2 (LLM)       picks the best (or returns -1 for "none fit")
```

Ported from `backend/app/extraction/snomed_normalizer.py` in the CP Library, with
the target as a runtime argument instead of hardcoded SNOMED. Alias tables stay
as an override — a signed-off decision still beats a model decision — so the
composed path is `alias → agentic → exact-name fallback`.

### Supported targets

Each target is a `TargetProfile` + a FAISS index on disk:

```python
from concept_normalizer import profile_by_name, agentic_normalize, Retriever
from pathlib import Path

profile = profile_by_name("snomed")        # or "loinc", "omop", "bso_ad"
retriever = Retriever(Path(profile.index_dir))
result = agentic_normalize(
    "Transgender and Gender Nonconforming (TGNC) Identification in EHRs",
    profile, retriever=retriever,
)
# result.code, result.name, result.candidate_used, result.all_candidates
```

### Building a target's index

```bash
python scripts/build_index.py --target snomed \
    --from /path/to/concepts.csv --out indexes/snomed
```

Input CSV needs at minimum `code` and `name` columns; everything else is kept as
metadata on each hit. The build is one-time per vocabulary release; a built
index is portable across machines.

### Dependencies

The agentic path needs:

- `langchain_openai` + an OpenRouter (or OpenAI-compatible) key for the LLM
- `transformers` + `torch` for SapBERT
- `faiss-cpu` (or `faiss-gpu`) for the index
- `rapidfuzz` for the syntactic re-rank

The alias-table and exact-name paths work without any of these — the module stays
importable so a project that doesn't need the agentic path can skip the heavy
install.
