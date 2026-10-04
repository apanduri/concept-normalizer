"""Shipped TargetProfiles.

One per vocabulary. The profile is the only vocabulary-specific thing in the
agentic pipeline — prompts, retriever and orchestrator are all target-agnostic —
so adding a new target is adding one profile here and building its index.

A profile does NOT own the index; the index is a file on disk and is shared
across every process that reads it. The profile just points at it via
`index_dir`, which defaults to a path resolver so no profile needs to know
anybody's absolute paths.
"""

from __future__ import annotations

from .agentic import TargetProfile
from .retriever import resolve_index_dir


def snomed_profile() -> TargetProfile:
    """SNOMED CT — the target CP uses today.

    Prompts and examples are lifted from CP's `snomed_normalizer.py` with no
    semantic change, so a port against the same index should produce
    comparable output. That's the regression check that proves the
    generalization is behaviour-preserving.
    """
    return TargetProfile(
        name="SNOMED CT",
        style_hint="SNOMED CT-style concept label",
        purpose=(
            "- semantic grouping\n"
            "- ontology-based retrieval\n"
            "- hierarchical browsing"
        ),
        examples=[
            (
                "Transgender and Gender Nonconforming (TGNC) Identification in EHRs",
                ["Gender identity finding", "Gender identity", "Transgender identity"],
            ),
            (
                "Type 2 Diabetes Mellitus",
                ["Type 2 diabetes mellitus", "Diabetes mellitus", "Non-insulin dependent diabetes"],
            ),
            (
                "Atherosclerotic Cardiovascular Disease (ASCVD) - ADAPTABLE Trial Eligibility",
                ["Atherosclerotic cardiovascular disease", "Atherosclerosis", "Cardiovascular disease"],
            ),
            (
                "Phenotypic and genetic classification of diabetes",
                ["Diabetes mellitus", "Diabetes classification", "Endocrine disorder"],
            ),
        ],
        index_dir=str(resolve_index_dir("snomed")),
    )


def loinc_profile() -> TargetProfile:
    """LOINC — labs and clinical observations with units.

    LOINC's long common names are stylistically distinct from SNOMED's: they are
    structured as "Analyte/Property in Specimen", e.g. "Hemoglobin A1c/Hemoglobin.
    total in Blood". The examples steer the agent toward that shape rather than
    toward free prose.
    """
    return TargetProfile(
        name="LOINC",
        style_hint="LOINC long-common-name",
        purpose=(
            "- mapping a documented test or observation to its LOINC code\n"
            "- preserving analyte, property and specimen components"
        ),
        examples=[
            (
                "A1c result",
                ["Hemoglobin A1c/Hemoglobin.total in Blood", "Hemoglobin A1c", "Hemoglobin"],
            ),
            (
                "Montreal Cognitive Assessment total score",
                [
                    "Montreal cognitive assessment",
                    "Montreal cognitive assessment total score",
                    "Cognitive assessment",
                ],
            ),
            (
                "Smoking status",
                ["Tobacco smoking status", "Smoking status", "Tobacco use"],
            ),
        ],
        index_dir=str(resolve_index_dir("loinc")),
    )


def omop_profile() -> TargetProfile:
    """OMOP standard concepts — any domain.

    Wider than SNOMED or LOINC alone (it absorbs both plus RxNorm, HCPCS, etc.),
    so examples span a few domains to steer away from over-fitting to one.
    """
    return TargetProfile(
        name="OMOP standard concepts",
        style_hint="OMOP-standard concept name",
        purpose=(
            "- populating *_concept_id columns in an OMOP CDM\n"
            "- making the data queryable via standard concept sets with descendants"
        ),
        examples=[
            ("Type 2 diabetes", ["Type 2 diabetes mellitus", "Diabetes mellitus"]),
            (
                "Mini-Mental State Examination score",
                [
                    "Mini-mental state examination",
                    "Cognitive assessment",
                ],
            ),
            ("Donepezil", ["Donepezil", "Cholinesterase inhibitor"]),
        ],
        index_dir=str(resolve_index_dir("omop")),
    )


def bso_ad_profile() -> TargetProfile:
    """BSO-AD — the social and behavioural ontology Xuguang's SDK maps to.

    The style is unlike the clinical vocabularies above: labels are compact,
    PascalCase-ish ("Food_Insecurity"), and the ontology is deliberately scoped
    to SDOH and dementia. Examples reflect that.
    """
    return TargetProfile(
        name="BSO-AD",
        style_hint="BSO-AD concept label (compact, SDOH-focused)",
        purpose=(
            "- normalizing text to the BSO-AD ontology for annotation and\n"
            "- downstream BSO-AD-aware analyses"
        ),
        examples=[
            ("patient reports not having enough food", ["Food_Insecurity", "Food"]),
            ("walks for exercise daily", ["Exercise", "Walking", "Physical_Activity"]),
            ("lives alone", ["Living_Alone", "Living_Situation", "Social_Context"]),
        ],
        index_dir=str(resolve_index_dir("bso_ad")),
    )


PROFILES = {
    "snomed": snomed_profile,
    "loinc": loinc_profile,
    "omop": omop_profile,
    "bso_ad": bso_ad_profile,
}


def by_name(name: str) -> TargetProfile:
    """Look up a shipped profile by short name. Raises on unknown names rather
    than returning a default; a wrong default would be worse than an error."""
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in PROFILES:
        raise KeyError(
            f"unknown target profile {name!r}. Known: {sorted(PROFILES)}. "
            f"Build a TargetProfile directly for a custom target."
        )
    return PROFILES[key]()
