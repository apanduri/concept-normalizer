"""concept-normalizer — extracted free text + a target vocabulary -> a concept.

    from pathlib import Path
    from concept_normalizer import OmopVocabulary, normalize

    target = OmopVocabulary(Path("concept.db"))
    result = normalize("Mini-mental state examination", target)
    result.concept.concept_id      # 4169175
    result.status                   # Status.MAPPED

The target is always an argument, never an assumption.  Nothing here knows about
the OMOP CDM, patients, or data insertion — a project with no database at all can
use this for normalization alone.
"""

from .normalize import Normalization, Status, normalize, normalize_all
from .agentic import AgenticResult, TargetProfile, generate_candidates, select_best
from .agentic import normalize as agentic_normalize
from .agentic_normalize import normalize as composed_normalize
from .profiles import (
    PROFILES,
    bso_ad_profile,
    by_name as profile_by_name,
    loinc_profile,
    omop_profile,
    snomed_profile,
)
from .tree_ontology import (
    ConceptNode,
    TreeOntology,
    TreeResult,
    TreeTargetProfile,
    load_tree,
    normalize_tree,
)
from .retriever import (
    RetrievedConcept,
    Retriever,
    get_retriever,
    resolve_index_dir,
)
from .aliases import AliasTable
from .target import (
    Candidate,
    Concept,
    ListVocabulary,
    OmopVocabulary,
    loinc,
    normalize_text,
    snomed,
)

__all__ = [
    "AliasTable",
    "AgenticResult",
    "Candidate",
    "Concept",
    "ListVocabulary",
    "Normalization",
    "OmopVocabulary",
    "Status",
    "loinc",
    "normalize",
    "normalize_all",
    "normalize_text",
    "snomed",
    "PROFILES",
    "ConceptNode",
    "RetrievedConcept",
    "Retriever",
    "TargetProfile",
    "agentic_normalize",
    "bso_ad_profile",
    "composed_normalize",
    "generate_candidates",
    "get_retriever",
    "loinc_profile",
    "omop_profile",
    "profile_by_name",
    "resolve_index_dir",
    "select_best",
    "TreeOntology",
    "TreeResult",
    "TreeTargetProfile",
    "load_tree",
    "normalize_tree",
    "snomed_profile",
]
