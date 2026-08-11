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
from .ontology import load as load_ontology
from .registry import (
    CUSTOM_ID_BASE,
    DEFAULT_DOMAIN,
    RegistrationReport,
    SourceConcept,
    VocabularyRegistrar,
    stable_concept_id,
)
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
    "CUSTOM_ID_BASE",
    "Candidate",
    "Concept",
    "DEFAULT_DOMAIN",
    "ListVocabulary",
    "Normalization",
    "OmopVocabulary",
    "RegistrationReport",
    "SourceConcept",
    "Status",
    "VocabularyRegistrar",
    "loinc",
    "load_ontology",
    "normalize",
    "normalize_all",
    "normalize_text",
    "snomed",
    "stable_concept_id",
]
