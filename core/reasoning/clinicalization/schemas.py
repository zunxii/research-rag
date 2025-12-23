# core/reasoning/clinicalization/schemas.py
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class EntityExtraction:
    age: Optional[int]
    sex: Optional[str]
    duration_days: Optional[int]
    vitals: Dict[str, float]
    symptoms: List[str]
    findings: List[str]
    negated: List[str]
    umls_cuis: List[str]


@dataclass
class ClinicalAbstraction:
    chief_complaints: List[str]
    key_symptoms: List[str]
    clinical_syndromes: List[str]
    risk_factors: List[str]
    red_flags: List[str]
    normalized_terms: Dict[str, str]


@dataclass
class ClinicalQuery:
    raw_text: str
    entities: EntityExtraction
    abstraction: ClinicalAbstraction
