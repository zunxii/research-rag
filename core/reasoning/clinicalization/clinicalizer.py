# core/reasoning/clinicalization/clinicalizer.py
from .entity_extractor import ClinicalEntityExtractor
from .llm_abstraction import ClinicalAbstractionLLM
from .schemas import ClinicalQuery


class Clinicalizer:
    def __init__(self, gemini_api_key: str):
        self.entity_extractor = ClinicalEntityExtractor()
        self.abstraction_llm = ClinicalAbstractionLLM(gemini_api_key)

    def clinicalize(self, text: str) -> ClinicalQuery:
        entities = self.entity_extractor.extract(text)
        abstraction = self.abstraction_llm.abstract(text, entities)

        return ClinicalQuery(
            raw_text=text,
            entities=entities,
            abstraction=abstraction,
        )
