# core/reasoning/clinicalization/entity_extractor.py
import re
from pathlib import Path
import spacy
from scispacy.umls_linking import UmlsEntityLinker
from .schemas import EntityExtraction


class ClinicalEntityExtractor:
    def __init__(self, model="en_core_sci_md"):
        self.nlp = spacy.load(model)

        try:
            linker = UmlsEntityLinker(
                resolve_abbreviations=True,
                max_entities_per_mention=1,
                cache_dir=str(Path.home() / ".scispacy")
            )
            self.nlp.add_pipe(linker, last=True)
            self.linker = linker
        except Exception:
            self.linker = None

    _age_re = re.compile(r"(\d{1,2})\s*(?:-year-old|years old|yo|y/o)", re.I)
    _duration_re = re.compile(r"for\s+(\d+)\s*(days|weeks|months)", re.I)

    def extract(self, text: str) -> EntityExtraction:
        doc = self.nlp(text)

        age = None
        sex = None
        duration_days = None
        vitals = {}
        symptoms = []
        findings = []
        negated = []
        umls_cuis = []

        # Age
        m = self._age_re.search(text)
        if m:
            age = int(m.group(1))

        # Sex
        if re.search(r"\b(boy|male|man)\b", text, re.I):
            sex = "male"
        elif re.search(r"\b(girl|female|woman)\b", text, re.I):
            sex = "female"

        # Duration
        m = self._duration_re.search(text)
        if m:
            duration_days = int(m.group(1))

        # Entities
        for ent in doc.ents:
            mention = ent.text.strip()

            if ent.label_ in {"DISEASE_OR_SYNDROME", "SIGN_OR_SYMPTOM"}:
                symptoms.append(mention)
            elif ent.label_ == "ANATOMICAL_SITE":
                findings.append(mention)

            if self.linker and hasattr(ent._, "umls_ents"):
                for cui, _ in ent._.umls_ents:
                    umls_cuis.append(cui)

        return EntityExtraction(
            age=age,
            sex=sex,
            duration_days=duration_days,
            vitals=vitals,
            symptoms=sorted(set(symptoms)),
            findings=sorted(set(findings)),
            negated=negated,
            umls_cuis=sorted(set(umls_cuis)),
        )
