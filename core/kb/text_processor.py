# core/kb/text_processor.py

import re
from typing import Dict, List
import spacy
# from scispacy.umls_linking import UmlsEntityLinker


class TextProcessor:
    """
    Text processing for KB creation.
    
    Responsibilities:
    - Clean raw dataset text (mechanical only)
    - Extract explicitly mentioned anatomical entities
    - Normalize anatomy using UMLS concepts
    
    Guarantees:
    - No inference
    - No hallucination
    - No synonym expansion
    """

    def __init__(self):
        # Load SciSpacy model
        self.nlp = spacy.load("en_core_sci_md")

        # Add UMLS linker
        # self.linker = UmlsEntityLinker(
        #     resolve_abbreviations=True,
        #     max_entities_per_mention=1
        # )
        # self.nlp.add_pipe(self.linker, last=True)

    # -------------------------
    # Text Cleaning (Mechanical)
    # -------------------------
    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    # -------------------------
    # Combine Context + Description
    # -------------------------
    def combine_text(self, context: str, description: str) -> str:
        context_clean = self.clean_text(context)
        desc_clean = self.clean_text(description)

        if context_clean and desc_clean:
            return f"{context_clean} {desc_clean}"
        return context_clean or desc_clean

    # -------------------------
    # Anatomy Extraction (NER-based)
    # -------------------------
    def extract_anatomy(self, text: str) -> Dict[str, List[str]]:
        doc = self.nlp(text)

        mentions = []
        semantic_types = []

        for ent in doc.ents:
            if ent.label_ in {"ANATOMICAL_SITE", "BODY_PART", "ORGAN"}:
                mentions.append(ent.text)

                # coarse semantic mapping
                semantic_types.append(ent.label_)

        return {
            "raw_mentions": list(set(mentions)),
            "semantic_types": list(set(semantic_types))
        }
    # -------------------------
    # Optional: Coarse Region Mapping
    # -------------------------
    def normalize_region(self, semantic_types: List[str]) -> str:
        if not semantic_types:
            return "unknown"

        if "ANATOMICAL_SITE" in semantic_types:
            return "anatomical_site"
        if "BODY_PART" in semantic_types:
            return "body_part"
        if "ORGAN" in semantic_types:
            return "organ"

        return "unknown"
