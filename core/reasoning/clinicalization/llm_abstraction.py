import json
from google import genai
from google.genai import types

from .schemas import ClinicalAbstraction, EntityExtraction


class ClinicalAbstractionLLM:
    """
    Gemini-based clinical abstraction layer.

    Guarantees:
    - JSON-only output
    - No diagnosis
    - No hallucination
    - Deterministic (low temperature)
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def abstract(
        self,
        raw_text: str,
        entities: EntityExtraction
    ) -> ClinicalAbstraction:

        prompt = f"""
You are a clinical abstraction system.

TASK:
Convert the clinical query into a structured abstraction.
- Do NOT diagnose
- Do NOT invent facts
- Use ONLY the provided text and entities
- Be concise and clinical

RAW QUERY:
{raw_text}

EXTRACTED ENTITIES:
{entities}

OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "chief_complaints": [],
  "key_symptoms": [],
  "clinical_syndromes": [],
  "risk_factors": [],
  "red_flags": [],
  "normalized_terms": {{}}
}}

IMPORTANT:
return only json no markdown no explanation no comments, no ```json markdowns nothing just cold blooded json output
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=3000
            )
        )

        text = response.text.strip()

        # Defensive JSON parsing
        try:
            data = json.loads(text)
        except Exception:
            raise RuntimeError(
                "Gemini returned invalid JSON:\n\n" + text
            )

        return ClinicalAbstraction(**data)
