import json
from google import genai
from .prompt import PROMPT
from .schemas import Explanation


class GeminiCounterfactualExplainer:
    """
    Layer 3: Explanation only.
    NEVER touches embeddings or retrieval.
    Pure interpretation of structured signals.
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Gemini API key not provided")

        self.client = genai.Client(api_key=api_key)

    def explain(self, structured_input: dict) -> Explanation:
        prompt = PROMPT.format(
            input_json=json.dumps(structured_input, indent=2)
        )

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        text = response.text.strip()

        try:
            data = json.loads(text)
        except Exception as e:
            raise RuntimeError(
                f"Gemini returned invalid JSON:\n{text}"
            ) from e

        return Explanation(**data)
