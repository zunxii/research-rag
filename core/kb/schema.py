# core/kb/schema.py
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class KBEntry:
    case_id: str
    image_path: str
    diagnosis_label: str

    clinical_text: Dict[str, str]
    anatomy: Dict[str, List[str]]

    embedding_id: int
