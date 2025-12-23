CSV row
 ├── image_path
 ├── category
 ├── context
 └── description
        ↓
TextProcessor.combine_text()
        ↓
TextProcessor.extract_anatomy()
        ↓
ImageEncoder (BioMedCLIP)
        ↓
TextEncoder (PubMedBERT)
        ↓
FusionMLP
        ↓
FAISS index
        ↓
metadata.json
