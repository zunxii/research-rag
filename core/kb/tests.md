## TEST 0 — Environment sanity

```bash
python - <<EOF
import torch, numpy, faiss, spacy
print("Torch:", torch.__version__)
print("NumPy:", numpy.__version__)
print("FAISS:", faiss.__version__)
nlp = spacy.load("en_core_sci_md")
print("SciSpacy OK")
EOF
```

## TEST 1 — Dry-run on 3 CSV rows

```bash
    

```