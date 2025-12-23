core/
├── kb/
│   ├── builder.py        # orchestrates KB creation
│   ├── schema.py         # KBEntry dataclass
│   ├── text_processor.py # text cleaning + anatomy extraction
│   ├── image_loader.py   # safe image loading
│   ├── index.py          # FAISS index wrapper
│   └── storage.py        # save/load metadata & vectors
