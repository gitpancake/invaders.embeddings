# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Space Invaders Flash Identifier - A Python service for identifying Space Invader street art flashes from images using CLIP embeddings and FAISS similarity search.

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  Reference Images   │     │   CLIP Encoder      │     │   FAISS Index       │
│  (3,882 flashes)    │ ──► │   (ViT-L/14, 768d)  │ ──► │   (IndexFlatIP)     │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
                                                                  │
┌─────────────────────┐     ┌─────────────────────┐               │
│  Query Image        │ ──► │   FastAPI Service   │ ◄─────────────┘
│  (PNG/AVIF/JPEG)    │     │   :8000/identify    │
└─────────────────────┘     └─────────────────────┘
```

## Project Structure

```
invaders.embeddings/
├── src/
│   ├── encoder/
│   │   └── clip.py              # CLIP model wrapper (uses mps on M1)
│   ├── index/
│   │   └── faiss_manager.py     # FAISS index build/search/persistence
│   ├── api/
│   │   ├── main.py              # FastAPI application
│   │   └── models.py            # Pydantic response models
│   └── scripts/
│       ├── build_index.py       # One-time: build reference index
│       ├── test_identify.py     # Test identification accuracy
│       └── process_batch.py     # Batch process IPFS images
├── data/
│   ├── flash_index.index        # FAISS index (generated)
│   └── flash_index.meta.json    # Flash metadata (generated)
├── requirements.txt
└── README.md
```

## Common Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Build reference index (one-time, ~5-10 min)
python -m src.scripts.build_index

# Run API server
KMP_DUPLICATE_LIB_OK=TRUE uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Test identification
KMP_DUPLICATE_LIB_OK=TRUE python -m src.scripts.test_identify

# Batch process IPFS images
KMP_DUPLICATE_LIB_OK=TRUE python -m src.scripts.process_batch \
  --input ipfs_hashes.csv \
  --output results.csv \
  --checkpoint checkpoint.json
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/identify` | POST | Identify flash from uploaded image |
| `/health` | GET | Health check (index size, model loaded) |
| `/` | GET | API info |

## Key Technical Details

- **CLIP Model**: `openai/clip-vit-large-patch14` (768-dim embeddings)
- **Device**: Automatically uses `mps` (Metal) on M1 Mac, `cuda` on NVIDIA, or `cpu`
- **Index Type**: `IndexFlatIP` (exact inner product search)
- **Image Formats**: PNG, JPEG, AVIF (via pillow-avif-plugin)

## Environment Variables

- `KMP_DUPLICATE_LIB_OK=TRUE` - Required to avoid OpenMP conflicts on macOS

## Performance

On M1 MacBook Pro (16GB RAM):
- Index build: ~5-10 minutes (3,882 images)
- Single identification: ~100-200ms
- FAISS search: <5ms

## Data Sources

Reference flash data comes from:
- `/Users/henrypye/Documents/code/invaders/flashcastr/public/data/json/invaders.json`
- Contains 3,882 flash entries with ID, name, location, and image URL
