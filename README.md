# Space Invaders Flash Identifier

A Python service for identifying Space Invader street art flashes from images using CLIP embeddings and FAISS similarity search.

## Overview

This service:
1. Uses CLIP (ViT-L/14) to generate 768-dimensional embeddings from images
2. Stores reference flash embeddings in a FAISS index for fast similarity search
3. Provides a FastAPI endpoint to identify flashes from query images

## Setup

### Prerequisites

- Python 3.10+
- M1/M2 Mac (uses Metal), or NVIDIA GPU, or CPU

### Installation

```bash
cd invaders.embeddings

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Build the Reference Index

This downloads all ~3,900 reference flash images and builds the FAISS index:

```bash
python -m src.scripts.build_index
```

This takes ~5-10 minutes on M1 Mac and creates:
- `data/flash_index.index` - FAISS index file
- `data/flash_index.meta.json` - Flash metadata

### Run the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000

## API Endpoints

### POST /identify

Identify a flash from an uploaded image.

```bash
curl -X POST "http://localhost:8000/identify" \
  -F "file=@path/to/image.jpg" \
  -F "top_k=5"
```

Response:
```json
{
  "matches": [
    {
      "flash_id": 1234,
      "flash_name": "PA_567",
      "similarity": 0.92,
      "confidence": 0.92,
      "location": {"lat": 48.8566, "lng": 2.3522},
      "image_url": "https://..."
    }
  ],
  "processing_time_ms": 150.5
}
```

### GET /health

Check service health.

```bash
curl http://localhost:8000/health
```

## Performance

On M1 MacBook Pro:
- Index build: ~5-10 minutes (one-time)
- Single image identification: ~150-200ms
- FAISS search: <1ms

## Project Structure

```
invaders.embeddings/
├── src/
│   ├── encoder/
│   │   └── clip.py          # CLIP model wrapper
│   ├── index/
│   │   └── faiss_manager.py # FAISS index management
│   ├── api/
│   │   ├── main.py          # FastAPI application
│   │   └── models.py        # Pydantic models
│   └── scripts/
│       └── build_index.py   # Build reference index
├── data/
│   ├── flash_index.index    # FAISS index (generated)
│   └── flash_index.meta.json # Metadata (generated)
├── requirements.txt
└── README.md
```

## References

- [CLIP by OpenAI](https://github.com/openai/CLIP)
- [FAISS by Meta](https://github.com/facebookresearch/faiss)
- [Space Invaders by Invader](https://www.space-invaders.com/)
