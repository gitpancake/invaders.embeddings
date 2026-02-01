"""FastAPI application for flash identification."""

import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import IdentifyResponse, HealthResponse, FlashMatch
from ..encoder import CLIPEncoder
from ..index import FAISSIndexManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
encoder: CLIPEncoder = None
index_manager: FAISSIndexManager = None

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
INDEX_PATH = DATA_DIR / "flash_index"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and index on startup."""
    global encoder, index_manager
    
    logger.info("Loading CLIP encoder...")
    encoder = CLIPEncoder()
    
    logger.info(f"Loading FAISS index from {INDEX_PATH}...")
    index_manager = FAISSIndexManager()
    
    if INDEX_PATH.with_suffix(".index").exists():
        index_manager.load(str(INDEX_PATH))
        logger.info(f"Loaded index with {index_manager.size} flashes")
    else:
        logger.warning(f"Index not found at {INDEX_PATH}. Run build_index.py first.")
    
    yield
    
    logger.info("Shutting down...")


app = FastAPI(
    title="Space Invaders Flash Identifier",
    description="Identify Space Invader street art flashes from images using CLIP + FAISS",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health."""
    return HealthResponse(
        status="healthy" if index_manager and index_manager.size > 0 else "degraded",
        index_size=index_manager.size if index_manager else 0,
        model_loaded=encoder is not None,
    )


@app.post("/identify", response_model=IdentifyResponse)
async def identify_flash(
    file: UploadFile = File(..., description="Image file to identify"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of matches to return"),
    min_confidence: float = Query(default=0.0, ge=0, le=1, description="Minimum confidence threshold"),
):
    """
    Identify which Space Invader flash is in the uploaded image.
    
    Returns the top-k most similar flashes from the reference database.
    """
    if index_manager is None or index_manager.size == 0:
        raise HTTPException(status_code=503, detail="Index not loaded. Run build_index.py first.")
    
    start_time = time.time()
    
    # Read and encode image
    try:
        image_bytes = await file.read()
        embedding = encoder.encode_image_from_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")
    
    # Search
    matches = index_manager.search(embedding, top_k=top_k)
    
    # Filter by confidence
    if min_confidence > 0:
        matches = [m for m in matches if m["confidence"] >= min_confidence]
    
    processing_time_ms = (time.time() - start_time) * 1000
    
    return IdentifyResponse(
        matches=[FlashMatch(**m) for m in matches],
        processing_time_ms=processing_time_ms,
    )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Space Invaders Flash Identifier",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/identify": "Identify a flash from an image (POST with file upload)",
            "/docs": "OpenAPI documentation",
        }
    }
