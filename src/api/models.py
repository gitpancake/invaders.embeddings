"""Pydantic models for the API."""

from pydantic import BaseModel, Field
from typing import List, Optional


class FlashMatch(BaseModel):
    """A single flash match result."""
    flash_id: int = Field(..., description="Unique flash identifier")
    flash_name: str = Field(..., description="Flash name (e.g., 'PA_1234')")
    similarity: float = Field(..., ge=-1, le=1, description="Cosine similarity score")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    location: Optional[dict] = Field(None, description="Flash location (lat/lng)")
    image_url: Optional[str] = Field(None, description="Reference image URL")


class IdentifyResponse(BaseModel):
    """Response from the /identify endpoint."""
    matches: List[FlashMatch] = Field(..., description="Top matching flashes")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    index_size: int = Field(..., description="Number of flashes in index")
    model_loaded: bool = Field(..., description="Whether CLIP model is loaded")


class BatchIdentifyRequest(BaseModel):
    """Request for batch identification."""
    image_urls: List[str] = Field(..., description="List of image URLs to identify")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of matches per image")


class BatchJob(BaseModel):
    """Batch job status."""
    job_id: str
    status: str  # queued, processing, completed, failed
    total: int
    processed: int
    results_url: Optional[str] = None
