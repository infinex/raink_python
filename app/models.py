"""
Pydantic models for request/response validation in the raink FastAPI application.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class ModelProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    OLLAMA = "ollama"


class OpenAIModel(str, Enum):
    """Supported OpenAI models."""
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_3_5_TURBO = "gpt-3.5-turbo"


class RankingObject(BaseModel):
    """An object to be ranked."""
    model_config = ConfigDict(extra="allow")
    
    id: Optional[str] = Field(None, description="Optional unique ID for the object")
    value: str = Field(..., description="The text content to be ranked")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the object")


class RankingConfig(BaseModel):
    """Configuration for the ranking process."""
    batch_size: int = Field(10, ge=2, le=50, description="Number of items per batch")
    num_runs: int = Field(10, ge=1, le=100, description="Number of ranking runs")
    token_limit: int = Field(128000, ge=1000, le=200000, description="Maximum tokens per batch")
    refinement_ratio: float = Field(0.5, ge=0.0, lt=1.0, description="Refinement ratio for recursive ranking")
    encoding: str = Field("o200k_base", description="Tokenizer encoding to use")
    
    # Model configuration
    provider: ModelProvider = Field(ModelProvider.OPENAI, description="LLM provider to use")
    openai_model: OpenAIModel = Field(OpenAIModel.GPT_4O_MINI, description="OpenAI model name")
    ollama_model: Optional[str] = Field(None, description="Ollama model name")
    
    # API configuration
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    openai_base_url: Optional[str] = Field(None, description="Custom OpenAI API base URL")
    ollama_url: str = Field("http://localhost:11434/api/chat", description="Ollama API URL")
    
    # Processing options
    template: Optional[str] = Field(None, description="Template for formatting objects")
    dry_run: bool = Field(False, description="Enable dry run mode")


class RankingRequest(BaseModel):
    """Request model for ranking objects."""
    prompt: str = Field(..., min_length=1, description="The ranking prompt/criteria")
    objects: List[RankingObject] = Field(..., min_items=1, description="List of objects to rank")
    config: RankingConfig = Field(default_factory=RankingConfig, description="Ranking configuration")


class RankedResult(BaseModel):
    """A single ranked result."""
    key: str = Field(..., description="Unique identifier for the object")
    value: str = Field(..., description="The original text content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Original metadata if provided")
    score: float = Field(..., description="Ranking score (lower is better)")
    exposure: int = Field(..., description="Number of times this object was compared")
    rank: int = Field(..., description="Final rank position (1-based)")


class RankingResponse(BaseModel):
    """Response model for ranking results."""
    results: List[RankedResult] = Field(..., description="Ranked results in order")
    total_objects: int = Field(..., description="Total number of objects ranked")
    config_used: RankingConfig = Field(..., description="Configuration used for ranking")
    processing_time_seconds: float = Field(..., description="Total processing time")


class BatchRankingRequest(BaseModel):
    """Request model for batch processing multiple ranking tasks."""
    tasks: List[RankingRequest] = Field(..., min_items=1, max_items=10, description="List of ranking tasks")


class BatchRankingResponse(BaseModel):
    """Response model for batch ranking results."""
    results: List[RankingResponse] = Field(..., description="Results for each ranking task")
    total_tasks: int = Field(..., description="Total number of tasks processed")
    total_processing_time_seconds: float = Field(..., description="Total processing time for all tasks")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class EstimateTokensRequest(BaseModel):
    """Request model for token estimation."""
    text: str = Field(..., description="Text to estimate tokens for")
    encoding: str = Field("o200k_base", description="Tokenizer encoding to use")


class EstimateTokensResponse(BaseModel):
    """Response model for token estimation."""
    token_count: int = Field(..., description="Estimated number of tokens")
    character_count: int = Field(..., description="Number of characters")
    encoding_used: str = Field(..., description="Tokenizer encoding used")


class ModelInfo(BaseModel):
    """Information about a supported model."""
    name: str = Field(..., description="Model name")
    provider: ModelProvider = Field(..., description="Model provider")
    max_tokens: int = Field(..., description="Maximum context window")
    description: Optional[str] = Field(None, description="Model description")


class ModelsResponse(BaseModel):
    """Response model for listing supported models."""
    models: List[ModelInfo] = Field(..., description="List of supported models")