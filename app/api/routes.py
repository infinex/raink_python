"""
FastAPI routes for the raink application.
"""

import time
import os
from typing import List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
import tiktoken
from loguru import logger

from ..models import (
    RankingRequest, RankingResponse, RankingConfig,
    BatchRankingRequest, BatchRankingResponse,
    HealthResponse, ErrorResponse,
    EstimateTokensRequest, EstimateTokensResponse,
    ModelsResponse, ModelInfo, ModelProvider, OpenAIModel
)
from ..core import (
    RankingEngine, ValidationError, ConfigurationError, 
    TokenLimitError, LLMAPIError, RankingError, TimeoutError
)

router = APIRouter()


def validate_config(config: RankingConfig) -> RankingConfig:
    """Validate and enrich ranking configuration."""
    # Set API key from environment if not provided
    if not config.openai_api_key and config.provider == ModelProvider.OPENAI:
        config.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not config.openai_api_key:
            raise ConfigurationError("OpenAI API key not provided and OPENAI_API_KEY not set")
    
    # Adjust token limits for Ollama
    if config.provider == ModelProvider.OLLAMA and config.token_limit == 128000:
        config.token_limit = 4096
        logger.info("Adjusted token limit to 4096 for Ollama")
    
    return config


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    )


@router.get("/models", response_model=ModelsResponse, tags=["Information"])
async def list_supported_models():
    """List all supported models and their capabilities."""
    models = [
        ModelInfo(
            name="gpt-4o",
            provider=ModelProvider.OPENAI,
            max_tokens=128000,
            description="Most capable OpenAI model"
        ),
        ModelInfo(
            name="gpt-4o-mini",
            provider=ModelProvider.OPENAI,
            max_tokens=128000,
            description="Smaller, faster OpenAI model"
        ),
        ModelInfo(
            name="gpt-4-turbo",
            provider=ModelProvider.OPENAI,
            max_tokens=128000,
            description="Previous generation GPT-4 model"
        ),
        ModelInfo(
            name="ollama-custom",
            provider=ModelProvider.OLLAMA,
            max_tokens=4096,
            description="Local Ollama model (specify model name in config)"
        )
    ]
    
    return ModelsResponse(models=models)


@router.post("/estimate-tokens", response_model=EstimateTokensResponse, tags=["Utilities"])
async def estimate_tokens(request: EstimateTokensRequest):
    """Estimate token count for given text."""
    try:
        encoding = tiktoken.get_encoding(request.encoding)
        tokens = len(encoding.encode(request.text))
        
        return EstimateTokensResponse(
            token_count=tokens,
            character_count=len(request.text),
            encoding_used=request.encoding
        )
    except Exception as e:
        logger.error(f"Token estimation error: {e}")
        raise HTTPException(status_code=400, detail=f"Token estimation failed: {str(e)}")


@router.post("/rank", response_model=RankingResponse, tags=["Ranking"])
async def rank_objects(request: RankingRequest):
    """
    Rank a list of objects according to the given prompt using LLM-based ranking.
    
    This endpoint implements a tournament-style ranking algorithm that:
    1. Splits objects into batches for processing
    2. Runs multiple randomized ranking rounds
    3. Aggregates scores across all runs
    4. Optionally applies recursive refinement to top results
    
    **Example Request:**
    ```json
    {
        "prompt": "Rank these items by their relevance to machine learning",
        "objects": [
            {"value": "Neural networks are computational models inspired by biological brains"},
            {"value": "Pizza is a popular Italian dish with cheese and tomato sauce"},
            {"value": "Deep learning uses multiple layers to learn complex patterns"}
        ],
        "config": {
            "batch_size": 10,
            "num_runs": 5,
            "provider": "openai",
            "openai_model": "gpt-4o-mini"
        }
    }
    ```
    
    **Example Response:**
    ```json
    {
        "results": [
            {
                "key": "abc123ef",
                "value": "Neural networks are computational models...",
                "score": 1.2,
                "exposure": 15,
                "rank": 1
            }
        ],
        "total_objects": 3,
        "processing_time_seconds": 12.4
    }
    ```
    """
    start_time = time.time()
    
    try:
        # Validate configuration
        config = validate_config(request.config)
        
        # Create ranking engine
        engine = RankingEngine(config)
        
        # Prepare objects for processing
        processing_objects = engine.prepare_objects(request.objects)
        
        # Perform ranking
        results = await engine.rank(processing_objects, request.prompt)
        
        processing_time = time.time() - start_time
        
        return RankingResponse(
            results=results,
            total_objects=len(request.objects),
            config_used=config,
            processing_time_seconds=round(processing_time, 2)
        )
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=e.message)
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(status_code=400, detail=e.message)
    except TokenLimitError as e:
        logger.error(f"Token limit error: {e}")
        raise HTTPException(status_code=413, detail=e.message)
    except LLMAPIError as e:
        logger.error(f"LLM API error: {e}")
        raise HTTPException(status_code=502, detail=f"LLM API error: {e.message}")
    except TimeoutError as e:
        logger.error(f"Timeout error: {e}")
        raise HTTPException(status_code=504, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/rank/batch", response_model=BatchRankingResponse, tags=["Ranking"])
async def rank_batch(request: BatchRankingRequest, background_tasks: BackgroundTasks):
    """
    Process multiple ranking tasks in batch.
    
    This endpoint allows you to submit multiple ranking tasks that will be processed
    concurrently to improve throughput.
    
    **Example Request:**
    ```json
    {
        "tasks": [
            {
                "prompt": "Rank by relevance to AI",
                "objects": [{"value": "Machine learning"}, {"value": "Cooking"}],
                "config": {"num_runs": 3}
            },
            {
                "prompt": "Rank by complexity",
                "objects": [{"value": "Hello world"}, {"value": "Quantum physics"}],
                "config": {"num_runs": 3}
            }
        ]
    }
    ```
    """
    start_time = time.time()
    
    try:
        # Process all tasks concurrently
        import asyncio
        
        async def process_task(task: RankingRequest) -> RankingResponse:
            # Validate configuration
            config = validate_config(task.config)
            
            # Create ranking engine
            engine = RankingEngine(config)
            
            # Prepare objects for processing
            processing_objects = engine.prepare_objects(task.objects)
            
            # Perform ranking
            task_start = time.time()
            results = await engine.rank(processing_objects, task.prompt)
            task_time = time.time() - task_start
            
            return RankingResponse(
                results=results,
                total_objects=len(task.objects),
                config_used=config,
                processing_time_seconds=round(task_time, 2)
            )
        
        # Run all tasks concurrently
        tasks = [process_task(task) for task in request.tasks]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        return BatchRankingResponse(
            results=results,
            total_tasks=len(request.tasks),
            total_processing_time_seconds=round(total_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@router.post("/rank/dry-run", response_model=RankingResponse, tags=["Testing"])
async def dry_run_ranking(request: RankingRequest):
    """
    Perform a dry run of the ranking process without making LLM API calls.
    
    This endpoint is useful for:
    - Testing your request format
    - Validating object sizes and token limits  
    - Understanding the processing pipeline
    - Estimating costs before running actual ranking
    
    **Example Request:**
    ```json
    {
        "prompt": "Test ranking prompt",
        "objects": [
            {"value": "First test item"},
            {"value": "Second test item"}
        ],
        "config": {
            "dry_run": true,
            "batch_size": 5
        }
    }
    ```
    """
    # Force dry run mode
    request.config.dry_run = True
    
    # Process as normal ranking request
    return await rank_objects(request)