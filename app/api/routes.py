"""
FastAPI routes for the raink application.
"""

import time
import os
import json
from typing import List, AsyncGenerator, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse
import tiktoken
from loguru import logger

from ..models import (
    RankingRequest, RankingResponse, RankingConfig,
    BatchRankingRequest, BatchRankingResponse,
    HealthResponse, ErrorResponse,
    EstimateTokensRequest, EstimateTokensResponse,
    ModelsResponse, ModelInfo, ModelProvider, OpenAIModel,
    StatusEvent, ProgressEvent, CompletionEvent, ErrorEvent,
    RankedResult
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
    
    # Set OpenRouter API key from environment if not provided
    if not config.openrouter_api_key and config.provider == ModelProvider.OPENROUTER:
        config.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not config.openrouter_api_key:
            raise ConfigurationError("OpenRouter API key not provided and OPENROUTER_API_KEY not set")
    
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
            name="openrouter-custom",
            provider=ModelProvider.OPENROUTER,
            max_tokens=128000,
            description="OpenRouter model (specify model name in config)"
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


def _should_stream(request_obj: Request) -> bool:
    """Determine if the request should be streamed based on Accept header or query parameter."""
    try:
        # Check Accept header for NDJSON
        accept_header = request_obj.headers.get("accept", "").lower()
        if "application/x-ndjson" in accept_header or "text/event-stream" in accept_header:
            return True
        
        # Check query parameter
        stream_param = request_obj.query_params.get("stream", "").lower()
        return stream_param in ("true", "1", "yes")
    except Exception as e:
        logger.warning(f"Error checking streaming preference: {e}")
        return False


def _validate_streaming_request(request: RankingRequest) -> None:
    """Validate request parameters for streaming compatibility."""
    if not request.objects:
        raise ValidationError("No objects provided for ranking")
    
    if len(request.objects) > 1000:  # Reasonable limit for streaming
        raise ValidationError("Too many objects for streaming (max 1000)")
    
    if not request.prompt or len(request.prompt.strip()) == 0:
        raise ValidationError("Empty prompt provided")
    
    if len(request.prompt) > 10000:  # Reasonable prompt length limit
        raise ValidationError("Prompt too long (max 10000 characters)")
    
    # Validate config for streaming
    if request.config.num_runs > 50:  # Reasonable limit for streaming
        raise ValidationError("Too many runs for streaming (max 50)")
    
    if request.config.batch_size > 100:  # Reasonable batch size limit
        raise ValidationError("Batch size too large for streaming (max 100)")


async def _stream_ranking_progress(
    request: RankingRequest,
    config: RankingConfig,
    start_time: float,
    http_request: Request = None
) -> AsyncGenerator[str, None]:
    """Generator function to stream ranking progress as NDJSON."""
    ranking_task = None
    event_queue = None
    client_id = f"{http_request.client.host}:{http_request.client.port}" if http_request and http_request.client else "unknown"
    events_sent = 0
    bytes_sent = 0
    
    logger.info(f"Streaming generator started for client {client_id}")
    
    try:
        # Send initial status event
        try:
            status_event = StatusEvent(
                status="starting",
                message=f"Starting {config.num_runs} ranking runs for {len(request.objects)} objects."
            )
            event_json = status_event.model_dump_json() + "\n"
            events_sent += 1
            bytes_sent += len(event_json.encode('utf-8'))
            logger.debug(f"Sent status event to {client_id} (event #{events_sent})")
            yield event_json
        except Exception as e:
            logger.error(f"Failed to send initial status event: {e}")
            # Send error and terminate gracefully
            error_event = ErrorEvent(message="Failed to initialize streaming response")
            yield error_event.model_dump_json() + "\n"
            return
        
        # Create ranking engine
        try:
            engine = RankingEngine(config)
            processing_objects = engine.prepare_objects(request.objects)
        except Exception as e:
            logger.error(f"Failed to initialize ranking engine: {e}")
            error_event = ErrorEvent(message=f"Initialization failed: {str(e)}")
            yield error_event.model_dump_json() + "\n"
            return
        
        # Create a queue for events
        import asyncio
        event_queue = asyncio.Queue()
        
        async def progress_callback(event_data: dict):
            """Callback to queue progress events for streaming."""
            try:
                await event_queue.put(event_data)
            except Exception as e:
                logger.error(f"Failed to queue progress event: {e}")
        
        # Start ranking task
        try:
            ranking_task = asyncio.create_task(
                engine.rank(processing_objects, request.prompt, progress_callback)
            )
        except Exception as e:
            logger.error(f"Failed to start ranking task: {e}")
            error_event = ErrorEvent(message=f"Failed to start ranking: {str(e)}")
            yield error_event.model_dump_json() + "\n"
            return
        
        # Stream events as they come
        while True:
            try:
                # Check for client disconnection (enhanced check)
                if http_request:
                    try:
                        is_disconnected = await http_request.is_disconnected()
                        if is_disconnected:
                            logger.info(
                                f"Client {client_id} disconnected, terminating stream gracefully "
                                f"(sent {events_sent} events, {bytes_sent} bytes)"
                            )
                            break
                    except Exception as e:
                        logger.warning(f"Could not check client connection status for {client_id}: {e}")
                        # Continue processing but be more aggressive about cleanup
                
                # Check if ranking task is done first
                if ranking_task.done():
                    try:
                        results = await ranking_task
                        
                        # Yield any remaining events
                        while not event_queue.empty():
                            try:
                                event_data = event_queue.get_nowait()
                                event_json = _format_streaming_event(event_data)
                                if event_json:
                                    events_sent += 1
                                    bytes_sent += len(event_json.encode('utf-8'))
                                    yield event_json + "\n"
                            except asyncio.QueueEmpty:
                                break
                            except Exception as e:
                                logger.error(f"Error processing remaining event: {e}")
                        
                        # Send completion event
                        try:
                            processing_time = time.time() - start_time
                            completion_event = CompletionEvent(
                                message=f"All {config.num_runs} runs finished. Final results available.",
                                final_results=results,
                                total_objects=len(request.objects),
                                config_used=config,
                                total_processing_time_seconds=round(processing_time, 2)
                            )
                            event_json = completion_event.model_dump_json() + "\n"
                            events_sent += 1
                            bytes_sent += len(event_json.encode('utf-8'))
                            logger.info(
                                f"Sent completion event to {client_id} "
                                f"(total events: {events_sent}, bytes: {bytes_sent})"
                            )
                            yield event_json
                        except Exception as e:
                            logger.error(f"Failed to send completion event: {e}")
                            error_event = ErrorEvent(message="Failed to complete streaming response")
                            yield error_event.model_dump_json() + "\n"
                        
                        break
                        
                    except Exception as e:
                        logger.error(f"Ranking task failed: {e}")
                        error_event = ErrorEvent(
                            message=f"Ranking task failed: {str(e)}"
                        )
                        yield error_event.model_dump_json() + "\n"
                        return
                
                # Try to get an event from the queue with timeout
                try:
                    event_data = await asyncio.wait_for(event_queue.get(), timeout=0.5)
                    event_json = _format_streaming_event(event_data)
                    if event_json:
                        events_sent += 1
                        bytes_sent += len(event_json.encode('utf-8'))
                        logger.debug(f"Sent progress event to {client_id} (event #{events_sent})")
                        yield event_json + "\n"
                    else:
                        logger.warning(f"Received invalid event data from queue: {event_data}")
                except asyncio.TimeoutError:
                    # No events available, continue loop to check ranking task status
                    continue
                except asyncio.CancelledError:
                    logger.info(f"Event queue operation cancelled for client {client_id}")
                    break
                except Exception as e:
                    logger.error(f"Error processing streaming event: {e}")
                    # Don't continue indefinitely on errors, break after too many failures
                    continue
                        
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                error_event = ErrorEvent(message=f"Streaming error: {str(e)}")
                yield error_event.model_dump_json() + "\n"
                return
                
    except Exception as e:
        # Send error event for any unhandled exceptions
        logger.error(f"Unexpected streaming error: {e}")
        try:
            error_event = ErrorEvent(
                message=f"An unexpected error occurred during processing: {str(e)}"
            )
            yield error_event.model_dump_json() + "\n"
        except Exception as nested_e:
            logger.error(f"Failed to send error event: {nested_e}")
    finally:
        # Cleanup: cancel ranking task if still running
        try:
            if ranking_task and not ranking_task.done():
                logger.info(f"Cleaning up ranking task for client {client_id}")
                ranking_task.cancel()
                try:
                    await asyncio.wait_for(ranking_task, timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Ranking task cleanup timed out for client {client_id}")
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error during ranking task cleanup for client {client_id}: {e}")
        except Exception as e:
            logger.error(f"Error during cleanup for client {client_id}: {e}")
        
        # Clear the event queue
        try:
            if event_queue:
                while not event_queue.empty():
                    try:
                        event_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
        except Exception as e:
            logger.error(f"Error clearing event queue for client {client_id}: {e}")
        
        # Final logging summary
        try:
            session_duration = time.time() - start_time
            logger.info(
                f"Streaming session completed for {client_id}: "
                f"duration={session_duration:.2f}s, events={events_sent}, bytes={bytes_sent}"
            )
        except Exception as e:
            logger.error(f"Error logging session summary for client {client_id}: {e}")


def _format_streaming_event(event_data: dict) -> Optional[str]:
    """Format event data into streaming JSON response."""
    try:
        if event_data["event_type"] == "progress":
            # Convert intermediate results to proper format
            intermediate_results = []
            for result in event_data.get("intermediate_results", []):
                try:
                    intermediate_results.append(RankedResult(
                        key=result['key'],
                        value=result['value'],
                        metadata=result.get('metadata'),
                        score=result['score'],
                        exposure=result['exposure'],
                        rank=result['rank']
                    ))
                except Exception as e:
                    logger.error(f"Error formatting intermediate result: {e}")
                    continue
            
            event = ProgressEvent(
                run_number=event_data["run_number"],
                message=event_data["message"],
                intermediate_results=intermediate_results,
                processing_time_current_run_ms=event_data["processing_time_current_run_ms"]
            )
            return event.model_dump_json()
            
        elif event_data["event_type"] == "error":
            # Handle error events
            error_event = ErrorEvent(
                message=event_data["message"],
                run_number=event_data.get("run_number")
            )
            return error_event.model_dump_json()
        
        return None
        
    except Exception as e:
        logger.error(f"Error formatting streaming event: {e}")
        return None


@router.post("/rank", tags=["Ranking"])
async def rank_objects(request: RankingRequest, http_request: Request):
    """
    Rank a list of objects according to the given prompt using LLM-based ranking.
    
    This endpoint implements a tournament-style ranking algorithm that:
    1. Splits objects into batches for processing
    2. Runs multiple randomized ranking rounds
    3. Aggregates scores across all runs
    4. Optionally applies recursive refinement to top results
    
    **Streaming Support:**
    - To get streaming progress updates, include `Accept: application/x-ndjson` header or add `?stream=true` parameter
    - Streaming responses use NDJSON format with real-time progress events
    - Each event has an `event_type` field: "status", "progress", "completion", or "error"
    
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
    
    **Example JSON Response:**
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
    
    **Example Streaming Response (NDJSON):**
    ```
    {"event_type": "status", "status": "starting", "message": "Starting 5 ranking runs for 3 objects."}
    {"event_type": "progress", "run_number": 1, "message": "Run 1/5 completed.", "intermediate_results": [...], "processing_time_current_run_ms": 650}
    {"event_type": "completion", "status": "completed", "message": "All 5 runs finished.", "final_results": [...], "total_objects": 3, "total_processing_time_seconds": 2.31}
    ```
    """
    start_time = time.time()
    
    try:
        # Validate configuration
        config = validate_config(request.config)
        
        # Check if streaming is requested
        if _should_stream(http_request):
            # Validate request for streaming compatibility
            _validate_streaming_request(request)
            
            # Log streaming request details
            client_ip = http_request.client.host if http_request.client else "unknown"
            user_agent = http_request.headers.get("user-agent", "unknown")
            logger.info(
                f"Starting streaming ranking request: "
                f"client={client_ip}, objects={len(request.objects)}, "
                f"runs={config.num_runs}, batch_size={config.batch_size}, "
                f"user_agent={user_agent[:100]}..."
            )
            
            # Return streaming response with enhanced headers
            return StreamingResponse(
                _stream_ranking_progress(request, config, start_time, http_request),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable nginx buffering for real-time streaming
                    "Transfer-Encoding": "chunked",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type, Accept",
                    "Access-Control-Allow-Methods": "POST, OPTIONS"
                }
            )
        
        # Regular JSON response
        # Create ranking engine
        engine = RankingEngine(config)
        
        # Prepare objects for processing
        processing_objects = engine.prepare_objects(request.objects)
        
        # Perform ranking without streaming
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