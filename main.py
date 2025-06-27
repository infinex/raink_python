"""
Main FastAPI application for raink - LLM-based document ranking service.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from loguru import logger
import sys

from app.api import router
from app.models import ErrorResponse
from app.core.exceptions import RainkException


# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting raink FastAPI application")
    yield
    logger.info("Shutting down raink FastAPI application")


# Create FastAPI application
app = FastAPI(
    title="Raink API",
    description="""
    **Raink** is a powerful document ranking service that uses Large Language Models (LLMs) 
    to rank documents according to custom criteria.

    ## Features

    🎯 **Tournament-Style Ranking**: Uses a sophisticated algorithm that addresses common LLM issues:
    - **Non-determinism**: Multiple runs with score aggregation
    - **Context window limits**: Intelligent batch processing  
    - **Output constraints**: Structured response validation
    - **Subjective scoring**: Position-based ranking system

    🔧 **Flexible Configuration**: 
    - Multiple LLM providers (OpenAI, OpenRouter)
    - Configurable batch sizes and token limits
    - Recursive refinement with adjustable ratios
    - Dry run mode for testing

    📊 **Robust Processing**:
    - Automatic batch size optimization
    - Token estimation and validation
    - Comprehensive error handling
    - Detailed result metadata

    ## Quick Start

    1. **Basic Ranking**: Use `/rank` to rank a list of text objects
    2. **Batch Processing**: Use `/rank/batch` for multiple ranking tasks
    3. **Testing**: Use `/rank/dry-run` to test without API calls
    4. **Utilities**: Use `/estimate-tokens` and `/models` for information

    ## Authentication

    - **OpenAI**: Set `OPENAI_API_KEY` environment variable or pass in config
    - **OpenRouter**: Set `OPENROUTER_API_KEY` environment variable or pass in config

    ## Rate Limits

    Rate limits depend on your LLM provider:
    - **OpenAI**: Subject to your API rate limits
    - **OpenRouter**: Subject to your API rate limits and credit balance

    Built with ❤️ for document ranking challenges.
    """,
    version="1.0.0",
    contact={
        "name": "Raink API Support",
        "url": "https://github.com/noperator/raink",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom exception handler for raink exceptions
@app.exception_handler(RainkException)
async def raink_exception_handler(request: Request, exc: RainkException):
    """Handle custom raink exceptions."""
    logger.error(f"Raink exception: {exc.message} - Details: {exc.details}")
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.message,
            details=exc.details
        ).model_dump()
    )


# Generic exception handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred"
        ).model_dump()
    )


# Include API routes
app.include_router(router, prefix="/api/v1")


# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Raink API",
        version="1.0.0",
        description=app.description,
        routes=app.routes,
    )
    
    # Add example servers
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Local development server"},
        {"url": "https://api.raink.example.com", "description": "Production server"},
    ]
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Welcome endpoint with API information."""
    return {
        "message": "Welcome to Raink API",
        "description": "LLM-based document ranking service",
        "version": "1.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_url": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )