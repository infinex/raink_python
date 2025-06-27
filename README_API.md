# Raink FastAPI Application

A powerful document ranking service that uses Large Language Models (LLMs) to rank documents according to custom criteria. This is a complete refactor of the original Go CLI tool into a modern FastAPI web service.

## 🌟 Features

- **Tournament-Style Ranking**: Sophisticated algorithm addressing common LLM challenges
- **Multiple LLM Providers**: Support for OpenAI and Ollama models
- **Batch Processing**: Handle multiple ranking tasks concurrently
- **Token Management**: Automatic batch size optimization and token estimation
- **RESTful API**: Clean, well-documented endpoints with OpenAPI/Swagger
- **Comprehensive Error Handling**: Detailed error responses and logging
- **Dry Run Mode**: Test your requests without making API calls

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/noperator/raink
cd raink

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key"  # For OpenAI models
```

### Running the Server

```bash
# Development mode with auto-reload
python main.py

# Or with uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API Base**: http://localhost:8000/api/v1
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📚 API Endpoints

### Core Ranking

#### `POST /api/v1/rank`
Rank a list of objects according to a custom prompt.

**Request:**
```json
{
  "prompt": "Rank these items by their relevance to machine learning",
  "objects": [
    {
      "value": "Neural networks are computational models inspired by biological brains",
      "metadata": {"category": "AI/ML"}
    },
    {
      "value": "Pizza is a popular Italian dish with cheese and tomato sauce",
      "metadata": {"category": "food"}
    }
  ],
  "config": {
    "batch_size": 10,
    "num_runs": 5,
    "provider": "openai",
    "openai_model": "gpt-4o-mini",
    "refinement_ratio": 0.5
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "key": "abc123ef",
      "value": "Neural networks are computational models...",
      "metadata": {"category": "AI/ML"},
      "score": 1.2,
      "exposure": 15,
      "rank": 1
    }
  ],
  "total_objects": 2,
  "config_used": {...},
  "processing_time_seconds": 12.4
}
```

#### `POST /api/v1/rank/batch`
Process multiple ranking tasks concurrently.

#### `POST /api/v1/rank/dry-run`
Test your ranking request without making LLM API calls.

### Utilities

#### `GET /api/v1/health`
Health check endpoint.

#### `GET /api/v1/models`
List supported models and their capabilities.

#### `POST /api/v1/estimate-tokens`
Estimate token count for given text.

## ⚙️ Configuration

### RankingConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 10 | Number of items per batch (2-50) |
| `num_runs` | int | 10 | Number of ranking runs (1-100) |
| `token_limit` | int | 128000 | Maximum tokens per batch |
| `refinement_ratio` | float | 0.5 | Refinement ratio for recursive ranking (0.0-1.0) |
| `provider` | enum | "openai" | LLM provider ("openai" or "ollama") |
| `openai_model` | enum | "gpt-4o-mini" | OpenAI model name |
| `ollama_model` | str | null | Ollama model name |
| `dry_run` | bool | false | Enable dry run mode |

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `RELOAD`: Enable auto-reload (default: false)

## 🔧 Advanced Usage

### Custom Templates
You can process structured data by providing templates in your objects:

```json
{
  "prompt": "Rank code quality",
  "objects": [
    {
      "value": "Function: calculate_sum\nComplexity: O(n)\nLines: 15",
      "metadata": {"file": "utils.py", "function": "calculate_sum"}
    }
  ]
}
```

### Batch Processing
Process multiple ranking tasks efficiently:

```json
{
  "tasks": [
    {
      "prompt": "Rank by complexity",
      "objects": [...],
      "config": {"num_runs": 3}
    },
    {
      "prompt": "Rank by importance", 
      "objects": [...],
      "config": {"num_runs": 5}
    }
  ]
}
```

### Using Ollama (Local Models)
```json
{
  "config": {
    "provider": "ollama",
    "ollama_model": "llama2",
    "ollama_url": "http://localhost:11434/api/chat",
    "token_limit": 4096
  }
}
```

## 🧪 Testing

### Run Examples
```bash
# Make sure the server is running, then:
python examples/example_requests.py
```

### Dry Run Testing
Use the `/rank/dry-run` endpoint to test your requests without API calls:

```bash
curl -X POST "http://localhost:8000/api/v1/rank/dry-run" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Test prompt",
    "objects": [{"value": "Test item"}],
    "config": {"dry_run": true}
  }'
```

## 🏗️ Architecture

### Project Structure
```
app/
├── __init__.py
├── models.py           # Pydantic models for request/response validation
├── api/
│   ├── __init__.py
│   └── routes.py       # FastAPI route handlers
└── core/
    ├── __init__.py
    ├── ranker.py       # Core ranking algorithm
    ├── llm_clients.py  # OpenAI and Ollama API clients
    └── exceptions.py   # Custom exception classes

main.py                 # FastAPI application entry point
examples/
└── example_requests.py # Usage examples
```

### Algorithm Overview

1. **Input Processing**: Convert objects to internal format with generated IDs
2. **Batch Optimization**: Automatically adjust batch sizes to fit token limits
3. **Tournament Ranking**: Multiple randomized runs with shuffle-batch-rank
4. **Score Aggregation**: Average scores across all runs for each object
5. **Recursive Refinement**: Optional recursive processing of top results
6. **Final Sorting**: Return ranked results with metadata

## 🔒 Error Handling

The API provides comprehensive error handling:

- **400 Bad Request**: Validation errors, configuration issues
- **413 Payload Too Large**: Token limit exceeded
- **502 Bad Gateway**: LLM API errors
- **504 Gateway Timeout**: Request timeout
- **500 Internal Server Error**: Unexpected errors

All errors include detailed messages and structured responses.

## 📈 Performance

- **Concurrent Processing**: Batch operations run concurrently
- **Smart Batching**: Automatic batch size optimization
- **Token Efficiency**: Precise token estimation and management
- **Caching**: Efficient object ID generation and validation

## 🤝 API Clients

### Python Client Example
```python
import httpx

async def rank_documents(prompt: str, documents: list):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/rank",
            json={
                "prompt": prompt,
                "objects": [{"value": doc} for doc in documents]
            }
        )
        return response.json()
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/api/v1/rank" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Rank by importance",
    "objects": [
      {"value": "First document"},
      {"value": "Second document"}
    ]
  }'
```

## 📖 Documentation

- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🐳 Docker Support

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original Go implementation inspiration
- FastAPI framework for excellent async support
- OpenAI and Ollama for LLM capabilities