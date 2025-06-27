"""
Example requests for the Raink FastAPI application.
These examples demonstrate how to use the API endpoints.
"""

import asyncio
import httpx
import json
from typing import Dict, Any


# API Base URL
BASE_URL = "http://localhost:8000/api/v1"


async def example_basic_ranking():
    """Example: Basic ranking with OpenAI."""
    print("🎯 Example: Basic Ranking")
    print("=" * 50)
    
    request_data = {
        "prompt": "Rank these items according to their relevance to machine learning and artificial intelligence",
        "objects": [
            {
                "value": "Neural networks are computational models inspired by biological neural networks",
                "metadata": {"category": "AI/ML", "source": "wikipedia"}
            },
            {
                "value": "Pizza is a delicious Italian dish made with dough, tomato sauce, and cheese",
                "metadata": {"category": "food", "source": "cookbook"}
            },
            {
                "value": "Deep learning uses artificial neural networks with multiple layers",
                "metadata": {"category": "AI/ML", "source": "textbook"}
            },
            {
                "value": "The weather is sunny today with a temperature of 75 degrees",
                "metadata": {"category": "weather", "source": "news"}
            },
            {
                "value": "Machine learning algorithms can learn patterns from data without explicit programming",
                "metadata": {"category": "AI/ML", "source": "research"}
            }
        ],
        "config": {
            "batch_size": 5,
            "num_runs": 3,
            "provider": "openai",
            "openai_model": "gpt-4o-mini",
            "refinement_ratio": 0.6
        }
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{BASE_URL}/rank",
                json=request_data,
                timeout=60.0
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"✅ Ranked {result['total_objects']} objects in {result['processing_time_seconds']}s")
            print("\n📊 Results:")
            
            for i, item in enumerate(result['results'][:3], 1):
                print(f"{i}. Score: {item['score']:.2f} | Exposure: {item['exposure']} | {item['value'][:80]}...")
                
        except httpx.HTTPError as e:
            print(f"❌ Request failed: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")


async def example_ollama_ranking():
    """Example: Ranking with local Ollama model."""
    print("\n🦙 Example: Ollama Local Ranking")
    print("=" * 50)
    
    request_data = {
        "prompt": "Rank these programming concepts by their importance for beginners",
        "objects": [
            {"value": "Variables and data types"},
            {"value": "Advanced design patterns"},
            {"value": "Control structures (if/else, loops)"},
            {"value": "Distributed systems architecture"},
            {"value": "Functions and methods"},
            {"value": "Quantum computing algorithms"}
        ],
        "config": {
            "batch_size": 3,
            "num_runs": 2,
            "provider": "ollama",
            "ollama_model": "llama2",  # Change to your available model
            "token_limit": 4096
        }
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{BASE_URL}/rank",
                json=request_data,
                timeout=120.0  # Ollama might be slower
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"✅ Ranked {result['total_objects']} objects in {result['processing_time_seconds']}s")
            print("\n📊 Results:")
            
            for i, item in enumerate(result['results'], 1):
                print(f"{i}. {item['value']} (Score: {item['score']:.2f})")
                
        except httpx.HTTPError as e:
            print(f"❌ Request failed: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")


async def example_batch_ranking():
    """Example: Batch processing multiple ranking tasks."""
    print("\n🔄 Example: Batch Ranking")
    print("=" * 50)
    
    request_data = {
        "tasks": [
            {
                "prompt": "Rank by complexity for beginner programmers",
                "objects": [
                    {"value": "Hello World program"},
                    {"value": "Distributed database design"},
                    {"value": "Simple calculator"},
                    {"value": "Machine learning model"}
                ],
                "config": {"num_runs": 2, "batch_size": 4}
            },
            {
                "prompt": "Rank by relevance to web development",
                "objects": [
                    {"value": "HTML and CSS fundamentals"},
                    {"value": "Quantum physics equations"},
                    {"value": "JavaScript programming"},
                    {"value": "Cooking recipes"}
                ],
                "config": {"num_runs": 2, "batch_size": 4}
            }
        ]
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{BASE_URL}/rank/batch",
                json=request_data,
                timeout=120.0
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"✅ Processed {result['total_tasks']} tasks in {result['total_processing_time_seconds']}s")
            
            for i, task_result in enumerate(result['results'], 1):
                print(f"\n📋 Task {i} Results:")
                for j, item in enumerate(task_result['results'], 1):
                    print(f"  {j}. {item['value']} (Score: {item['score']:.2f})")
                    
        except httpx.HTTPError as e:
            print(f"❌ Request failed: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")


async def example_dry_run():
    """Example: Dry run for testing without API calls."""
    print("\n🧪 Example: Dry Run Testing")
    print("=" * 50)
    
    request_data = {
        "prompt": "Test ranking prompt - this won't be sent to LLM",
        "objects": [
            {"value": "First test item with some content"},
            {"value": "Second test item with different content"},
            {"value": "Third test item for validation"}
        ],
        "config": {
            "batch_size": 2,
            "num_runs": 3,
            "dry_run": True
        }
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{BASE_URL}/rank/dry-run",
                json=request_data,
                timeout=30.0
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"✅ Dry run completed in {result['processing_time_seconds']}s")
            print("📊 Simulated results:")
            
            for i, item in enumerate(result['results'], 1):
                print(f"{i}. {item['value']} (Simulated Score: {item['score']:.2f})")
                
        except httpx.HTTPError as e:
            print(f"❌ Request failed: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")


async def example_token_estimation():
    """Example: Estimate tokens for text."""
    print("\n🔢 Example: Token Estimation")
    print("=" * 50)
    
    request_data = {
        "text": "This is a sample text that we want to estimate tokens for. " * 10,
        "encoding": "o200k_base"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{BASE_URL}/estimate-tokens",
                json=request_data
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"✅ Text analysis:")
            print(f"   Characters: {result['character_count']}")
            print(f"   Tokens: {result['token_count']}")
            print(f"   Encoding: {result['encoding_used']}")
            print(f"   Ratio: {result['character_count'] / result['token_count']:.2f} chars/token")
                
        except httpx.HTTPError as e:
            print(f"❌ Request failed: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")


async def example_health_and_info():
    """Example: Health check and service information."""
    print("\n❤️ Example: Health Check & Service Info")
    print("=" * 50)
    
    async with httpx.AsyncClient() as client:
        try:
            # Health check
            health_response = await client.get(f"{BASE_URL}/health")
            health_response.raise_for_status()
            health = health_response.json()
            print(f"✅ Service Status: {health['status']}")
            print(f"   Version: {health['version']}")
            print(f"   Timestamp: {health['timestamp']}")
            
            # Models information
            models_response = await client.get(f"{BASE_URL}/models")
            models_response.raise_for_status()
            models = models_response.json()
            print(f"\n🤖 Available Models:")
            for model in models['models']:
                print(f"   • {model['name']} ({model['provider']}) - {model['max_tokens']} tokens")
                
        except httpx.HTTPError as e:
            print(f"❌ Request failed: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    """Run all examples."""
    print("🚀 Raink API Examples")
    print("=" * 50)
    print("Make sure the Raink API server is running on http://localhost:8000")
    print("For OpenAI examples, set your OPENAI_API_KEY environment variable")
    print("For Ollama examples, make sure Ollama is running locally")
    print()
    
    # Run examples
    await example_health_and_info()
    await example_token_estimation()
    await example_dry_run()
    
    # Uncomment these if you have API keys/services configured
    # await example_basic_ranking()
    # await example_ollama_ranking()  
    # await example_batch_ranking()
    
    print("\n✨ Examples completed!")
    print("Check the API documentation at http://localhost:8000/docs")


if __name__ == "__main__":
    asyncio.run(main())