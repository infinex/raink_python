# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

raink is a Go CLI tool that uses LLMs for document ranking. It addresses challenges with LLM non-determinism, context window limitations, and subjective scoring by implementing a tournament-style ranking algorithm across multiple runs and batches.

## Development Commands

### Build and Install
```bash
go install
```

### Run the Tool
```bash
raink -f <input_file> -p "Your ranking prompt"
```

### Example Usage
```bash
# Basic ranking with 10 runs, 10 items per batch
raink -f testdata/sentences.txt -r 10 -s 10 -p 'Rank each of these items according to their relevancy to the concept of "time".'

# With JSON output
raink -f input.json -template '{{ .fieldName }}' -p 'Rank by relevance' -o results.json

# Dry run mode (no API calls)
raink -dry-run -f input.txt -p "Test prompt"
```

### Testing
No formal test suite is present. Test the tool using:
- Sample data files in testdata/ directory (currently deleted but referenced in README)
- Dry run mode with `-dry-run` flag
- Small batch sizes for quick validation
- Use docker compose exec raink-api to go into the container to test

## Architecture

### Core Components

**Config struct** (`main.go:44-58`): Holds all user-configurable parameters including API settings, batch sizes, and model configurations.

**Ranker struct** (`main.go:83-89`): Main processing engine that manages the ranking algorithm, tokenization, and batch processing.

**Object struct** (`main.go:150-157`): Represents items to be ranked with ID, value string, and optional original structured data.

### Key Algorithms

**Tournament Ranking** (`main.go:406-474`): Recursive algorithm that ranks objects through multiple rounds, refining top portions based on refinement ratio.

**Shuffle-Batch-Rank** (`main.go:482-585`): Core ranking logic that shuffles objects across multiple runs, processes them in batches, and aggregates scores.

**Dynamic Batch Sizing** (`main.go:108-148`): Automatically adjusts batch size to stay within token limits while maximizing throughput.

### API Integrations

Supports both OpenAI and Ollama APIs with robust error handling, rate limiting, and retry logic:
- **OpenAI client** (`main.go:758-881`): Uses structured JSON schema responses with conversation history for error recovery
- **Ollama client** (`main.go:883-978`): HTTP-based integration with JSON format responses

### Input Processing

**File Loading** (`main.go:207-284`): Supports both text files (line-by-line) and JSON arrays with Go template support for flexible data extraction.

**Template System**: Uses Go's text/template package to extract specific fields from JSON objects or format text lines.

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for OpenAI API access

### Key Parameters
- `-r`: Number of runs (default 10) - affects ranking accuracy
- `-s`: Batch size (default 10) - auto-adjusted based on token limits
- `-t`: Token limit per batch (default 128000, 4096 for Ollama)
- `-ratio`: Refinement ratio (default 0.5) - controls recursive refinement

### Model Support
- OpenAI models via `-openai-model` (default: gpt-4o-mini)
- Ollama models via `-ollama-model` 
- Custom OpenAI-compatible APIs via `-openai-url`

## File Structure

- `main.go`: Single-file application containing all logic
- `go.mod`/`go.sum`: Go module dependencies
- `testdata/`: Sample data files (referenced but not present)
- `LICENSE`: MIT license

## Common Patterns

- All API calls include retry logic with exponential backoff
- ID validation ensures all input objects are ranked in responses
- Token estimation prevents context window overflow
- Conversation history enables error recovery with API providers
- Deterministic short IDs (8 chars) for efficient object tracking