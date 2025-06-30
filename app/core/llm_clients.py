"""
LLM client implementations for OpenAI and OpenRouter APIs.
"""

import json
import asyncio
import time
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from openai import AsyncOpenAI
from loguru import logger

from .ranker import ProcessingObject, RankedObject
from ..models import RankingConfig


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def rank_batch(
        self, 
        batch: List[ProcessingObject], 
        prompt: str,
        run_num: int,
        batch_num: int
    ) -> List[RankedObject]:
        """Rank a batch of objects using the LLM."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client for ranking."""
    
    def __init__(self, config: RankingConfig):
        self.config = config
        
        client_kwargs = {}
        if config.openai_api_key:
            client_kwargs['api_key'] = config.openai_api_key
        if config.openai_base_url:
            client_kwargs['base_url'] = config.openai_base_url.rstrip('/') + '/'
        
        self.client = AsyncOpenAI(**client_kwargs)
        self.schema = self._get_response_schema()
    
    def _get_response_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for structured responses."""
        return {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ranked object IDs"
                }
            },
            "required": ["objects"],
            "additionalProperties": False
        }
    
    def _build_prompt(self, batch: List[ProcessingObject], user_prompt: str) -> str:
        """Build the full prompt for ranking."""
        prompt = f"{user_prompt}\n\n"
        prompt += """
REMEMBER to:
- ALWAYS respond with the short 8-character ID of each item found above the value
- NEVER respond with the actual value!
- NEVER include backticks around IDs in your response!
- NEVER include scores or written justification in your response!
- Respond in RANKED DESCENDING order, where the FIRST item is MOST RELEVANT
- Respond in JSON format: {"objects": ["<ID1>", "<ID2>", ...]}

Here are the objects to be ranked:

"""
        
        for obj in batch:
            prompt += f"id: `{obj.id}`\nvalue:\n```\n{obj.value}\n```\n\n"
        
        return prompt
    
    def _validate_and_fix_response(
        self, 
        response_data: Dict[str, Any], 
        input_ids: set[str]
    ) -> tuple[Dict[str, Any] | None, list[str] | None]:
        """Validate response and return fixed data or missing IDs."""
        if 'objects' not in response_data:
            return None, None
        
        # Clean IDs (remove backticks, fix case)
        input_ids_lower = {id.lower(): id for id in input_ids}
        cleaned_objects = []
        missing_ids = set(input_ids)
        
        for obj_id in response_data['objects']:
            clean_id = obj_id.replace('`', '').strip()
            
            # Try exact match first
            if clean_id in input_ids:
                cleaned_objects.append(clean_id)
                missing_ids.discard(clean_id)
            # Try case-insensitive match
            elif clean_id.lower() in input_ids_lower:
                correct_id = input_ids_lower[clean_id.lower()]
                cleaned_objects.append(correct_id)
                missing_ids.discard(correct_id)
        
        response_data['objects'] = cleaned_objects
        
        if missing_ids:
            return response_data, list(missing_ids)
        
        return response_data, None
    
    async def rank_batch(
        self, 
        batch: List[ProcessingObject], 
        prompt: str,
        run_num: int,
        batch_num: int
    ) -> List[RankedObject]:
        """Rank a batch using OpenAI API."""
        logger.info(f"OpenAI: Run {run_num}, Batch {batch_num} - {len(batch)} objects")
        
        full_prompt = self._build_prompt(batch, prompt)
        input_ids = {obj.id for obj in batch}
        
        conversation = [{"role": "user", "content": full_prompt}]
        max_retries = 5
        backoff = 1.0
        
        for attempt in range(max_retries):
            try:
                # Create response with enhanced timeout and error handling
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.config.openai_model.value,
                        messages=conversation,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "ranked_objects",
                                "schema": self.schema,
                                "strict": True
                            }
                        },
                        timeout=30.0
                    ),
                    timeout=45.0  # Slightly longer timeout for the overall operation
                )
                
                if not response.choices:
                    logger.warning("Empty response from OpenAI API")
                    conversation.append({
                        "role": "user", 
                        "content": "You provided an empty response. Please rank the objects as requested."
                    })
                    continue
                
                content = response.choices[0].message.content
                if not content:
                    logger.warning("Empty content in OpenAI response")
                    conversation.append({
                        "role": "user", 
                        "content": "Your response was empty. Please provide the ranking as requested."
                    })
                    continue
                
                conversation.append({"role": "assistant", "content": content})
                
                try:
                    response_data = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON response: {content[:200]}... Error: {e}")
                    conversation.append({
                        "role": "user", 
                        "content": "Your response was not valid JSON. Please try again and ensure proper JSON format!"
                    })
                    continue
                
                # Validate and fix response
                fixed_data, missing_ids = self._validate_and_fix_response(response_data, input_ids)
                
                if missing_ids:
                    logger.warning(f"Missing IDs in response: {missing_ids}")
                    conversation.append({
                        "role": "user",
                        "content": f"Your response was missing these IDs: {missing_ids}. "
                                 "Please include ALL IDs in your response!"
                    })
                    continue
                
                # Convert to RankedObject list
                ranked_objects = []
                for i, obj_id in enumerate(fixed_data['objects']):
                    for obj in batch:
                        if obj.id == obj_id:
                            ranked_objects.append(RankedObject(
                                obj=obj,
                                score=float(i + 1)  # Position-based scoring
                            ))
                            break
                
                if not ranked_objects:
                    logger.error("No valid ranked objects created from response")
                    conversation.append({
                        "role": "user",
                        "content": "No valid objects were found in your response. Please provide the ranking correctly."
                    })
                    continue
                
                logger.info(f"Successfully ranked {len(ranked_objects)} objects")
                return ranked_objects
                
            except asyncio.TimeoutError:
                logger.error(f"OpenAI API timeout (attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"OpenAI API timed out after {max_retries} attempts")
                
                await asyncio.sleep(backoff)
                backoff *= 2
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"OpenAI API error (attempt {attempt + 1}): {error_msg}")
                
                # Check for specific error types
                if "rate limit" in error_msg.lower():
                    logger.warning("Rate limit hit, waiting longer before retry")
                    await asyncio.sleep(backoff * 2)
                elif "timeout" in error_msg.lower():
                    logger.warning("Timeout error, increasing backoff")
                    await asyncio.sleep(backoff)
                elif "connection" in error_msg.lower():
                    logger.warning("Connection error, retrying with backoff")
                    await asyncio.sleep(backoff)
                else:
                    await asyncio.sleep(backoff)
                
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to rank batch after {max_retries} attempts: {error_msg}")
                
                backoff *= 2
        
        raise RuntimeError(f"Failed to rank batch after {max_retries} attempts")


class OpenRouterClient(LLMClient):
    """OpenRouter API client for ranking."""
    
    def __init__(self, config: RankingConfig):
        self.config = config
        
        client_kwargs = {
            'base_url': config.openrouter_base_url.rstrip('/') + '/'
        }
        if config.openrouter_api_key:
            client_kwargs['api_key'] = config.openrouter_api_key
        
        self.client = AsyncOpenAI(**client_kwargs)
        self.schema = self._get_response_schema()
    
    def _get_response_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for structured responses."""
        return {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ranked object IDs"
                }
            },
            "required": ["objects"],
            "additionalProperties": False
        }
    
    def _build_prompt(self, batch: List[ProcessingObject], user_prompt: str) -> str:
        """Build the full prompt for ranking."""
        prompt = f"{user_prompt}\n\n"
        prompt += """
REMEMBER to:
- ALWAYS respond with the short 8-character ID of each item found above the value
- NEVER respond with the actual value!
- NEVER include backticks around IDs in your response!
- NEVER include scores or written justification in your response!
- Respond in RANKED DESCENDING order, where the FIRST item is MOST RELEVANT
- Respond in JSON format: {"objects": ["<ID1>", "<ID2>", ...]}

Here are the objects to be ranked:

"""
        
        for obj in batch:
            prompt += f"id: `{obj.id}`\nvalue:\n```\n{obj.value}\n```\n\n"
        
        return prompt
    
    def _validate_and_fix_response(
        self, 
        response_data: Dict[str, Any], 
        input_ids: set[str]
    ) -> tuple[Dict[str, Any] | None, list[str] | None]:
        """Validate response and return fixed data or missing IDs."""
        if 'objects' not in response_data:
            return None, None
        
        # Clean IDs (remove backticks, fix case)
        input_ids_lower = {id.lower(): id for id in input_ids}
        cleaned_objects = []
        missing_ids = set(input_ids)
        
        for obj_id in response_data['objects']:
            clean_id = obj_id.replace('`', '').strip()
            
            # Try exact match first
            if clean_id in input_ids:
                cleaned_objects.append(clean_id)
                missing_ids.discard(clean_id)
            # Try case-insensitive match
            elif clean_id.lower() in input_ids_lower:
                correct_id = input_ids_lower[clean_id.lower()]
                cleaned_objects.append(correct_id)
                missing_ids.discard(correct_id)
        
        response_data['objects'] = cleaned_objects
        
        if missing_ids:
            return response_data, list(missing_ids)
        
        return response_data, None
    
    async def rank_batch(
        self, 
        batch: List[ProcessingObject], 
        prompt: str,
        run_num: int,
        batch_num: int
    ) -> List[RankedObject]:
        """Rank a batch using OpenRouter API."""
        logger.info(f"OpenRouter: Run {run_num}, Batch {batch_num} - {len(batch)} objects")
        
        full_prompt = self._build_prompt(batch, prompt)
        input_ids = {obj.id for obj in batch}
        
        conversation = [{"role": "user", "content": full_prompt}]
        max_retries = 5
        backoff = 1.0
        
        for attempt in range(max_retries):
            try:
                # Create response with enhanced timeout and error handling
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.config.openrouter_model,
                        messages=conversation,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "ranked_objects",
                                "schema": self.schema,
                                "strict": True
                            }
                        },
                        timeout=30.0
                    ),
                    timeout=45.0  # Slightly longer timeout for the overall operation
                )
                
                if not response.choices:
                    logger.warning("Empty response from OpenRouter API")
                    conversation.append({
                        "role": "user", 
                        "content": "You provided an empty response. Please rank the objects as requested."
                    })
                    continue
                
                content = response.choices[0].message.content
                if not content:
                    logger.warning("Empty content in OpenRouter response")
                    conversation.append({
                        "role": "user", 
                        "content": "Your response was empty. Please provide the ranking as requested."
                    })
                    continue
                
                conversation.append({"role": "assistant", "content": content})
                
                try:
                    response_data = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON response: {content[:200]}... Error: {e}")
                    conversation.append({
                        "role": "user", 
                        "content": "Your response was not valid JSON. Please try again and ensure proper JSON format!"
                    })
                    continue
                
                # Validate and fix response
                fixed_data, missing_ids = self._validate_and_fix_response(response_data, input_ids)
                
                if missing_ids:
                    logger.warning(f"Missing IDs in response: {missing_ids}")
                    conversation.append({
                        "role": "user",
                        "content": f"Your response was missing these IDs: {missing_ids}. "
                                 "Please include ALL IDs in your response!"
                    })
                    continue
                
                # Convert to RankedObject list
                ranked_objects = []
                for i, obj_id in enumerate(fixed_data['objects']):
                    for obj in batch:
                        if obj.id == obj_id:
                            ranked_objects.append(RankedObject(
                                obj=obj,
                                score=float(i + 1)  # Position-based scoring
                            ))
                            break
                
                if not ranked_objects:
                    logger.error("No valid ranked objects created from response")
                    conversation.append({
                        "role": "user",
                        "content": "No valid objects were found in your response. Please provide the ranking correctly."
                    })
                    continue
                
                logger.info(f"Successfully ranked {len(ranked_objects)} objects")
                return ranked_objects
                
            except asyncio.TimeoutError:
                logger.error(f"OpenRouter API timeout (attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"OpenRouter API timed out after {max_retries} attempts")
                
                await asyncio.sleep(backoff)
                backoff *= 2
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"OpenRouter API error (attempt {attempt + 1}): {error_msg}")
                
                # Check for specific error types
                if "rate limit" in error_msg.lower():
                    logger.warning("Rate limit hit, waiting longer before retry")
                    await asyncio.sleep(backoff * 2)
                elif "timeout" in error_msg.lower():
                    logger.warning("Timeout error, increasing backoff")
                    await asyncio.sleep(backoff)
                elif "connection" in error_msg.lower():
                    logger.warning("Connection error, retrying with backoff")
                    await asyncio.sleep(backoff)
                else:
                    await asyncio.sleep(backoff)
                
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to rank batch after {max_retries} attempts: {error_msg}")
                
                backoff *= 2
        
        raise RuntimeError(f"Failed to rank batch after {max_retries} attempts")


def create_llm_client(config: RankingConfig) -> LLMClient:
    """Factory function to create appropriate LLM client."""
    if config.provider.value == "openai":
        return OpenAIClient(config)
    elif config.provider.value == "openrouter":
        if not config.openrouter_model:
            raise ValueError("OpenRouter model name is required when using OpenRouter provider")
        return OpenRouterClient(config)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")