"""
Core ranking logic for the raink FastAPI application.
Port of the Go ranking algorithm to Python.
"""

import asyncio
import hashlib
import base64
import random
import time
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import tiktoken
from loguru import logger

from ..models import RankingObject, RankingConfig, RankedResult


@dataclass
class ProcessingObject:
    """Internal object representation for processing."""
    id: str
    value: str
    metadata: Optional[Dict[str, Any]] = None
    original: Optional[RankingObject] = None


@dataclass
class RankedObject:
    """Object with ranking score."""
    obj: ProcessingObject
    score: float


class RankingEngine:
    """Core ranking engine that implements the tournament-style ranking algorithm."""
    
    def __init__(self, config: RankingConfig):
        self.config = config
        self.encoding = None
        self.rng = random.Random()
        self.round = 1
        self.num_batches = 0
        
        # Initialize tokenizer
        try:
            self.encoding = tiktoken.get_encoding(config.encoding)
        except Exception as e:
            logger.warning(f"Failed to load encoding {config.encoding}, using default: {e}")
            self.encoding = tiktoken.get_encoding("o200k_base")
    
    def generate_short_id(self, input_text: str, length: int = 8) -> str:
        """Generate a short deterministic ID from input text."""
        hash_obj = hashlib.sha256(input_text.encode())
        base64_encoded = base64.urlsafe_b64encode(hash_obj.digest()).decode()
        
        # Keep only alphanumeric characters
        filtered = ''.join(c for c in base64_encoded if c.isalnum())
        return filtered[:min(length, len(filtered))]
    
    def prepare_objects(self, objects: List[RankingObject]) -> List[ProcessingObject]:
        """Convert input objects to internal processing format."""
        processed = []
        for obj in objects:
            # Generate ID if not provided
            obj_id = obj.id if obj.id else self.generate_short_id(obj.value)
            
            processed.append(ProcessingObject(
                id=obj_id,
                value=obj.value,
                metadata=obj.metadata,
                original=obj
            ))
        
        return processed
    
    def validate_objects(self, objects: List[ProcessingObject]) -> None:
        """Validate that objects can be processed within token limits."""
        for obj in objects:
            tokens = self.estimate_tokens([obj], include_prompt=True)
            if tokens > self.config.token_limit:
                raise ValueError(
                    f"Object '{obj.id}' is too large with {tokens} tokens "
                    f"(limit: {self.config.token_limit})"
                )
    
    def adjust_batch_size(self, objects: List[ProcessingObject], samples: int = 10) -> None:
        """Dynamically adjust batch size to fit within token limits."""
        logger.info(f"Adjusting batch size for {len(objects)} objects")
        
        while self.config.batch_size >= 2:
            valid = True
            total_tokens = 0
            num_batches = 0
            
            # Test with multiple random samples
            for _ in range(samples):
                shuffled = objects.copy()
                self.rng.shuffle(shuffled)
                
                num_batches = max(1, len(shuffled) // self.config.batch_size)
                
                for i in range(num_batches):
                    start_idx = i * self.config.batch_size
                    end_idx = min((i + 1) * self.config.batch_size, len(shuffled))
                    batch = shuffled[start_idx:end_idx]
                    
                    batch_tokens = self.estimate_tokens(batch, include_prompt=True)
                    total_tokens += batch_tokens
                    
                    if batch_tokens > self.config.token_limit:
                        logger.warning(
                            f"Batch {i+1} estimated tokens {batch_tokens} > limit {self.config.token_limit}"
                        )
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                avg_tokens = total_tokens // (samples * num_batches) if num_batches > 0 else 0
                pct = (avg_tokens / self.config.token_limit) * 100
                logger.info(
                    f"Batch size {self.config.batch_size}: avg {avg_tokens} tokens "
                    f"({pct:.1f}% of limit)"
                )
                break
            
            self.config.batch_size -= 1
            logger.info(f"Reducing batch size to {self.config.batch_size}")
        
        if self.config.batch_size < 2:
            raise ValueError("Cannot create valid batches within token limit")
    
    def estimate_tokens(self, objects: List[ProcessingObject], include_prompt: bool = False) -> int:
        """Estimate token count for a batch of objects."""
        text = ""
        
        if include_prompt:
            # Add base prompt and instructions
            text += "Rank these objects according to the given criteria.\n\n"
            text += self._get_ranking_instructions()
        
        # Add object content
        for obj in objects:
            text += f"id: `{obj.id}`\nvalue:\n```\n{obj.value}\n```\n\n"
        
        # Use tiktoken for OpenAI models
        return len(self.encoding.encode(text))
    
    def _get_ranking_instructions(self) -> str:
        """Get the ranking instructions prompt."""
        return """
REMEMBER to:
- ALWAYS respond with the short 8-character ID of each item found above the value
- NEVER respond with the actual value!
- NEVER include backticks around IDs in your response!
- NEVER include scores or written justification in your response!
- Respond in RANKED DESCENDING order, where the FIRST item is MOST RELEVANT
- Respond in JSON format: {"objects": ["<ID1>", "<ID2>", ...]}

Here are the objects to be ranked:

"""
    
    async def rank(self, objects: List[ProcessingObject], prompt: str) -> List[RankedResult]:
        """Main ranking function implementing the tournament algorithm."""
        logger.info(f"Starting ranking of {len(objects)} objects")
        
        # Validate and adjust configuration
        self.validate_objects(objects)
        self.adjust_batch_size(objects)
        
        # Start recursive ranking
        results = await self._rank_recursive(objects, prompt, round_num=1)
        
        # Convert to final format
        final_results = []
        for i, result in enumerate(results):
            final_results.append(RankedResult(
                key=result['key'],
                value=result['value'],
                metadata=result.get('metadata'),
                score=result['score'],
                exposure=result['exposure'],
                rank=i + 1
            ))
        
        return final_results
    
    async def _rank_recursive(
        self, 
        objects: List[ProcessingObject], 
        prompt: str,
        round_num: int
    ) -> List[Dict[str, Any]]:
        """Recursive ranking implementation."""
        self.round = round_num
        logger.info(f"Round {round_num}: Ranking {len(objects)} objects")
        
        # Base case: single object
        if len(objects) == 1:
            return [{
                'key': objects[0].id,
                'value': objects[0].value,
                'metadata': objects[0].metadata,
                'score': 0.0,
                'exposure': 1
            }]
        
        # Adjust batch size if needed
        if self.config.batch_size > len(objects):
            self.config.batch_size = len(objects)
        
        self.num_batches = len(objects) // self.config.batch_size
        
        # Process with shuffle-batch-rank
        results = await self._shuffle_batch_rank(objects, prompt)
        
        # Check if we should continue with refinement
        if self.config.refinement_ratio == 0:
            return results
        
        # Calculate refinement split
        mid = int(len(results) * self.config.refinement_ratio)
        top_portion = results[:mid]
        bottom_portion = results[mid:]
        
        # If no reduction, we're done
        if len(top_portion) == len(objects):
            return results
        
        logger.info(f"Refining top {len(top_portion)} objects")
        
        # Convert back to ProcessingObject for recursion
        top_objects = []
        for result in top_portion:
            # Find original object
            for obj in objects:
                if obj.id == result['key']:
                    top_objects.append(obj)
                    break
        
        # Recursive call
        refined_top = await self._rank_recursive(top_objects, prompt, round_num + 1)
        
        # Adjust scores by depth (lower scores = better rank)
        for result in refined_top:
            result['score'] /= (2 * round_num)
        
        # Combine results
        return refined_top + bottom_portion
    
    async def _shuffle_batch_rank(
        self, 
        objects: List[ProcessingObject], 
        prompt: str
    ) -> List[Dict[str, Any]]:
        """Implement shuffle-batch-rank algorithm with multiple runs."""
        scores = {}
        exposure_counts = {}
        
        # Track remainder items for consistent processing
        first_run_remainder = []
        
        for run in range(self.config.num_runs):
            logger.info(f"Round {self.round}, Run {run+1}/{self.config.num_runs}")
            
            # Shuffle objects
            shuffled = objects.copy()
            self.rng.shuffle(shuffled)
            
            # Handle remainder items consistency (from Go implementation)
            if run == 1 and first_run_remainder:
                shuffled = self._ensure_remainder_consistency(shuffled, first_run_remainder)
            
            # Process batches concurrently
            batch_tasks = []
            for batch_idx in range(self.num_batches):
                start_idx = batch_idx * self.config.batch_size
                end_idx = min((batch_idx + 1) * self.config.batch_size, len(shuffled))
                batch = shuffled[start_idx:end_idx]
                
                task = self._rank_batch(batch, prompt, run + 1, batch_idx + 1)
                batch_tasks.append(task)
            
            # Wait for all batches to complete
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Aggregate results
            for ranked_batch in batch_results:
                for ranked_obj in ranked_batch:
                    obj_id = ranked_obj.obj.id
                    if obj_id not in scores:
                        scores[obj_id] = []
                        exposure_counts[obj_id] = 0
                    
                    scores[obj_id].append(ranked_obj.score)
                    exposure_counts[obj_id] += 1
            
            # Save remainder for first run
            if run == 0:
                remainder_start = self.num_batches * self.config.batch_size
                if remainder_start < len(shuffled):
                    first_run_remainder = shuffled[remainder_start:].copy()
        
        # Calculate final scores
        final_scores = {}
        for obj_id, score_list in scores.items():
            final_scores[obj_id] = sum(score_list) / len(score_list)
        
        # Create results
        results = []
        for obj in objects:
            if obj.id in final_scores:
                results.append({
                    'key': obj.id,
                    'value': obj.value,
                    'metadata': obj.metadata,
                    'score': final_scores[obj.id],
                    'exposure': exposure_counts[obj.id]
                })
        
        # Sort by score (lower is better)
        results.sort(key=lambda x: x['score'])
        
        return results
    
    def _ensure_remainder_consistency(
        self, 
        shuffled: List[ProcessingObject],
        first_run_remainder: List[ProcessingObject]
    ) -> List[ProcessingObject]:
        """Ensure remainder items don't conflict between runs."""
        max_attempts = 100
        attempt = 0
        
        while attempt < max_attempts:
            remainder_start = self.num_batches * self.config.batch_size
            current_remainder = shuffled[remainder_start:] if remainder_start < len(shuffled) else []
            
            # Check for conflicts
            conflict_found = False
            for curr_item in current_remainder:
                for first_item in first_run_remainder:
                    if curr_item.id == first_item.id:
                        conflict_found = True
                        break
                if conflict_found:
                    break
            
            if not conflict_found:
                break
            
            # Reshuffle and try again
            self.rng.shuffle(shuffled)
            attempt += 1
        
        if attempt == max_attempts:
            logger.warning("Could not resolve remainder conflicts after maximum attempts")
        
        return shuffled
    
    async def _rank_batch(
        self, 
        batch: List[ProcessingObject], 
        prompt: str,
        run_num: int, 
        batch_num: int
    ) -> List[RankedObject]:
        """Rank a single batch of objects."""
        logger.info(f"Round {self.round}, Run {run_num}, Batch {batch_num}: Processing {len(batch)} objects")
        
        if self.config.dry_run:
            # Simulate ranking for dry run
            ranked = []
            for i, obj in enumerate(batch):
                ranked.append(RankedObject(obj=obj, score=float(i + 1)))
            return ranked
        
        # Use the LLM client to rank the batch
        from .llm_clients import create_llm_client
        client = create_llm_client(self.config)
        return await client.rank_batch(batch, prompt, run_num, batch_num)