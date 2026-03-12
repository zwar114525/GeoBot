"""
JSON validation utilities with Pydantic schemas and retry logic.
Provides robust parsing of LLM responses with fallback mechanisms.
"""
import json
import re
import time
from typing import Any, Type, TypeVar, Optional, List
from pydantic import BaseModel, ValidationError, Field
from loguru import logger

from src.utils.llm_client import call_llm

T = TypeVar("T", bound=BaseModel)


def extract_json_from_response(response: str) -> str:
    """
    Extract JSON content from LLM response that may contain markdown code blocks.
    
    Args:
        response: Raw LLM response text
        
    Returns:
        Cleaned JSON string
    """
    if not response:
        return ""
    
    cleaned = response.strip()
    
    # Try to find JSON between markdown code blocks first
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    # Try to find JSON array between markdown code blocks
    array_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, re.DOTALL)
    if array_match:
        return array_match.group(1).strip()
    
    # If no markdown blocks, try to remove common prefixes
    if cleaned.startswith("```"):
        # Remove first and last line if they contain only ```
        lines = cleaned.split("\n")
        if len(lines) > 2:
            # Remove first line (``` or ```json)
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            # Remove last line (```)
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)
    
    return cleaned.strip()


def parse_json_with_retry(
    response: str,
    target_type: Type[T],
    max_retries: int = 3,
    retry_delay: float = 1.0,
    context: str = "",
) -> T:
    """
    Parse JSON response with validation and retry logic.
    
    Args:
        response: Raw LLM response
        target_type: Pydantic model class to validate against
        max_retries: Number of retry attempts
        retry_delay: Base delay between retries (exponential backoff)
        context: Description of what we're parsing for error messages
        
    Returns:
        Validated Pydantic model instance
        
    Raises:
        ValueError: If parsing fails after all retries
    """
    last_error: Optional[Exception] = None
    
    for attempt in range(max_retries):
        try:
            cleaned = extract_json_from_response(response)
            
            if not cleaned:
                raise ValueError("Empty JSON response from LLM")
            
            # Try parsing with Pydantic
            if hasattr(target_type, "model_validate_json"):
                # Pydantic v2
                return target_type.model_validate_json(cleaned)
            else:
                # Fallback for older pydantic
                data = json.loads(cleaned)
                return target_type(**data)
                
        except ValidationError as e:
            last_error = e
            logger.warning(f"Validation error (attempt {attempt + 1}/{max_retries}): {e}")
            logger.debug(f"Invalid JSON: {cleaned[:500]}")
            
        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(f"JSON decode error (attempt {attempt + 1}/{max_retries}): {e}")
            logger.debug(f"Invalid JSON: {cleaned[:500]}")
            
        except Exception as e:
            last_error = e
            logger.warning(f"Parse error (attempt {attempt + 1}/{max_retries}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay * (2 ** attempt))
    
    error_msg = f"Failed to parse {context} after {max_retries} attempts. Last error: {last_error}"
    logger.error(error_msg)
    raise ValueError(error_msg) from last_error


def call_llm_with_json_validation(
    prompt: str,
    target_type: Type[T],
    system_prompt: str = "Return ONLY valid JSON. No explanations.",
    model: str = "",
    temperature: float = 0.1,
    max_tokens: int = 2000,
    max_retries: int = 3,
    context: str = "JSON response",
) -> T:
    """
    Call LLM and validate JSON response with retry logic.
    
    Args:
        prompt: User prompt
        target_type: Pydantic model to validate against
        system_prompt: System instruction
        model: Model to use (uses default if empty)
        temperature: LLM temperature (low for structured output)
        max_tokens: Max tokens in response
        max_retries: Number of retry attempts
        context: Description for error messages
        
    Returns:
        Validated Pydantic model instance
    """
    enhanced_system_prompt = (
        f"{system_prompt}\n\n"
        "IMPORTANT: Return ONLY valid JSON that matches the expected schema. "
        "Do not include markdown code blocks, explanations, or extra text. "
        "All required fields must be present."
    )
    
    for attempt in range(max_retries):
        try:
            response = call_llm(
                prompt=prompt,
                system_prompt=enhanced_system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            return parse_json_with_retry(
                response=response,
                target_type=target_type,
                max_retries=2,
                retry_delay=0.5,
                context=context,
            )
            
        except ValueError as e:
            logger.warning(f"LLM JSON validation failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                # Retry with stronger prompt
                enhanced_system_prompt = (
                    f"{enhanced_system_prompt}\n\n"
                    "CRITICAL: The previous response was invalid. "
                    "You MUST return valid JSON with all required fields."
                )
                time.sleep(1.0 * (2 ** attempt))
            else:
                raise
    
    raise ValueError(f"Failed to get valid JSON after {max_retries} LLM calls")


def safe_parse_json(
    response: str,
    target_type: Type[T],
    default_factory: callable,
    context: str = "",
) -> T:
    """
    Safely parse JSON with a fallback default value.
    
    Args:
        response: Raw LLM response
        target_type: Pydantic model class
        default_factory: Function that returns default value on failure
        context: Description for logging
        
    Returns:
        Validated model or default value
    """
    try:
        # First extract JSON from potential markdown wrapper
        cleaned = extract_json_from_response(response)
        if not cleaned:
            return default_factory()
        
        if hasattr(target_type, "model_validate_json"):
            return target_type.model_validate_json(cleaned)
        else:
            data = json.loads(cleaned)
            return target_type(**data)
            
    except (json.JSONDecodeError, ValidationError) as e:
        # Try to parse as raw JSON if markdown extraction failed
        try:
            if hasattr(target_type, "model_validate_json"):
                return target_type.model_validate_json(response)
            else:
                data = json.loads(response)
                return target_type(**data)
        except Exception:
            logger.warning(f"Safe parse failed for {context}: {e}")
            return default_factory()
