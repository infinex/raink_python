"""
Custom exceptions for the raink FastAPI application.
"""

from typing import Optional, Dict, Any


class RainkException(Exception):
    """Base exception for raink application."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(RainkException):
    """Raised when input validation fails."""
    pass


class ConfigurationError(RainkException):
    """Raised when configuration is invalid."""
    pass


class TokenLimitError(RainkException):
    """Raised when token limits are exceeded."""
    pass


class LLMAPIError(RainkException):
    """Raised when LLM API calls fail."""
    pass


class RankingError(RainkException):
    """Raised when ranking process fails."""
    pass


class TimeoutError(RainkException):
    """Raised when operations timeout."""
    pass