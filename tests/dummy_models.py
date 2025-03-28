from typing import Dict, List, Optional
from pydantic import BaseModel


class ProviderCapabilities(BaseModel):
    """Dummy model representing provider capabilities for testing import functionality."""

    supports_chat: bool = True
    supports_completion: bool = True
    supports_embedding: bool = False
    max_tokens: int = 4096
    available_models: List[str] = []


class RateLimitConfig(BaseModel):
    """Dummy model representing rate limit configuration for testing import functionality."""

    requests_per_minute: int = 60
    tokens_per_minute: int = 40000
    backup_providers: List[str] = []
    retry_strategy: Dict[str, int] = {}
