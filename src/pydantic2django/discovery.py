"""
High-level interface for model discovery and registration.

This module provides a simplified interface for discovering and registering
Pydantic models as Django models.
"""
from typing import Dict, Type, Optional, List
import logging

from django.db import models
from pydantic import BaseModel

from .registry import ModelRegistryManager

logger = logging.getLogger(__name__)

# Global registry instance
_registry: Optional[ModelRegistryManager] = None


def get_registry() -> ModelRegistryManager:
    """Get or create the global registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistryManager()
    return _registry


def discover_models(package_names: List[str], app_label: str = "django_llm") -> None:
    """
    Discover and analyze Pydantic models from specified packages.
    
    Args:
        package_names: List of package names to search for models
        app_label: The Django app label to use for model registration
    """
    registry = get_registry()
    
    logger.info(f"Discovering models from packages {package_names} for {app_label}...")
    
    # Discover models from all packages
    for package_name in package_names:
        registry.discover_models(package_name, app_label=app_label)
    
    logger.info(f"Discovered {len(registry.discovered_models)} models")


def setup_dynamic_models(app_label: str = "django_llm", skip_admin: bool = False) -> Dict[str, Type[models.Model]]:
    """
    Set up dynamic models from discovered Pydantic models.
    
    This function should be called during migration operations, not during app initialization.
    It creates Django models from discovered Pydantic models.
    
    Args:
        app_label: The Django app label to use for model registration
        skip_admin: Whether to skip registering models with the Django admin interface
        
    Returns:
        Dict mapping model names to Django model classes
    """
    registry = get_registry()
    return registry.setup_models(app_label, skip_admin)


def get_discovered_models() -> Dict[str, Type[BaseModel]]:
    """Get all discovered Pydantic models."""
    return get_registry().discovered_models


def get_django_models(app_label: str = "django_llm") -> Dict[str, Type[models.Model]]:
    """Get all registered Django models for the given app label."""
    return get_registry().django_models 