"""
Model registry management for pydantic2django.

This module provides utilities for discovering, tracking dependencies,
and managing registration of Pydantic models as Django models.
"""
import importlib
import inspect
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Type, Any, Optional, get_args, get_origin

from django.db import models
from pydantic import BaseModel
from django.apps import apps

from .core import (
    make_django_model,
    clear_model_registry,
    normalize_model_name,
    get_model_dependencies_recursive,
    validate_model_references,
    topological_sort,
)


def is_pydantic_model(obj: Type) -> bool:
    """Check if an object is a Pydantic model class."""
    return (
        inspect.isclass(obj) 
        and issubclass(obj, BaseModel) 
        and obj != BaseModel
    )


def get_concrete_type_name(type_: Type[Any]) -> str:
    """Get a concrete type name for a generic type."""
    if hasattr(type_, "__origin__"):
        origin = get_origin(type_)
        args = get_args(type_)
        if origin and args:
            arg_names = [get_concrete_type_name(arg) for arg in args]
            return f"{origin.__name__}{''.join(arg_names)}"
    return type_.__name__


class ModelRegistryManager:
    """
    Manages the discovery, dependency tracking, and registration order of Pydantic models
    for conversion to Django models.
    
    This class provides utilities for Django apps to:
    1. Discover Pydantic models in their packages
    2. Track dependencies between models
    3. Determine the correct order for model registration
    4. Register models in the proper order
    
    Example usage:
        registry = ModelRegistryManager()
        registry.discover_models("my_app")
        registry.discover_models("another_app")
        
        # Get registration order
        ordered_models = registry.get_registration_order()
        
        # Register models
        django_models = registry.register_models()
    """
    
    def __init__(self):
        self.discovered_models: Dict[str, Type[BaseModel]] = {}
        self.normalized_models: Dict[str, Type[BaseModel]] = {}
        self.dependencies: Dict[str, Set[str]] = {}
        self.django_models: Dict[str, Type[models.Model]] = {}
        self._registration_order: Optional[List[str]] = None
        
    def discover_models(self, package_name: str, app_label: str = "django_llm") -> Dict[str, Type[BaseModel]]:
        """
        Discover Pydantic models in a package.
        
        Args:
            package_name: The package to search for models
            app_label: The Django app label to use for model registration
            
        Returns:
            Dict of discovered models
        """
        discovered_models = {}
        
        try:
            # Get the package
            package = importlib.import_module(package_name)
            if not package.__file__:
                return discovered_models
                
            package_path = Path(package.__file__).parent
            
            # Walk through all Python files in the package
            for py_file in package_path.rglob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                    
                # Convert file path to module path
                relative_path = py_file.relative_to(package_path)
                module_path = f"{package_name}.{'.'.join(relative_path.with_suffix('').parts)}"
                
                try:
                    # Import the module
                    module = importlib.import_module(module_path)
                    
                    # Find all Pydantic models in the module
                    for name, obj in inspect.getmembers(module):
                        if is_pydantic_model(obj):
                            model_name = f"Django{name}"
                            discovered_models[model_name] = obj
                            
                except Exception as e:
                    print(f"Error importing {module_path}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error discovering models in {package_name}: {e}")
            
        # Update registry state
        self.discovered_models.update(discovered_models)
        
        # Update normalized models (but defer dependency analysis)
        for model_name, model_cls in discovered_models.items():
            norm_name = normalize_model_name(model_name)
            self.normalized_models[norm_name] = model_cls
            self.dependencies[norm_name] = set()  # Initialize empty dependencies
                
        return discovered_models

    def analyze_dependencies(self, app_label: str) -> None:
        """
        Analyze dependencies between models after they've been registered.
        Should be called after all models are registered with Django.
        
        Args:
            app_label: The Django app label the models are registered under
        """
        for model_name, model_cls in self.normalized_models.items():
            try:
                # Get the registered Django model
                django_model = apps.get_model(app_label, model_name.replace("Django", ""))
                deps = get_model_dependencies_recursive(django_model, app_label)
                self.dependencies[model_name] = deps
            except Exception as e:
                print(f"Warning: Could not analyze dependencies for {model_name}: {e}")
                continue
    
    def validate_dependencies(self) -> List[str]:
        """
        Validate that all model dependencies exist.
        
        Returns:
            List of error messages for missing dependencies
        """
        return validate_model_references(self.normalized_models, self.dependencies)
    
    def get_registration_order(self) -> List[str]:
        """
        Get the order in which models should be registered based on their dependencies.
        
        Returns:
            List of model names in dependency order
        """
        if self._registration_order is None:
            errors = self.validate_dependencies()
            if errors:
                raise ValueError(f"Cannot determine registration order due to dependency errors: {errors}")
            self._registration_order = topological_sort(self.dependencies)
        return self._registration_order
    
    def register_models(self, app_label: str = "django_llm") -> Dict[str, Type[models.Model]]:
        """
        Register all discovered models in dependency order.
        
        Args:
            app_label: The Django app label to use for registration
            
        Returns:
            Dict mapping model names to registered Django model classes
        """
        # Clear existing registrations
        clear_model_registry()
        
        # Get registration order
        ordered_models = self.get_registration_order()
        
        # Register models in order
        for full_name in ordered_models:
            try:
                _, model_name = full_name.split(".")
                if model_name not in self.normalized_models:
                    continue
                    
                pydantic_model = self.normalized_models[model_name]
                
                # Create and register Django model
                django_model, _ = make_django_model(
                    pydantic_model,
                    app_label=app_label,
                    db_table=f"{app_label}_{model_name.lower()}",
                    check_migrations=False,
                )
                self.django_models[model_name] = django_model
                
            except Exception as e:
                print(f"Error registering model {full_name}: {e}")
                continue
                
        # Analyze dependencies after all models are registered
        self.analyze_dependencies(app_label)
        
        return self.django_models
    
    def get_app_dependencies(self) -> Dict[str, Set[str]]:
        """
        Get a mapping of app dependencies based on model relationships.
        This can be used by Django apps to determine initialization order.
        
        Returns:
            Dict mapping app labels to their dependent app labels
        """
        app_deps = defaultdict(set)
        
        for model_name, deps in self.dependencies.items():
            app_label = model_name.split(".")[0]
            for dep in deps:
                dep_app = dep.split(".")[0]
                if dep_app != app_label:
                    app_deps[app_label].add(dep_app)
                    
        return dict(app_deps)
    
    def clear(self):
        """Clear all discovered and registered models."""
        self.discovered_models.clear()
        self.normalized_models.clear()
        self.dependencies.clear()
        self.django_models.clear()
        self._registration_order = None
        clear_model_registry() 