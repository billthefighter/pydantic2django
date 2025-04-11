import abc  # Import abc
import dataclasses
import inspect
import logging
from typing import Any, Union, get_args, get_origin

# from .core import BaseDiscovery # Incorrect relative import
from pydantic2django.core.discovery import BaseDiscovery  # Corrected import path

logger = logging.getLogger(__name__)

# Refined placeholder type alias
DataclassType = type[Any]  # Represents a type that is a dataclass


class DataclassDiscovery(BaseDiscovery[DataclassType]):
    """Discovers Python dataclasses within specified packages."""

    def __init__(self):
        super().__init__()
        # Dataclass specific attributes (all_models now in base)

    def _is_target_model(self, obj: Any) -> bool:
        """Check if an object is a dataclass type."""
        return inspect.isclass(obj) and dataclasses.is_dataclass(obj)

    def _default_eligibility_filter(self, model: DataclassType) -> bool:
        """Check default eligibility for dataclasses (e.g., not inheriting directly from ABC)."""
        # Skip models that directly inherit from ABC
        if abc.ABC in model.__bases__:
            logger.debug(f"Filtering out dataclass {model.__name__} (inherits directly from ABC)")
            return False
        # Dataclasses don't have a standard __abstract__ marker like Pydantic
        # Add other default checks if needed for dataclasses
        return True

    # discover_models is now implemented in the BaseDiscovery class
    # It will call the _is_target_model and _default_eligibility_filter defined above.

    # --- analyze_dependencies and get_models_in_registration_order remain ---

    def analyze_dependencies(self) -> None:
        """Build the dependency graph for the filtered dataclasses."""
        logger.info("Analyzing dependencies between filtered dataclasses...")
        self.dependencies: dict[DataclassType, set[DataclassType]] = {}

        filtered_model_qualnames = set(self.filtered_models.keys())

        def _find_and_add_dependency(model_type: DataclassType, potential_dep_type: Any):
            if not self._is_target_model(potential_dep_type):
                return

            dep_qualname = f"{potential_dep_type.__module__}.{potential_dep_type.__name__}"

            if dep_qualname in filtered_model_qualnames and potential_dep_type is not model_type:
                dep_model_obj = self.filtered_models.get(dep_qualname)
                if dep_model_obj:
                    if model_type in self.dependencies:
                        self.dependencies[model_type].add(dep_model_obj)
                    else:
                        logger.warning(
                            f"Model {model_type.__name__} wasn't pre-initialized in dependencies dict during analysis. Initializing now."
                        )
                        self.dependencies[model_type] = {dep_model_obj}
                else:
                    logger.warning(
                        f"Inconsistency: Dependency '{dep_qualname}' for dataclass '{model_type.__name__}' found by name but not object in filtered set."
                    )

        # Initialize keys based on filtered models
        for model_type in self.filtered_models.values():
            self.dependencies[model_type] = set()

        # Analyze fields using dataclasses.fields
        for model_type in self.filtered_models.values():
            assert dataclasses.is_dataclass(model_type), f"Expected {model_type} to be a dataclass"
            for field in dataclasses.fields(model_type):
                annotation = field.type
                if annotation is None:
                    continue

                origin = get_origin(annotation)
                args = get_args(annotation)

                if origin is Union and type(None) in args and len(args) == 2:
                    annotation = next(arg for arg in args if arg is not type(None))
                    origin = get_origin(annotation)
                    args = get_args(annotation)

                _find_and_add_dependency(model_type, annotation)

                if origin in (list, dict, set, tuple):
                    for arg in args:
                        arg_origin = get_origin(arg)
                        arg_args = get_args(arg)

                        if arg_origin is Union and type(None) in arg_args and len(arg_args) == 2:
                            nested_type = next(t for t in arg_args if t is not type(None))
                            _find_and_add_dependency(model_type, nested_type)
                        else:
                            _find_and_add_dependency(model_type, arg)

        logger.info("Dataclass dependency analysis complete.")
        # Debug logging moved inside BaseDiscovery

    def get_models_in_registration_order(self) -> list[DataclassType]:
        """
        Return dataclasses sorted topologically based on dependencies.
        (Largely similar to Pydantic version, uses DataclassType)
        """
        if not self.dependencies:
            logger.warning("No dependencies found or analyzed, returning dataclasses in arbitrary order.")
            return list(self.filtered_models.values())

        sorted_models = []
        visited: set[DataclassType] = set()
        visiting: set[DataclassType] = set()
        filtered_model_objects = set(self.filtered_models.values())

        def visit(model: DataclassType):
            if model in visited:
                return
            if model in visiting:
                logger.error(f"Circular dependency detected involving dataclass {model.__name__}")
                # Option: raise TypeError(...)
                return  # Break cycle

            visiting.add(model)

            if model in self.dependencies:
                # Use .get for safety, ensure deps are also in filtered set
                for dep in self.dependencies.get(model, set()):
                    if dep in filtered_model_objects:
                        visit(dep)

            visiting.remove(model)
            visited.add(model)
            sorted_models.append(model)

        all_target_models = list(self.filtered_models.values())
        for model in all_target_models:
            if model not in visited:
                visit(model)

        logger.info(f"Dataclasses sorted for registration: {[m.__name__ for m in sorted_models]}")
        return sorted_models

    # Dataclasses typically don't require complex forward ref resolution like Pydantic V1
    # The base placeholder should suffice.
    # def _resolve_forward_refs(self) -> None:
    #    logger.debug("Attempting Dataclass-specific forward ref resolution...")
    #    # Add specific logic if needed, e.g., using typing.get_type_hints
    #    super()._resolve_forward_refs()
