import abc  # Import abc
import inspect
import logging
from collections.abc import Callable
from typing import Any, Optional, Union, get_args, get_origin

from pydantic import BaseModel

from ..core.discovery import BaseDiscovery

logger = logging.getLogger(__name__)


class PydanticDiscovery(BaseDiscovery[type[BaseModel]]):
    """Discovers Pydantic models within specified packages."""

    # __init__ is inherited and sufficient

    def _is_target_model(self, obj: Any) -> bool:
        """Check if an object is a Pydantic BaseModel, excluding the base itself."""
        return inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel

    def _default_eligibility_filter(self, model: type[BaseModel]) -> bool:
        """Check default eligibility: not abstract and not inheriting directly from ABC."""
        # Skip models that directly inherit from ABC
        if abc.ABC in model.__bases__:
            logger.debug(f"Filtering out {model.__name__} (inherits directly from ABC)")
            return False

        # Skip models that are marked as abstract
        if getattr(model, "__abstract__", False):
            logger.debug(f"Filtering out {model.__name__} (marked as __abstract__)")
            return False

        # Example for potentially filtering Pydantic internal models (uncomment if needed)
        # if model.__module__.startswith('pydantic._internal'):
        #     logger.debug(f"Filtering out internal Pydantic model: {model.__name__}")
        #     return False

        return True  # Eligible by default

    def discover_models(
        self,
        packages: list[str],
        app_label: str,
        user_filters: Optional[
            Union[Callable[[type[BaseModel]], bool], list[Callable[[type[BaseModel]], bool]]]
        ] = None,
    ):
        """Discover Pydantic models in the specified packages, applying filters."""
        # Pass user_filters directly to the base class method
        super().discover_models(packages, app_label, user_filters=user_filters)

    # --- analyze_dependencies and get_models_in_registration_order remain ---

    def analyze_dependencies(self) -> None:
        """Build the dependency graph for the filtered Pydantic models."""
        logger.info("Analyzing dependencies between filtered Pydantic models...")
        self.dependencies: dict[type[BaseModel], set[type[BaseModel]]] = {}

        filtered_model_qualnames = set(self.filtered_models.keys())

        def _find_and_add_dependency(model_type: type[BaseModel], potential_dep_type: Any):
            if not self._is_target_model(potential_dep_type):
                return

            dep_qualname = f"{potential_dep_type.__module__}.{potential_dep_type.__name__}"

            if dep_qualname in filtered_model_qualnames and potential_dep_type is not model_type:
                dep_model_obj = self.filtered_models.get(dep_qualname)
                if dep_model_obj:
                    if model_type in self.dependencies:
                        self.dependencies[model_type].add(dep_model_obj)
                    else:
                        # Initialize if missing (shouldn't happen often with new base discover_models)
                        logger.warning(
                            f"Model {model_type.__name__} wasn't pre-initialized in dependencies dict during analysis. Initializing now."
                        )
                        self.dependencies[model_type] = {dep_model_obj}
                else:
                    logger.warning(
                        f"Inconsistency: Dependency '{dep_qualname}' for model '{model_type.__name__}' found by name but not object in filtered set."
                    )

        # Initialize keys based on filtered models (important step)
        for model_type in self.filtered_models.values():
            self.dependencies[model_type] = set()

        # Analyze fields
        for model_type in self.filtered_models.values():
            for field in model_type.model_fields.values():
                annotation = field.annotation
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
                            nested_model_type = next(t for t in arg_args if t is not type(None))
                            _find_and_add_dependency(model_type, nested_model_type)
                        else:
                            _find_and_add_dependency(model_type, arg)

        logger.info("Dependency analysis complete.")
        # Debug logging moved inside BaseDiscovery

    def get_models_in_registration_order(self) -> list[type[BaseModel]]:
        """
        Return models sorted topologically based on dependencies.
        Models with no dependencies come first.
        """
        if not self.dependencies:
            logger.warning("No dependencies found or analyzed, returning Pydantic models in arbitrary order.")
            return list(self.filtered_models.values())

        sorted_models = []
        visited: set[type[BaseModel]] = set()
        visiting: set[type[BaseModel]] = set()
        filtered_model_objects = set(self.filtered_models.values())

        def visit(model: type[BaseModel]):
            if model in visited:
                return
            if model in visiting:
                logger.error(f"Circular dependency detected involving Pydantic model {model.__name__}")
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

        logger.info(f"Pydantic models sorted for registration: {[m.__name__ for m in sorted_models]}")
        return sorted_models

    # Subclasses can override _resolve_forward_refs if Pydantic-specific handling is needed
    # For Pydantic V2, the base placeholder is likely sufficient.
    # For V1, might need: model.update_forward_refs(**global_vars, **local_vars)
    # def _resolve_forward_refs(self) -> None:
    #     logger.debug("Attempting Pydantic-specific forward ref resolution...")
    #     # Add V1 logic here if required
    #     super()._resolve_forward_refs() # Call base if needed
