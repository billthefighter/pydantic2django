import dataclasses
import inspect
import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from pydantic2django.core.discovery import BaseDiscovery

logger = logging.getLogger(__name__)

# Placeholder for the type of classes we are targeting.
# This might be refined to something more specific than object if we introduce a marker.
T = TypeVar("T")  # Generic type for the source model
TypedClassType = type[Any]  # Represents a type that is a generic class instance


class TypedClassDiscovery(BaseDiscovery[TypedClassType]):
    """Discovers generic Python classes within specified packages."""

    def __init__(self):
        super().__init__()
        # TypedClass specific attributes, if any, can be initialized here.

    def _is_pydantic_model(self, obj: Any) -> bool:
        """Checks if an object is a Pydantic model."""
        return inspect.isclass(obj) and issubclass(obj, BaseModel)

    def _is_target_model(self, obj: Any) -> bool:
        """
        Check if an object is a generic class suitable for conversion.
        It must be a class, not an ABC, not a Pydantic model, and not a dataclass.
        """
        if not inspect.isclass(obj):
            return False
        if inspect.isabstract(obj):
            logger.debug(f"Skipping abstract class {obj.__name__}")
            return False
        if self._is_pydantic_model(obj):  # Check if it's a Pydantic model
            logger.debug(f"Skipping Pydantic model {obj.__name__}")
            return False
        if dataclasses.is_dataclass(obj):
            logger.debug(f"Skipping dataclass {obj.__name__}")
            return False

        # Further checks can be added here, e.g., must be in a specific list
        # or have certain characteristics. For now, this is a basic filter.
        logger.debug(f"Identified potential target typed class: {obj.__name__}")
        return True

    def _default_eligibility_filter(self, model: TypedClassType) -> bool:
        """
        Check default eligibility for generic classes.
        For example, we might want to ensure it's not an ABC, though
        _is_target_model should already catch this.
        """
        # Redundant check if _is_target_model is comprehensive, but good for safety.
        if inspect.isabstract(model):
            logger.debug(f"Filtering out typed class {model.__name__} (is abstract)")
            return False

        # Add other default checks if needed.
        # For instance, are there specific base classes (not ABCs) we want to exclude/include?
        return True

    def analyze_dependencies(self) -> None:
        """
        Build the dependency graph for the filtered generic classes.
        Dependencies are determined by type hints in __init__ arguments
        and class-level attribute annotations.
        """
        logger.info("Analyzing dependencies between filtered typed classes...")
        self.dependencies: dict[TypedClassType, set[TypedClassType]] = {}

        # Ensure all filtered models are keys in the dependencies dict
        for model_qualname in self.filtered_models:
            model_obj = self.filtered_models[model_qualname]
            self.dependencies[model_obj] = set()

        filtered_model_qualnames = set(self.filtered_models.keys())

        def _find_and_add_dependency(source_model: TypedClassType, potential_dep_type: Any):
            """
            Helper to check if a potential dependency type is a target model
            and add it to the graph.
            """
            # Check if potential_dep_type itself is a class and one of our targets
            if self._is_target_model(potential_dep_type):
                dep_qualname = f"{potential_dep_type.__module__}.{potential_dep_type.__name__}"
                if dep_qualname in filtered_model_qualnames and potential_dep_type is not source_model:
                    dep_model_obj = self.filtered_models.get(dep_qualname)
                    if dep_model_obj:
                        self.dependencies[source_model].add(dep_model_obj)
                    else:
                        logger.warning(
                            f"Inconsistency: Dependency '{dep_qualname}' for typed class "
                            f"'{source_model.__name__}' found by name but not as object in filtered set."
                        )
            # TODO: Handle generics like list[TargetType], dict[str, TargetType], Union[TargetType, None]
            # This would involve using get_origin and get_args from typing.

        for model_type in self.filtered_models.values():
            # 1. Analyze __init__ parameters
            try:
                init_signature = inspect.signature(model_type.__init__)
                for param in init_signature.parameters.values():
                    if param.name == "self" or param.annotation is inspect.Parameter.empty:
                        continue
                    _find_and_add_dependency(model_type, param.annotation)
            except (ValueError, TypeError) as e:  # Some built-ins or exotic classes might not have inspectable __init__
                logger.debug(f"Could not inspect __init__ for {model_type.__name__}: {e}")

            # 2. Analyze class-level annotations
            try:
                annotations = inspect.get_annotations(model_type, eval_str=True)
                for _, attr_type in annotations.items():
                    _find_and_add_dependency(model_type, attr_type)
            except Exception as e:
                logger.debug(f"Could not get annotations for {model_type.__name__}: {e}")

        logger.info("Typed class dependency analysis complete.")
        # Debug logging of dependencies will be handled by BaseDiscovery.log_dependencies

    def get_models_in_registration_order(self) -> list[TypedClassType]:
        """
        Return generic classes sorted topologically based on dependencies.
        This method can often be inherited from BaseDiscovery if the dependency
        graph is built correctly.
        """
        # For now, assume BaseDiscovery's implementation is sufficient.
        # If specific logic for typed classes is needed, override here.
        return super().get_models_in_registration_order()

    # _resolve_forward_refs might be needed if generic classes use string forward references
    # for their types, similar to Pydantic. For now, assume direct type hints.
    # def _resolve_forward_refs(self) -> None:
    #    logger.debug("Attempting TypedClass-specific forward ref resolution...")
    #    super()._resolve_forward_refs()
