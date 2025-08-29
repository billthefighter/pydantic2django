import abc
import importlib
import inspect
import logging
import pkgutil
from collections.abc import Callable
from typing import Any, Generic, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel")  # Type variable for the source model (Pydantic or Dataclass)
TFieldInfo = TypeVar("TFieldInfo")  # Type variable for field info (Pydantic FieldInfo or Dataclass Field)


# --- Base Discovery ---
class BaseDiscovery(abc.ABC, Generic[TModel]):
    """Abstract base class for discovering models (e.g., Pydantic, Dataclasses)."""

    def __init__(self):
        self.all_models: dict[str, type[TModel]] = {}  # All discovered models before any filtering
        self.filtered_models: dict[str, type[TModel]] = {}  # Models after all filters
        self.dependencies: dict[type[TModel], set[type[TModel]]] = {}  # Dependencies between filtered models

    @abc.abstractmethod
    def _is_target_model(self, obj: Any) -> bool:
        """Check if an object is the type of model this discovery class handles."""
        pass

    @abc.abstractmethod
    def _default_eligibility_filter(self, model: type[TModel]) -> bool:
        """
        Apply default filtering logic inherent to the model type (e.g., exclude abstract classes).
        Return True if the model is eligible, False otherwise.
        """
        pass

    def discover_models(
        self,
        packages: list[str],
        app_label: str,  # Keep for potential use in filters or subclasses
        user_filters: Optional[Union[Callable[[type[TModel]], bool], list[Callable[[type[TModel]], bool]]]] = None,
    ) -> None:
        """Discover target models in the specified packages, applying default and user filters."""
        self.all_models = {}
        self.filtered_models = {}
        self.dependencies = {}

        # Normalize user_filters to always be a list
        if user_filters is None:
            filters = []
        elif isinstance(user_filters, list):
            filters = user_filters
        else:  # It's a single callable
            filters = [user_filters]

        model_type_name = getattr(self, "__name__", "TargetModel")  # Get class name for logging

        logger.info(f"Starting {model_type_name} discovery in packages: {packages}")
        for package_name in packages:
            try:
                package = importlib.import_module(package_name)
                logger.debug(f"Scanning package: {package_name}")

                for _importer, modname, _ispkg in pkgutil.walk_packages(
                    path=package.__path__ if hasattr(package, "__path__") else None,
                    prefix=package.__name__ + ".",
                    onerror=lambda name: logger.warning(f"Error accessing module {name}"),
                ):
                    try:
                        module = importlib.import_module(modname)
                        for name, obj in inspect.getmembers(module):
                            # Use the subclass implementation to check if it's the right model type
                            if self._is_target_model(obj):
                                model_qualname = f"{modname}.{name}"
                                if model_qualname not in self.all_models:
                                    self.all_models[model_qualname] = obj
                                    logger.debug(f"Discovered potential {model_type_name}: {model_qualname}")

                                    # Apply filters sequentially using subclass implementation
                                    is_eligible = self._default_eligibility_filter(obj)
                                    if is_eligible:
                                        for user_filter in filters:
                                            try:
                                                if not user_filter(obj):
                                                    is_eligible = False
                                                    logger.debug(
                                                        f"Filtered out {model_type_name} by user filter: {model_qualname}"
                                                    )
                                                    break  # No need to check other filters
                                            except Exception as filter_exc:
                                                # Attempt to get filter name, default to repr
                                                filter_name = getattr(user_filter, "__name__", repr(user_filter))
                                                logger.error(
                                                    f"Error applying user filter {filter_name} to {model_qualname}: {filter_exc}",
                                                    exc_info=True,
                                                )
                                                is_eligible = False  # Exclude on filter error
                                                break

                                    if is_eligible:
                                        self.filtered_models[model_qualname] = obj
                                        logger.debug(f"Added eligible {model_type_name}: {model_qualname}")

                    except ImportError as e:
                        logger.warning(f"Could not import module {modname}: {e}")
                    except Exception as e:
                        logger.error(f"Error inspecting module {modname} for {model_type_name}s: {e}", exc_info=True)

            except ImportError:
                logger.error(f"Package {package_name} not found.")
            except Exception as e:
                logger.error(f"Error discovering {model_type_name}s in package {package_name}: {e}", exc_info=True)

        logger.info(
            f"{model_type_name} discovery complete. Found {len(self.all_models)} total models, {len(self.filtered_models)} after filtering."
        )

        # Hooks for subclass-specific post-processing if needed
        self._post_discovery_hook()

        # Resolve forward references if applicable (might be subclass specific)
        self._resolve_forward_refs()

        # Build dependency graph for filtered models
        self.analyze_dependencies()

    @abc.abstractmethod
    def analyze_dependencies(self) -> None:
        """Analyze dependencies between the filtered models."""
        pass

    @abc.abstractmethod
    def get_models_in_registration_order(self) -> list[type[TModel]]:
        """Return filtered models sorted topologically based on dependencies."""
        pass

    # Optional hook for subclasses to run code after discovery loop but before analyze
    def _post_discovery_hook(self) -> None:
        pass

    # Keep placeholder, subclasses might override if needed
    def _resolve_forward_refs(self) -> None:
        """Placeholder for resolving forward references if needed."""
        logger.debug("Base _resolve_forward_refs called (if applicable).")
        pass
