import dataclasses
import logging
from collections.abc import Callable
from typing import Optional

from django.db import models

from pydantic2django.factory import DjangoModelFactoryCarrier

from ..base import BaseStaticGenerator
from .discovery import DataclassDiscovery, DataclassType
from .factories import DataclassModelFactory

logger = logging.getLogger(__name__)

# Define the specific FieldInfo type for dataclasses
DataclassFieldInfo = dataclasses.Field


class DataclassDjangoModelGenerator(BaseStaticGenerator[DataclassType, DataclassFieldInfo]):
    """Generates Django models.py file content from Python dataclasses."""

    def __init__(
        self,
        output_path: str,
        packages: list[str],
        app_label: str,
        filter_function: Optional[Callable[[DataclassType], bool]],
        verbose: bool,
        # Provide concrete types for discovery and model factory
        discovery_instance: DataclassDiscovery,  # Use concrete DataclassDiscovery
        model_factory_instance: DataclassModelFactory,  # Use concrete DataclassModelFactory
        module_mappings: Optional[dict[str, str]],
        base_model_class: type[models.Model],
    ):
        # Initialize the base class with all required arguments
        super().__init__(
            output_path=output_path,
            packages=packages,
            app_label=app_label,
            filter_function=filter_function,
            verbose=verbose,
            discovery_instance=discovery_instance,
            model_factory_instance=model_factory_instance,
            module_mappings=module_mappings,
            base_model_class=base_model_class,
        )
        logger.info("DataclassDjangoModelGenerator initialized.")

    # --- Implement abstract methods from BaseStaticGenerator ---

    def _get_source_model_name(self, carrier: DjangoModelFactoryCarrier) -> str:
        """Get the name of the original dataclass from the carrier."""
        if carrier.pydantic_model:
            return carrier.pydantic_model.__name__
        return "UnknownDataclass"

    def _add_source_model_import(self, carrier: DjangoModelFactoryCarrier):
        """Add the necessary import for the original dataclass."""
        if carrier.pydantic_model:
            # Use the appropriate method from ImportHandler
            self.import_handler.add_pydantic_model_import(carrier.pydantic_model)
        else:
            logger.warning("Cannot add source model import: source model missing in carrier.")

    def _prepare_template_context(self, unique_model_definitions, django_model_names, imports) -> dict:
        """Prepare the context specific to dataclasses for the main models_file.py.j2 template."""
        context = {
            "model_definitions": "\n\n".join(unique_model_definitions),
            "django_model_names": ", ".join(sorted(django_model_names)),
            "imports": imports,
            "is_dataclass_generator": True,
        }
        return context

    def _get_models_in_processing_order(self) -> list[DataclassType]:
        """Return dataclasses in dependency order using the discovery instance."""
        # Add assertion to inform type checker about the concrete discovery type
        assert isinstance(
            self.discovery, DataclassDiscovery
        ), "Discovery instance must be DataclassDiscovery for this generator"
        return self.discovery.get_models_in_registration_order()

    def _get_model_definition_extra_context(self, carrier: DjangoModelFactoryCarrier) -> dict:
        """Provide extra context specific to dataclasses for model_definition.py.j2."""
        # Removed problematic metadata access
        return {
            "is_dataclass_source": True,
            "is_pydantic_source": False,
            # Add other specific details if needed, ensuring they access carrier correctly
            # Example: "source_model_module": carrier.pydantic_model.__module__ if carrier.pydantic_model else ""
        }

    # No old methods to remove as they were already replaced in the previous step
