"""
XML Schema discovery module.
Discovers XML Schema complex types from XSD files and prepares them for Django model generation.
"""
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core.discovery import BaseDiscovery
from .models import XmlSchemaComplexType, XmlSchemaDefinition
from .parser import XmlSchemaParseError, XmlSchemaParser

logger = logging.getLogger(__name__)


class XmlSchemaDiscovery(BaseDiscovery[XmlSchemaComplexType]):
    """Discovers XML Schema complex types from XSD files."""

    def __init__(self, schema_files: list[str | Path] | None = None):
        super().__init__()
        self.schema_files = [Path(f) for f in schema_files] if schema_files else []
        self.schema_parser = XmlSchemaParser()
        self.parsed_schemas: list[XmlSchemaDefinition] = []

    def _is_target_model(self, obj: Any) -> bool:
        """Check if object is a suitable XmlSchemaComplexType for conversion."""
        return (
            isinstance(obj, XmlSchemaComplexType)
            and not obj.abstract
            and (len(obj.elements) > 0 or len(obj.attributes) > 0)
        )

    def _default_eligibility_filter(self, model: XmlSchemaComplexType) -> bool:
        """Apply default filtering logic for XML Schema complex types."""
        if model.abstract:
            return False

        if len(model.elements) == 0 and len(model.attributes) == 0:
            return False

        return True

    def discover_models(
        self,
        packages: list[str],  # Schema file paths
        app_label: str,
        user_filters: Callable[[XmlSchemaComplexType], bool]
        | list[Callable[[XmlSchemaComplexType], bool]]
        | None = None,
    ) -> None:
        """Discover XML Schema complex types from XSD files."""
        logger.info(f"Starting XML Schema discovery for files: {packages}")

        self.all_models = {}
        self.filtered_models = {}
        self.dependencies = {}
        self.parsed_schemas = []

        schema_files = packages if packages else [str(f) for f in self.schema_files]

        # Parse all schema files
        for schema_file in schema_files:
            try:
                schema_def = self.schema_parser.parse_schema_file(schema_file)
                self.parsed_schemas.append(schema_def)

                for complex_type in schema_def.get_all_complex_types():
                    namespace = complex_type.namespace or "default"
                    qualname = f"{namespace}.{complex_type.name}"
                    self.all_models[qualname] = complex_type

            except XmlSchemaParseError as e:
                logger.error(f"Failed to parse schema file {schema_file}: {e}")
                continue

        logger.info(f"Discovered {len(self.all_models)} total complex types")

        # Apply filters
        eligible_models = {}
        for qualname, model in self.all_models.items():
            if self._is_target_model(model) and self._default_eligibility_filter(model):
                eligible_models[qualname] = model

        # Apply user filters
        if user_filters:
            filters = user_filters if isinstance(user_filters, list) else [user_filters]
            for filter_func in filters:
                filtered_models = {}
                for qualname, model in eligible_models.items():
                    try:
                        if filter_func(model):
                            filtered_models[qualname] = model
                    except Exception as e:
                        logger.error(f"Error applying filter to {qualname}: {e}")
                        filtered_models[qualname] = model
                eligible_models = filtered_models

        self.filtered_models = eligible_models
        logger.info(f"XML Schema discovery complete. {len(self.filtered_models)} models after filtering")

    def analyze_dependencies(self) -> None:
        """Analyze dependencies between XML Schema complex types."""
        logger.info("Analyzing XML Schema dependencies...")

        self.dependencies = {}
        for model in self.filtered_models.values():
            self.dependencies[model] = set()

        # Build type name mapping
        type_name_to_model = {}
        for model in self.filtered_models.values():
            type_name_to_model[model.name] = model
            if model.namespace:
                type_name_to_model[f"{model.namespace}.{model.name}"] = model

        # Find dependencies
        for model in self.filtered_models.values():
            dependencies = set()

            # Base type inheritance
            if model.base_type:
                base_model = type_name_to_model.get(model.base_type)
                if base_model and base_model is not model:
                    dependencies.add(base_model)

            # Element type references
            for element in model.elements:
                if element.type_name:
                    type_name = element.type_name.split(":")[-1]  # Remove namespace
                    referenced_model = type_name_to_model.get(type_name)
                    if referenced_model and referenced_model is not model:
                        dependencies.add(referenced_model)

            # Attribute type references
            for attribute in model.attributes.values():
                if attribute.type_name:
                    type_name = attribute.type_name.split(":")[-1]
                    referenced_model = type_name_to_model.get(type_name)
                    if referenced_model and referenced_model is not model:
                        dependencies.add(referenced_model)

            self.dependencies[model] = dependencies

        logger.info("XML Schema dependency analysis complete")

    def get_models_in_registration_order(self) -> list[XmlSchemaComplexType]:
        """Return complex types sorted topologically based on dependencies."""
        if not self.dependencies:
            return list(self.filtered_models.values())

        sorted_models = []
        visited = set()
        visiting = set()

        def visit(model: XmlSchemaComplexType):
            if model in visited:
                return
            if model in visiting:
                logger.error(f"Circular dependency detected: {model.name}")
                return

            visiting.add(model)
            for dep in self.dependencies.get(model, set()):
                if dep in self.filtered_models.values():
                    visit(dep)

            visiting.remove(model)
            visited.add(model)
            sorted_models.append(model)

        for model in self.filtered_models.values():
            if model not in visited:
                visit(model)

        logger.info(f"Models sorted for registration: {[m.name for m in sorted_models]}")
        return sorted_models
