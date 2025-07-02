"""
XML Schema Django model generator.
Main entry point for generating Django models from XML Schema files.
"""
import logging
from collections.abc import Callable
from pathlib import Path

from django.db import models

from ..core.base_generator import BaseStaticGenerator
from ..core.factories import ConversionCarrier
from ..django.models import Xml2DjangoBaseClass
from .discovery import XmlSchemaDiscovery
from .factory import XmlSchemaFieldFactory, XmlSchemaFieldInfo, XmlSchemaModelFactory
from .models import XmlSchemaComplexType

logger = logging.getLogger(__name__)


class XmlSchemaDjangoModelGenerator(BaseStaticGenerator[XmlSchemaComplexType, XmlSchemaFieldInfo]):
    """
    Generates Django models from XML Schema files.

    This is the main entry point for XML Schema to Django model conversion.
    It follows the same pattern as StaticPydanticModelGenerator and DataclassDjangoModelGenerator.
    """

    def __init__(
        self,
        schema_files: list[str | Path],
        output_path: str = "generated_models.py",
        app_label: str = "xmlschema_app",
        filter_function: Callable[[XmlSchemaComplexType], bool] | None = None,
        verbose: bool = False,
        module_mappings: dict[str, str] | None = None,
        base_model_class: type[models.Model] = Xml2DjangoBaseClass,
        class_name_prefix: str = "",
    ):
        """
        Initialize the XML Schema generator.

        Args:
            schema_files: List of XSD file paths to parse
            output_path: Path where generated models.py will be written
            app_label: Django app label for the generated models
            filter_function: Optional function to filter which complex types to include
            verbose: Enable verbose logging
            module_mappings: Optional mapping for import remapping
            base_model_class: Base Django model class to inherit from
            class_name_prefix: Prefix for generated Django model names. Defaults to empty string for direct mapping.
        """

        # Initialize XML Schema-specific components
        self.schema_files = [Path(f) for f in schema_files]

        # Create discovery instance
        self.xmlschema_discovery = XmlSchemaDiscovery(self.schema_files)

        # Create field and model factories
        from ..core.relationships import RelationshipConversionAccessor

        self.relationship_accessor = RelationshipConversionAccessor()

        self.xmlschema_field_factory = XmlSchemaFieldFactory(relationship_accessor=self.relationship_accessor)

        self.xmlschema_model_factory = XmlSchemaModelFactory(
            field_factory=self.xmlschema_field_factory,
            relationship_accessor=self.relationship_accessor,
        )

        # Call parent constructor with packages as schema file paths
        super().__init__(
            output_path=output_path,
            packages=[str(f) for f in self.schema_files],  # Pass schema files as "packages"
            app_label=app_label,
            filter_function=filter_function,
            verbose=verbose,
            discovery_instance=self.xmlschema_discovery,
            model_factory_instance=self.xmlschema_model_factory,
            module_mappings=module_mappings,
            base_model_class=base_model_class,
            class_name_prefix=class_name_prefix,
        )

        logger.info(f"XmlSchemaDjangoModelGenerator initialized with {len(self.schema_files)} schema files")

    # --- Implement abstract methods from BaseStaticGenerator ---

    def _get_source_model_name(self, carrier: ConversionCarrier[XmlSchemaComplexType]) -> str:
        """Get the name of the original XML Schema complex type."""
        return carrier.source_model.name if carrier.source_model else "UnknownComplexType"

    def _add_source_model_import(self, carrier: ConversionCarrier[XmlSchemaComplexType]):
        """Add import for the original XML Schema (not applicable, but required by interface)."""
        # XML Schema doesn't have Python imports, but we could add comments
        # about the source schema file
        if carrier.source_model and carrier.source_model.schema_location:
            # Could add as a comment in the generated file
            pass

    def _prepare_template_context(
        self, unique_model_definitions: list[str], django_model_names: list[str], imports: dict
    ) -> dict:
        """Prepare template context for XML Schema models."""
        return {
            "model_definitions": unique_model_definitions,
            "django_model_names": django_model_names,
            "imports": imports,
            "generation_source_type": "xmlschema",
            "context_definitions": [],  # XML Schema doesn't use context classes
            "context_class_names": [],
            "model_has_context": {},
            "schema_files": [str(f) for f in self.schema_files],  # Add schema file info
        }

    def _get_models_in_processing_order(self) -> list[XmlSchemaComplexType]:
        """Return XML Schema complex types in dependency order."""
        return self.xmlschema_discovery.get_models_in_registration_order()

    def _get_model_definition_extra_context(self, carrier: ConversionCarrier[XmlSchemaComplexType]) -> dict:
        """Provide extra context for XML Schema model template."""
        context = {
            "field_definitions": carrier.django_field_definitions,
        }

        if carrier.source_model:
            # Add XML Schema specific context
            context.update(
                {
                    "xml_namespace": carrier.source_model.namespace,
                    "xml_documentation": carrier.source_model.documentation,
                    "schema_location": carrier.source_model.schema_location,
                    "is_mixed_content": carrier.source_model.mixed,
                    "content_model": "choice" if carrier.source_model.choice else "sequence",
                }
            )

        return context

    # --- Additional XML Schema specific methods ---

    def generate_models_with_xml_metadata(self) -> str:
        """
        Generate Django models with additional XML metadata.

        This method extends the base generate() to add XML-specific
        comments and metadata to the generated models.
        """
        content = self.generate_models_file()

        # Add XML Schema file references as comments at the top
        schema_files_comment = "\n".join(
            [f"# Generated from XML Schema: {schema_file}" for schema_file in self.schema_files]
        )

        # Insert after the initial comments
        lines = content.split("\n")
        insert_index = 0
        for i, line in enumerate(lines):
            if line.startswith('"""') and '"""' in line[3:]:  # Single line docstring
                insert_index = i + 1
                break
            elif line.startswith('"""'):  # Multi-line docstring start
                for j in range(i + 1, len(lines)):
                    if '"""' in lines[j]:
                        insert_index = j + 1
                        break
                break

        lines.insert(insert_index, schema_files_comment)
        lines.insert(insert_index + 1, "")

        return "\n".join(lines)

    def get_schema_statistics(self) -> dict:
        """Get statistics about the parsed schemas."""
        stats = {
            "total_schemas": len(self.xmlschema_discovery.parsed_schemas),
            "total_complex_types": len(self.xmlschema_discovery.all_models),
            "filtered_complex_types": len(self.xmlschema_discovery.filtered_models),
            "generated_models": len(self.carriers),
        }

        # Add per-schema breakdown
        schema_breakdown = []
        for schema_def in self.xmlschema_discovery.parsed_schemas:
            schema_breakdown.append(
                {
                    "schema_location": schema_def.schema_location,
                    "target_namespace": schema_def.target_namespace,
                    "complex_types": len(schema_def.complex_types),
                    "simple_types": len(schema_def.simple_types),
                    "elements": len(schema_def.elements),
                }
            )

        stats["schema_breakdown"] = schema_breakdown
        return stats

    def validate_schemas(self) -> list[str]:
        """
        Validate the parsed schemas and return any warnings or errors.

        Returns:
            List of validation messages
        """
        messages = []

        for schema_def in self.xmlschema_discovery.parsed_schemas:
            # Check for common issues
            if not schema_def.target_namespace:
                messages.append(f"Schema {schema_def.schema_location} has no target namespace")

            # Check for name conflicts
            all_names = set()
            for complex_type in schema_def.complex_types.values():
                if complex_type.name in all_names:
                    messages.append(f"Duplicate type name: {complex_type.name}")
                all_names.add(complex_type.name)

        return messages

    @classmethod
    def from_schema_files(cls, schema_files: list[str | Path], **kwargs) -> "XmlSchemaDjangoModelGenerator":
        """
        Convenience class method to create generator from schema files.

        Args:
            schema_files: List of XSD file paths
            **kwargs: Additional arguments passed to __init__

        Returns:
            Configured XmlSchemaDjangoModelGenerator instance
        """
        return cls(schema_files=schema_files, **kwargs)
