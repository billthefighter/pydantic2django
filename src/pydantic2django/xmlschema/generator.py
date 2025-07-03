"""
XML Schema Django model generator.
Main entry point for generating Django models from XML Schema files.
"""
import logging
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from django.db import models

from pydantic2django.core.base_generator import BaseStaticGenerator
from pydantic2django.core.factories import ConversionCarrier
from pydantic2django.django.models import Xml2DjangoBaseClass

from .discovery import XmlSchemaDiscovery
from .factory import XmlSchemaFieldInfo, XmlSchemaModelFactory
from .models import XmlSchemaComplexType

logger = logging.getLogger(__name__)


class XmlSchemaDjangoModelGenerator(BaseStaticGenerator[XmlSchemaComplexType, XmlSchemaFieldInfo]):
    """
    Main class to orchestrate the generation of Django models from XML Schemas.
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
        discovery = XmlSchemaDiscovery()
        model_factory = XmlSchemaModelFactory(app_label=app_label)

        super().__init__(
            output_path=output_path,
            packages=[str(f) for f in schema_files],
            app_label=app_label,
            discovery_instance=discovery,
            model_factory_instance=model_factory,
            base_model_class=base_model_class,
            class_name_prefix=class_name_prefix,
            module_mappings=module_mappings,
            verbose=verbose,
            filter_function=filter_function,
        )

    def _get_model_definition_extra_context(self, carrier: ConversionCarrier) -> dict:
        """
        Extracts additional context required for rendering the Django model,
        including field definitions and enum classes.
        """
        field_definitions = carrier.django_field_definitions

        enum_classes = carrier.context_data.get("enums", {})

        return {
            "field_definitions": field_definitions,
            "enum_classes": enum_classes.values(),
        }

    # All rendering logic is now handled by the BaseStaticGenerator using the implemented abstract methods.
    # The custom generate and _generate_file_content methods are no longer needed.

    # --- Implement abstract methods from BaseStaticGenerator ---

    def _get_source_model_name(self, carrier: ConversionCarrier[XmlSchemaComplexType]) -> str:
        """Get the name of the original source model from the carrier."""
        return carrier.source_model.name

    def _add_source_model_import(self, carrier: ConversionCarrier[XmlSchemaComplexType]):
        """Add the necessary import for the original source model."""
        # For XML Schema, the models are generated, not imported
        pass

    def _prepare_template_context(
        self, unique_model_definitions: list[str], django_model_names: list[str], imports: dict
    ) -> dict:
        """Prepare the subclass-specific context for the main models_file.py.j2 template."""
        return {
            "model_definitions": unique_model_definitions,
            "django_model_names": django_model_names,
            "imports": imports,
            "generation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "app_label": self.app_label,
        }

    def _get_models_in_processing_order(self) -> list[XmlSchemaComplexType]:
        """Return source models in the correct processing (dependency) order."""
        return self.discovery_instance.get_models_in_registration_order()

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
            "total_schemas": len(self.discovery_instance.parsed_schemas),
            "total_complex_types": len(self.discovery_instance.all_models),
            "filtered_complex_types": len(self.discovery_instance.filtered_models),
            "generated_models": len(self.carriers),
        }

        # Add per-schema breakdown
        schema_breakdown = []
        for schema_def in self.discovery_instance.parsed_schemas:
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

        for schema_def in self.discovery_instance.parsed_schemas:
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

    def _render_choices_class(self, choices_info: dict) -> str:
        """Render a single TextChoices class."""
        class_name = choices_info["name"]
        choices = choices_info["choices"]
        lines = [f"class {class_name}(models.TextChoices):"]
        for value, label in choices:
            # Attempt to create a valid Python identifier for the member name
            member_name = re.sub(r"[^a-zA-Z0-9_]", "_", label.upper())
            if not member_name or not member_name[0].isalpha():
                member_name = f"CHOICE_{member_name}"
            lines.append(f'    {member_name} = "{value}", "{label}"')
        return "\\n".join(lines)

    def generate(self):
        """
        Main method to generate the Django models file.
        """
        logger.info(f"Starting Django model generation to {self.output_path}")

        # The base class now handles the full generation pipeline
        super().generate()

        logger.info(f"Successfully generated Django models in {self.output_path}")

    def _write_to_file(self, content: str):
        with open(self.output_path, "w") as f:
            f.write(content)
