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
from pydantic2django.django.timescale.bases import XmlTimescaleBase
from pydantic2django.django.timescale.heuristics import (
    TimescaleConfig,
    TimescaleRole,
    classify_xml_complex_types,
    has_direct_time_feature,
    should_use_timescale_base,
)

from . import register_generated_model
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
        class_name_prefix: str = "",
        # Relationship handling for nested complex types
        nested_relationship_strategy: str = "fk",  # one of: "fk", "json", "auto"
        list_relationship_style: str = "child_fk",  # one of: "child_fk", "m2m", "json"
        nesting_depth_threshold: int = 1,
        # Optional override for the Django base model class
        base_model_class: type[models.Model] | None = None,
        # Feature flags
        enable_timescale: bool = True,
        # Timescale configuration
        timescale_overrides: dict[str, TimescaleRole] | None = None,
        timescale_config: TimescaleConfig | None = None,
        timescale_strict: bool = False,
        # Control behavior for missing leaf/child targets in finalize pass
        auto_generate_missing_leaves: bool = False,
        # --- GFK flags ---
        enable_gfk: bool = True,
        gfk_policy: str | None = "threshold_by_children",
        gfk_threshold_children: int | None = 8,
        gfk_value_mode: str | None = "typed_columns",
        gfk_normalize_common_attrs: bool = False,
        gfk_overrides: dict[str, bool] | None = None,
    ):
        """Create a XML Schema → Django generator with optional Generic Entries mode.

        GFK (Generic Entries) flags:
        - enable_gfk: Enable Generic Entries mode. When True and policy matches, parents
          gain `entries = GenericRelation('GenericEntry')` and concrete children are not emitted.
        - gfk_policy: Controls which nested elements are routed to GenericEntry.
          Values: "substitution_only" | "repeating_only" | "all_nested" | "threshold_by_children".
        - gfk_threshold_children: Used only with "threshold_by_children"; minimum distinct child
          complex types inside a wrapper-like container to activate GFK.
        - gfk_value_mode: "json_only" stores text under attrs_json['value']; "typed_columns"
          extracts text into text_value, num_value, time_value when unambiguous.
        - gfk_normalize_common_attrs: Reserved (default False) for promoting frequently used
          attributes (e.g., timestamp) into normalized columns when using typed columns.
        - gfk_overrides: Optional per-element overrides by local element name;
          True forces GFK, False disables it even if the policy would apply.
        """
        discovery = XmlSchemaDiscovery()
        model_factory = XmlSchemaModelFactory(
            app_label=app_label,
            nested_relationship_strategy=nested_relationship_strategy,
            list_relationship_style=list_relationship_style,
            nesting_depth_threshold=nesting_depth_threshold,
            enable_gfk=enable_gfk,
            gfk_policy=gfk_policy,
            gfk_threshold_children=gfk_threshold_children,
            gfk_overrides=gfk_overrides,
        )

        super().__init__(
            output_path=output_path,
            packages=[str(f) for f in schema_files],
            app_label=app_label,
            discovery_instance=discovery,
            model_factory_instance=model_factory,
            base_model_class=base_model_class or self._get_default_base_model_class(),
            class_name_prefix=class_name_prefix,
            module_mappings=module_mappings,
            verbose=verbose,
            filter_function=filter_function,
            enable_timescale=enable_timescale,
            enable_gfk=enable_gfk,
            gfk_policy=gfk_policy,
            gfk_threshold_children=gfk_threshold_children,
            gfk_value_mode=gfk_value_mode,
            gfk_normalize_common_attrs=gfk_normalize_common_attrs,
        )
        # Timescale classification results cached per run
        self._timescale_roles: dict[str, TimescaleRole] = {}
        self._timescale_overrides: dict[str, TimescaleRole] | None = timescale_overrides
        self._timescale_config: TimescaleConfig | None = timescale_config
        self._timescale_strict: bool = timescale_strict
        self._auto_generate_missing_leaves: bool = auto_generate_missing_leaves

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
            # Extra Meta emission (e.g., indexes)
            "meta_indexes": carrier.context_data.get("meta_indexes", []),
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
            # Flattened import categories for templates
            "pydantic_imports": sorted(imports.get("pydantic", [])),
            "context_imports": sorted(imports.get("context", [])),
            "django_imports": sorted(imports.get("django", [])),
            "general_imports": sorted(imports.get("general", [])),
            # Also pass the structured imports dict if templates need it
            "imports": imports,
            "generation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "app_label": self.app_label,
            "generation_source_type": "xmlschema",
        }

    def _get_default_base_model_class(self) -> type[models.Model]:
        """Return the default Django base model for XML Schema conversion."""
        # Prefer local pydantic2django base; fall back to typed2django if present
        try:
            from pydantic2django.django.models import Xml2DjangoBaseClass as _Base

            return _Base
        except Exception:
            pass
        try:
            from typed2django.django.models import (
                Xml2DjangoBaseClass as _Base,  # type: ignore[import-not-found]
            )

            return _Base
        except Exception as exc:  # pragma: no cover - defensive
            raise ImportError(
                "pydantic2django.django.models.Xml2DjangoBaseClass (or typed2django equivalent) is required for XML Schema generation."
            ) from exc

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

    def generate_models_file(self) -> str:
        """
        Override to allow relationship finalization after carriers are built
        but before rendering templates.
        """
        # Discover and create carriers first via base implementation pieces
        self.discover_models()
        # --- GFK Prepass: when enabled, identify wrapper owners and exclude their polymorphic children ---
        if getattr(self, "enable_gfk", False) and getattr(self, "gfk_policy", None):
            try:
                excluded_child_types: set[str] = set()
                gfk_owner_wrappers: set[str] = set()
                policy = str(getattr(self, "gfk_policy", ""))
                threshold_val = int(getattr(self, "gfk_threshold_children", 0) or 0)
                # Walk parsed schemas to infer wrapper owners and their child types
                for schema_def in getattr(self.discovery_instance, "parsed_schemas", []) or []:
                    try:
                        for ct in schema_def.get_all_complex_types():
                            # Iterate elements of each complex type
                            for el in ct.elements:
                                # Only interested in complex targets
                                if not el.type_name or el.base_type is not None:
                                    continue
                                target_type = el.type_name.split(":")[-1]
                                target_ct = schema_def.complex_types.get(target_type)
                                if not target_ct:
                                    continue
                                # Wrapper-like heuristic (may be used by providers; structural checks take precedence)
                                _is_wrapper_like = bool(el.name[:1].isupper()) or str(target_type).endswith(
                                    "WrapperType"
                                )
                                # Collect distinct child complex types under target wrapper
                                distinct_child_types: set[str] = set()
                                has_substitution_members = False
                                repeating_leaf_child: str | None = None
                                for child_el in target_ct.elements:
                                    if getattr(child_el, "substitution_group", None):
                                        has_substitution_members = True
                                    if getattr(child_el, "type_name", None):
                                        cand = child_el.type_name.split(":")[-1]
                                        if cand:
                                            distinct_child_types.add(cand)
                                    if getattr(child_el, "is_list", False) and getattr(child_el, "type_name", None):
                                        repeating_leaf_child = child_el.type_name.split(":")[-1]

                                # Decide based on policy (favor generic structural checks over name heuristics)
                                route_by_subst = policy == "substitution_only" and has_substitution_members
                                route_by_all_nested = policy == "all_nested"
                                route_by_threshold = policy == "threshold_by_children" and (
                                    threshold_val <= 0 or len(distinct_child_types) >= threshold_val
                                )
                                route_by_repeating = policy == "repeating_only" and bool(el.is_list)

                                if route_by_subst or route_by_all_nested or route_by_threshold or route_by_repeating:
                                    # Mark the CURRENT complex type (container) as the owner so placeholders on it are suppressed
                                    try:
                                        owner_name = getattr(ct, "name", None) or getattr(ct, "__name__", str(ct))
                                    except Exception:
                                        owner_name = target_type
                                    gfk_owner_wrappers.add(owner_name)
                                    try:
                                        if getattr(self, "verbose", False):
                                            reason = (
                                                "substitution_only"
                                                if route_by_subst
                                                else (
                                                    "all_nested"
                                                    if route_by_all_nested
                                                    else (
                                                        "threshold_by_children"
                                                        if route_by_threshold
                                                        else "repeating_only"
                                                    )
                                                )
                                            )
                                            logger.info(
                                                "[GFK][prepass] owner=%s element=%s policy=%s distinct_children=%d",
                                                owner_name,
                                                el.name,
                                                reason,
                                                len(distinct_child_types),
                                            )
                                    except Exception:
                                        pass
                                    # Exclude polymorphic children under this wrapper from concrete generation
                                    if el.is_list and repeating_leaf_child:
                                        excluded_child_types.add(repeating_leaf_child)
                                    # Also exclude distinct child complex types observed under the wrapper
                                    excluded_child_types.update({t for t in distinct_child_types if t})
                    except Exception:
                        continue  # best-effort per schema

                # Filter discovery output to remove excluded children, keeping wrappers/parents
                try:
                    # filtered_models is a mapping of qualname -> complex_type
                    fm = dict(getattr(self.discovery_instance, "filtered_models", {}) or {})
                    if fm and excluded_child_types:
                        new_fm = {k: v for k, v in fm.items() if getattr(v, "name", None) not in excluded_child_types}
                        self.discovery_instance.filtered_models = new_fm  # type: ignore[attr-defined]
                        try:
                            if getattr(self, "verbose", False):
                                logger.info(
                                    "[GFK][prepass] excluded_children_count=%d (examples: %s)",
                                    len(excluded_child_types),
                                    ", ".join(sorted(excluded_child_types)[:10]),
                                )
                        except Exception:
                            pass
                except Exception:
                    pass

                # Share exclusions and owner marks with the field factory for intra-model decisions
                try:
                    ff = getattr(self.model_factory_instance, "field_factory", None)
                    if ff is not None:
                        ff._gfk_excluded_child_types = excluded_child_types
                        # Seed owner names to suppress JSON placeholders for wrapper children
                        owner_names = set(getattr(ff, "_gfk_owner_names", set()))
                        owner_names.update(gfk_owner_wrappers)
                        ff._gfk_owner_names = owner_names
                except Exception:
                    pass
            except Exception:
                # Prepass is best-effort and should never crash generation
                pass
        # Conflict guard: if GFK owners detected but JSON relationship styles are requested, fail fast.
        try:
            if getattr(self, "enable_gfk", False):
                ff = getattr(self.model_factory_instance, "field_factory", None)
                owners = set(getattr(ff, "_gfk_owner_names", set()) or set())
                list_style = getattr(ff, "list_relationship_style", "child_fk")
                nested_style = getattr(ff, "nested_relationship_strategy", "fk")
                if owners and (list_style == "json" or nested_style == "json"):
                    raise ValueError(
                        "GFK mode active with owners detected, but JSON relationship strategy was requested. "
                        "This configuration would produce dual emission. Adjust flags or disable JSON strategy."
                    )
        except Exception as _e:
            if isinstance(_e, ValueError):
                raise
        # Inform the field factory which models are actually included so it can
        # avoid generating relations to filtered-out models (fallback to JSON).
        try:
            # Prefer local (unqualified) complex type names to match factory lookups
            # filtered_models keys may be qualified as "{namespace}.{name}", so derive plain names
            included_local_names = {
                getattr(m, "name", str(m))
                for m in self.discovery_instance.filtered_models.values()  # type: ignore[attr-defined]
            }
            # Also include any already-qualified keys for maximum compatibility
            included_qualified_keys = set(self.discovery_instance.filtered_models.keys())  # type: ignore[attr-defined]
            included_names = included_local_names | included_qualified_keys
            # type: ignore[attr-defined]
            self.model_factory_instance.field_factory.included_model_names = included_names  # noqa: E501
            # Also pass through any GFK prepass exclusions to the field factory if not already set
            try:
                ff = self.model_factory_instance.field_factory
                if not getattr(ff, "_gfk_excluded_child_types", None):
                    ff._gfk_excluded_child_types = set()
            except Exception:
                pass
        except Exception:
            pass
        models_to_process = self._get_models_in_processing_order()

        # Classify models for Timescale usage (hypertable vs dimension)
        if self.enable_timescale:
            try:
                self._timescale_roles = classify_xml_complex_types(
                    models_to_process,
                    overrides=self._timescale_overrides,
                    config=self._timescale_config,
                )
            except Exception:
                self._timescale_roles = {}
        else:
            self._timescale_roles = {}

        # Strict validation: if any hypertable lacks a direct time-like field, fail fast
        if self.enable_timescale and self._timescale_strict:
            for m in models_to_process:
                name = getattr(m, "name", getattr(m, "__name__", str(m)))
                if self._timescale_roles.get(name) == TimescaleRole.HYPERTABLE and not has_direct_time_feature(m):
                    raise ValueError(
                        f"Timescale strict mode: '{name}' classified as hypertable but has no direct time-like field. "
                        f"Add a time/timestamp-like attribute/element or demote via overrides."
                    )

        # Reset state for this run (mirror BaseStaticGenerator)
        self.carriers = []
        self.import_handler.extra_type_imports.clear()
        self.import_handler.pydantic_imports.clear()
        self.import_handler.context_class_imports.clear()
        self.import_handler.imported_names.clear()
        self.import_handler.processed_field_types.clear()
        self.import_handler._add_type_import(self.base_model_class)

        for source_model in models_to_process:
            self.setup_django_model(source_model)

        # Give the factory a chance to add cross-model relationship fields (e.g., child FKs)
        carriers_by_name = {
            (getattr(c.source_model, "name", None) or getattr(c.source_model, "__name__", "")): c
            for c in self.carriers
            if c.django_model is not None
        }
        if hasattr(self.model_factory_instance, "finalize_relationships"):
            try:
                # type: ignore[attr-defined]
                self.model_factory_instance.finalize_relationships(carriers_by_name, self.app_label)  # noqa: E501
            except Exception as e:
                logger.error(f"Error finalizing XML relationships: {e}")

        # Inject GenericRelation on parents when GFK is enabled and pending children exist
        gfk_used = False
        try:
            for carrier in self.carriers:
                if getattr(carrier, "enable_gfk", False) and getattr(carrier, "pending_gfk_children", None):
                    # Ensure contenttypes imports
                    self.import_handler.add_import("django.contrib.contenttypes.fields", "GenericRelation")
                    self.import_handler.add_import("django.contrib.contenttypes.fields", "GenericForeignKey")
                    self.import_handler.add_import("django.contrib.contenttypes.models", "ContentType")
                    # Add reverse relation field on parent model
                    carrier.django_field_definitions[
                        "entries"
                    ] = "GenericRelation('GenericEntry', related_query_name='entries')"
                    gfk_used = True
        except Exception:
            pass

        # Optional diagnostics: summarize GFK routing per parent
        try:
            if gfk_used and getattr(self, "verbose", False):
                summary: dict[str, int] = {}
                for carrier in self.carriers:
                    for entry in getattr(carrier, "pending_gfk_children", []) or []:
                        owner = str(entry.get("owner") or getattr(carrier.source_model, "name", ""))
                        summary[owner] = summary.get(owner, 0) + 1
                if summary:
                    try:
                        details = ", ".join(f"{k}:{v}" for k, v in sorted(summary.items()))
                        logger.info(f"[GFK] Routed children counts by owner: {details}")
                    except Exception:
                        logger.info("[GFK] Routed children counts by owner available")
        except Exception:
            pass

        # Register generated model classes for in-memory lookup by ingestors
        try:
            for carrier in self.carriers:
                if carrier.django_model is not None:
                    register_generated_model(self.app_label, carrier.django_model)
                    try:
                        logger.info(
                            "Registered generated model %s (abstract=%s) for app '%s'",
                            carrier.django_model.__name__,
                            getattr(getattr(carrier.django_model, "_meta", None), "abstract", None),
                            self.app_label,
                        )
                    except Exception:
                        pass
                    # Prevent dynamic classes from polluting Django's global app registry
                    try:
                        from django.apps import (
                            apps as django_apps,  # Local import to avoid hard dependency
                        )

                        model_lower = getattr(getattr(carrier.django_model, "_meta", None), "model_name", None)
                        if model_lower:
                            django_apps.all_models.get(self.app_label, {}).pop(model_lower, None)
                    except Exception:
                        # Best-effort cleanup only
                        pass
        except Exception as e:
            logger.debug(f"Non-fatal: failed to register generated models for app '{self.app_label}': {e}")

        # Proceed with standard definition rendering
        model_definitions = []
        django_model_names = []
        for carrier in self.carriers:
            if carrier.django_model:
                try:
                    model_def = self.generate_model_definition(carrier)
                    if model_def:
                        model_definitions.append(model_def)
                        django_model_names.append(f"'{self._clean_generic_type(carrier.django_model.__name__)}'")
                except Exception as e:
                    logger.error(
                        f"Error generating definition for source model {getattr(carrier.source_model, '__name__', '?')}: {e}",
                        exc_info=True,
                    )

        # If GFK used, append GenericEntry model definition and include in __all__
        if gfk_used:
            model_definitions.append(self._build_generic_entry_model_definition())
            django_model_names.append("'GenericEntry'")

        unique_model_definitions = self._deduplicate_definitions(model_definitions)
        # Ensure that every advertised model name in __all__ has a corresponding class definition.
        try:
            import re as _re

            joined_defs = "\n".join(unique_model_definitions)
            existing_names = {m.group(1) for m in _re.finditer(r"^\s*class\s+(\w+)\(", joined_defs, _re.MULTILINE)}
            advertised_names = [
                self._clean_generic_type(getattr(c.django_model, "__name__", ""))
                for c in self.carriers
                if c.django_model is not None
            ]
            for name in advertised_names:
                if name and name not in existing_names:
                    minimal_def = (
                        f'"""\nDjango model for {name}.\n"""\n\n'
                        f"class {name}({self.base_model_class.__name__}):\n    # No fields defined for this model.\n    pass\n\n    class Meta:\n        app_label = '{self.app_label}'\n        abstract = False\n\n"
                    )
                    unique_model_definitions.append(minimal_def)
        except Exception:
            pass

        imports = self.import_handler.deduplicate_imports()
        template_context = self._prepare_template_context(unique_model_definitions, django_model_names, imports)
        template_context.update(
            {
                "generation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "base_model_module": self.base_model_class.__module__,
                "base_model_name": self.base_model_class.__name__,
                "extra_type_imports": sorted(self.import_handler.extra_type_imports),
            }
        )
        # Detect validator usage to ensure imports are present
        used_validators: set[str] = set()
        # Primary source: field factory tracking
        try:
            field_factory = getattr(self.model_factory_instance, "field_factory", None)
            factory_used = getattr(field_factory, "used_validators", None)
            if isinstance(factory_used, set):
                used_validators.update(factory_used)
        except Exception:
            pass
        # Fallback: scan rendered definitions
        try:
            if not used_validators:
                validator_symbols = [
                    "RegexValidator",
                    "MinValueValidator",
                    "MaxValueValidator",
                    "URLValidator",
                    "EmailValidator",
                    "validate_slug",
                    "validate_uuid4",
                    "validate_email",
                    "validate_ipv46_address",
                    "validate_ipv4_address",
                    "validate_ipv6_address",
                    "validate_unicode_slug",
                ]
                joined_defs = "\n".join(unique_model_definitions)
                for symbol in validator_symbols:
                    if re.search(rf"\\b{re.escape(symbol)}\\b", joined_defs):
                        used_validators.add(symbol)
        except Exception:
            pass
        template_context["validator_imports"] = sorted(used_validators)
        template = self.jinja_env.get_template("models_file.py.j2")
        return template.render(**template_context)

    def _build_generic_entry_model_definition(self) -> str:
        """Build the GenericEntry model definition string for XMLSchema generation."""
        self.import_handler.add_import("django.contrib.contenttypes.fields", "GenericForeignKey")
        self.import_handler.add_import("django.contrib.contenttypes.fields", "GenericRelation")
        self.import_handler.add_import("django.contrib.contenttypes.models", "ContentType")

        fields: list[str] = []
        fields.append("content_type = models.ForeignKey('contenttypes.ContentType', on_delete=models.CASCADE)")
        fields.append("object_id = models.PositiveIntegerField()")
        fields.append("content_object = GenericForeignKey('content_type', 'object_id')")
        fields.append("element_qname = models.CharField(max_length=255)")
        fields.append("type_qname = models.CharField(max_length=255, null=True, blank=True)")
        fields.append("attrs_json = models.JSONField(default=dict, blank=True)")
        if getattr(self, "gfk_value_mode", None) == "typed_columns":
            fields.append("text_value = models.TextField(null=True, blank=True)")
            fields.append("num_value = models.DecimalField(max_digits=20, decimal_places=6, null=True, blank=True)")
            fields.append("time_value = models.DateTimeField(null=True, blank=True)")
        fields.append("order_index = models.IntegerField(default=0)")
        fields.append("path_hint = models.CharField(max_length=255, null=True, blank=True)")

        indexes_lines = ["models.Index(fields=['content_type', 'object_id'])"]
        # Optional indexes when typed value columns are enabled
        if getattr(self, "gfk_value_mode", None) == "typed_columns":
            indexes_lines.append("models.Index(fields=['element_qname'])")
            indexes_lines.append("models.Index(fields=['type_qname'])")
            indexes_lines.append("models.Index(fields=['time_value'])")
            indexes_lines.append("models.Index(fields=['content_type', 'object_id', '-time_value'])")

        lines: list[str] = []
        lines.append(f"class GenericEntry({self.base_model_class.__name__}):")
        for f in fields:
            lines.append(f"    {f}")
        lines.append("")
        lines.append("    class Meta:")
        lines.append(f"        app_label = '{self.app_label}'")
        lines.append("        abstract = False")
        lines.append("        indexes = [")
        for idx in indexes_lines:
            lines.append(f"            {idx},")
        lines.append("        ]")
        lines.append("")
        return "\n".join(lines)

    # Override to choose Timescale base per model and pass roles map to factory
    def setup_django_model(self, source_model: XmlSchemaComplexType) -> ConversionCarrier | None:  # type: ignore[override]
        source_model_name = getattr(source_model, "__name__", getattr(source_model, "name", str(source_model)))
        use_ts_base = False
        if self.enable_timescale:
            try:
                use_ts_base = should_use_timescale_base(source_model_name, self._timescale_roles)
            except Exception:
                use_ts_base = False

        base_class: type[models.Model]
        if use_ts_base:
            base_class = XmlTimescaleBase
        else:
            base_class = self.base_model_class

        carrier = ConversionCarrier(
            source_model=source_model,
            meta_app_label=self.app_label,
            base_django_model=base_class,
            class_name_prefix=self.class_name_prefix,
            strict=False,
            enable_gfk=self.enable_gfk,
            gfk_policy=self.gfk_policy,
            gfk_threshold_children=self.gfk_threshold_children,
            gfk_value_mode=self.gfk_value_mode,
            gfk_normalize_common_attrs=self.gfk_normalize_common_attrs,
        )

        # Make roles map available to field factory for FK decisions
        carrier.context_data["_timescale_roles"] = self._timescale_roles

        try:
            self.model_factory_instance.make_django_model(carrier)
            if carrier.django_model:
                self.carriers.append(carrier)
                return carrier
            return None
        except Exception:
            logger.exception("Error creating Django model for %s", source_model_name)
            return None

    def _write_to_file(self, content: str):
        with open(self.output_path, "w") as f:
            f.write(content)
