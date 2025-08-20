"""
XML instance ingestion utilities.

Provides a schema-aware ingestor that can parse an XML document and create
corresponding Django model instances generated from an XML Schema.

Scope (initial):
- Supports nested complex types (single and repeated) following the same
  relationship strategies used during generation (default child_fk for lists,
  FK on parent for single nested elements).
- Maps simple elements and attributes to Django fields with basic name
  conversion (camelCase → snake_case), mirroring Xml2DjangoBaseClass helpers.
- Minimal namespace handling by stripping namespace URIs when matching names.

Future extensions (not implemented here):
- Robust namespace mapping and cross-schema references
- key/keyref post-pass resolution using ID/IDREF values
- Type coercion beyond Django's default conversion
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from django.apps import apps as django_apps

from . import get_generated_model
from .discovery import XmlSchemaDiscovery
from .models import XmlSchemaComplexType, XmlSchemaDefinition, XmlSchemaElement

logger = logging.getLogger(__name__)


class XmlInstanceIngestor:
    """
    Schema-aware ingestor for XML instance documents.

    Given XSD schema files and an app_label where models were generated,
    this ingestor parses an XML instance and creates the corresponding Django
    model instances, wiring up relationships according to generation strategy.
    """

    def __init__(
        self,
        *,
        schema_files: list[str | Path],
        app_label: str,
    ):
        try:  # Validate dependency early
            import lxml.etree  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("lxml is required for XML ingestion. Install with: pip install lxml") from exc

        self.app_label = app_label
        self._save_objects: bool = True
        self.created_instances: list[Any] = []

        discovery = XmlSchemaDiscovery()
        discovery.discover_models(packages=[str(p) for p in schema_files], app_label=app_label)
        # Keep references for mapping
        self._schemas: list[XmlSchemaDefinition] = list(discovery.parsed_schemas)

    # --- Public API ---
    def ingest_from_string(self, xml_string: str, *, save: bool = True) -> Any:
        """
        Ingest an XML instance from a string, returning the created root Django instance.
        """
        import lxml.etree as _etree

        self._save_objects = bool(save)
        self.created_instances = []
        root = _etree.fromstring(xml_string)
        return self._ingest_root_element(root)

    def ingest_from_file(self, xml_path: str | Path, *, save: bool = True) -> Any:
        """
        Ingest an XML instance from a file path, returning the created root Django instance.
        """
        xml_path = Path(xml_path)
        import lxml.etree as _etree

        self._save_objects = bool(save)
        self.created_instances = []
        with xml_path.open("rb") as f:
            tree = _etree.parse(f)
        root = tree.getroot()
        return self._ingest_root_element(root)

    # --- Core ingestion ---
    def _ingest_root_element(self, elem: Any) -> Any:
        local_name = self._local_name(elem.tag)
        complex_type = self._resolve_root_complex_type(local_name)
        if complex_type is None:
            raise ValueError(f"Could not resolve complex type for root element '{local_name}'")

        model_cls = self._get_model_for_complex_type(complex_type)
        if model_cls is None:
            raise ValueError(f"Could not find Django model for complex type '{complex_type.name}'")

        instance = self._build_instance_from_element(elem, complex_type, model_cls, parent_instance=None)
        return instance

    def _build_instance_from_element(
        self,
        elem: Any,
        complex_type: XmlSchemaComplexType,
        model_cls: type,
        parent_instance: Optional[Any],
    ) -> Any:
        """
        Create and save a Django model instance from an XML element according to its complex type.
        """
        # Prepare field values for simple elements and attributes first
        field_values: dict[str, Any] = {}

        # Attributes on complex types
        for attr_name, _attr in complex_type.attributes.items():
            xml_attr_value = elem.get(attr_name)
            if xml_attr_value is not None:
                dj_name = self._xml_name_to_django_field(attr_name)
                field_values[dj_name] = xml_attr_value

        # Child elements
        children_by_local: dict[str, list[Any]] = {}
        for child in elem:
            if not isinstance(child.tag, str):
                continue
            lname = self._local_name(child.tag)
            children_by_local.setdefault(lname, []).append(child)

        # Map simple fields and collect nested complex elements to process
        nested_to_process: list[tuple[XmlSchemaElement, list[Any]]] = []
        for el_def in complex_type.elements:
            name = el_def.name
            matched_children = children_by_local.get(name, [])
            if not matched_children:
                continue

            if el_def.type_name and el_def.base_type is None:
                # Nested complex type
                nested_to_process.append((el_def, matched_children))
                continue

            # Simple content
            # Multiple occurrences of a simple element -> pick first; advanced list handling can be added later
            first = matched_children[0]
            dj_name = self._xml_name_to_django_field(name)
            field_values[dj_name] = first.text

        # Instantiate without relationships first
        if self._save_objects:
            instance = model_cls.objects.create(**field_values)
        else:
            try:
                instance = model_cls(**field_values)
            except TypeError as exc:
                # If the dynamically generated class is abstract (not installed app),
                # construct a lightweight proxy object for unsaved workflows.
                is_abstract = getattr(getattr(model_cls, "_meta", None), "abstract", None)
                if is_abstract:
                    try:
                        proxy_cls = type(model_cls.__name__, (), {})
                        instance = proxy_cls()
                        for k, v in field_values.items():
                            setattr(instance, k, v)
                    except Exception:
                        raise TypeError(f"Cannot instantiate abstract model '{model_cls.__name__}'") from exc
                else:
                    raise

        # Track created/instantiated instance
        try:
            self.created_instances.append(instance)
        except Exception:
            pass

        # Attach any remaining XML attributes as dynamic attributes if not mapped
        try:
            model_field_names = {f.name for f in model_cls._meta.fields}
            for attr_name, attr_val in getattr(elem, "attrib", {}).items():
                dj_name = self._xml_name_to_django_field(attr_name)
                if dj_name not in field_values and dj_name not in model_field_names:
                    setattr(instance, dj_name, attr_val)
        except Exception:
            pass

        # Handle nested complex elements
        for el_def, elements in nested_to_process:
            target_type_name = (el_def.type_name or "").split(":")[-1]
            target_complex_type = self._find_complex_type(target_type_name)
            if target_complex_type is None:
                logger.warning("Unknown nested complex type '%s' for element '%s'", target_type_name, el_def.name)
                continue
            target_model_cls = self._get_model_for_complex_type(target_complex_type)
            if target_model_cls is None:
                logger.warning(
                    "Missing Django model for nested type '%s' (element '%s')", target_type_name, el_def.name
                )
                continue

            if el_def.is_list:
                # Default generation style 'child_fk': inject FK on child named after parent class in lowercase
                parent_fk_field = instance.__class__.__name__.lower()
                for child_elem in elements:
                    child_instance = self._build_instance_from_element(
                        child_elem, target_complex_type, target_model_cls, parent_instance=instance
                    )
                    # Set parent FK on child; save update if field exists
                    if hasattr(child_instance, parent_fk_field):
                        setattr(child_instance, parent_fk_field, instance)
                        if self._save_objects:
                            child_instance.save(update_fields=[parent_fk_field])
                    else:
                        # If strategy was m2m/json, this will be a no-op; can extend later
                        logger.debug(
                            "Child model %s lacks FK field '%s' to parent; skipping back-link",
                            child_instance.__class__.__name__,
                            parent_fk_field,
                        )
                continue

            # Single nested complex element: parent holds FK to child in default strategy
            child_elem = elements[0]
            child_instance = self._build_instance_from_element(
                child_elem, target_complex_type, target_model_cls, parent_instance=instance
            )

            parent_fk_name = self._xml_name_to_django_field(el_def.name)
            # For proxy instances (non-Django), the attribute may not exist yet; set unconditionally
            try:
                setattr(instance, parent_fk_name, child_instance)
                if self._save_objects and hasattr(instance, "save"):
                    instance.save(update_fields=[parent_fk_name])
            except Exception:
                logger.debug(
                    "Could not set nested field '%s' on parent %s",
                    parent_fk_name,
                    instance.__class__.__name__,
                )

        return instance

    # --- Helpers ---
    def _resolve_root_complex_type(self, root_local_name: str) -> Optional[XmlSchemaComplexType]:
        # Try global elements with explicit type
        for schema in self._schemas:
            element = schema.elements.get(root_local_name)
            if element and element.type_name:
                type_name = element.type_name.split(":")[-1]
                ct = schema.find_complex_type(type_name, namespace=schema.target_namespace)
                if ct:
                    return ct
        # Try complex type named exactly as root
        for schema in self._schemas:
            ct = schema.find_complex_type(root_local_name, namespace=schema.target_namespace)
            if ct:
                return ct
        return None

    def _find_complex_type(self, type_name: str) -> Optional[XmlSchemaComplexType]:
        for schema in self._schemas:
            ct = schema.find_complex_type(type_name, namespace=schema.target_namespace)
            if ct:
                return ct
        return None

    def _get_model_for_complex_type(self, complex_type: XmlSchemaComplexType) -> Optional[type]:
        model_name = complex_type.name
        try:
            return django_apps.get_model(f"{self.app_label}.{model_name}")
        except Exception:
            # Fallback to in-memory registry for dynamically generated classes
            try:
                return get_generated_model(self.app_label, model_name)
            except Exception:
                return None

    @staticmethod
    def _local_name(qname: str) -> str:
        if "}" in qname:
            return qname.split("}", 1)[1]
        if ":" in qname:
            return qname.split(":", 1)[1]
        return qname

    @staticmethod
    def _xml_name_to_django_field(xml_name: str) -> str:
        # Mirror Xml2DjangoBaseClass._xml_name_to_django_field
        import re

        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", xml_name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
