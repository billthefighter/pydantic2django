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
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime as _dt
from decimal import Decimal
from pathlib import Path
from typing import Any

from django.apps import apps as django_apps

from pydantic2django.core.utils.naming import sanitize_field_identifier
from pydantic2django.core.utils.timescale import (
    TIMESERIES_TIME_ALIASES,
    TimeseriesTimestampMissingError,
    has_timescale_time_field,
    map_time_alias_into_time,
)

from . import get_generated_model
from .discovery import XmlSchemaDiscovery
from .models import XmlSchemaComplexType, XmlSchemaDefinition, XmlSchemaElement

logger = logging.getLogger(__name__)


# --- Process-wide cache for singleton-style reuse (LRU + TTL) ---
CacheKey = tuple[str, tuple[str, ...], tuple[float, ...]]


class _LruTtlCache:
    """Simple LRU cache with TTL semantics for shared XmlInstanceIngestor instances."""

    def __init__(self, maxsize: int = 4, ttl_seconds: float = 600.0, now: Callable[[], float] | None = None) -> None:
        self._data: OrderedDict[CacheKey, tuple[float, XmlInstanceIngestor]] = OrderedDict()
        self._maxsize = int(maxsize)
        self._ttl = float(ttl_seconds)
        self._lock = threading.Lock()
        self._now = now or time.time

    def set_params(self, *, maxsize: int | None = None, ttl_seconds: float | None = None) -> None:
        with self._lock:
            if maxsize is not None:
                self._maxsize = int(maxsize)
            if ttl_seconds is not None:
                self._ttl = float(ttl_seconds)
            self._prune_locked()

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def get(self, key: CacheKey) -> XmlInstanceIngestor | None:
        with self._lock:
            if key not in self._data:
                return None
            created_at, value = self._data.get(key, (0.0, None))  # type: ignore[assignment]
            assert value is not None
            if (self._now() - created_at) > self._ttl:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key: CacheKey, value: XmlInstanceIngestor) -> None:
        with self._lock:
            self._data[key] = (self._now(), value)
            self._data.move_to_end(key)
            self._prune_locked()

    def _prune_locked(self) -> None:
        # Expired
        now = self._now()
        expired = [k for k, (t, _v) in self._data.items() if (now - t) > self._ttl]
        for k in expired:
            self._data.pop(k, None)
        # LRU overflow
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._data),
                "maxsize": self._maxsize,
                "ttl_seconds": self._ttl,
                "keys": list(self._data.keys()),
            }


_INGESTOR_CACHE = _LruTtlCache()
_WARMED_KEYS: set[CacheKey] = set()


def _normalize_schema_files(schema_files: list[str | Path]) -> list[str]:
    return [str(Path(p).resolve()) for p in schema_files]


def _schema_key(app_label: str, schema_files: list[str | Path]) -> CacheKey:
    files = _normalize_schema_files(schema_files)
    mtimes = []
    for f in files:
        try:
            mtimes.append(Path(f).stat().st_mtime)
        except Exception:
            mtimes.append(0.0)
    return (app_label, tuple(files), tuple(mtimes))


def warmup_xmlschema_models(schema_files: list[str | Path], *, app_label: str) -> None:
    """
    Pre-generate and register XML-derived Django models once per (app_label, schema set).

    This runs the generator to populate the in-memory registry used by the ingestor,
    avoiding repeated generation when used in long-lived processes (e.g., workers).
    """
    try:
        from .generator import XmlSchemaDjangoModelGenerator
    except Exception:
        # Generator not available or optional deps missing; skip warmup.
        return

    key = _schema_key(app_label, schema_files)
    if key in _WARMED_KEYS:
        return

    try:
        gen = XmlSchemaDjangoModelGenerator(
            schema_files=[Path(p) for p in _normalize_schema_files(schema_files)],
            app_label=app_label,
            output_path="__discard__.py",
            verbose=False,
        )
        _ = gen.generate_models_file()
        _WARMED_KEYS.add(key)
        logger.info("Warmed XML models for app '%s' with %d schema files", app_label, len(schema_files))
    except Exception as exc:
        logger.debug("Warmup failed for app '%s': %s", app_label, exc)


def get_shared_ingestor(
    *, schema_files: list[str | Path], app_label: str, dynamic_model_fallback: bool = False, strict: bool = False
) -> XmlInstanceIngestor:
    """
    Get or create a shared XmlInstanceIngestor for a given (app_label, schema set).

    Reuses the same ingestor instance across calls if the schema files (paths and mtimes)
    are unchanged, eliminating repeated discovery and setup work.

    Args:
        schema_files: Absolute or relative paths to XSD files. Paths are normalized.
        app_label: Django app label that contains the installed models for the discovered types.
        dynamic_model_fallback: If True, when an installed model is not found in the app
            registry, the ingestor will fall back to dynamically generated stand-in classes registered
            in the in-memory registry. If False, the ingestor will raise a ModelResolutionError instead
            of falling back. This is useful to guarantee that only concrete, installed Django models
            (with managers) are used during ingestion.

    Notes:
        - The shared-ingestor cache key includes the value of `dynamic_model_fallback` to avoid reusing
          an instance created with a different fallback policy.
    """
    # Include fallback and strict flags in the cache key (without changing CacheKey type)
    cache_label = f"{app_label}|dyn={1 if dynamic_model_fallback else 0}|strict={1 if strict else 0}"
    key = _schema_key(cache_label, schema_files)
    cached = _INGESTOR_CACHE.get(key)
    if cached is not None:
        return cached

    # Optional pre-warm to ensure classes are registered
    warmup_xmlschema_models(schema_files, app_label=app_label)

    inst = XmlInstanceIngestor(
        schema_files=[Path(p) for p in _normalize_schema_files(schema_files)],
        app_label=app_label,
        dynamic_model_fallback=dynamic_model_fallback,
        strict=strict,
    )
    _INGESTOR_CACHE.put(key, inst)
    return inst


def set_ingestor_cache(maxsize: int | None = None, ttl_seconds: float | None = None) -> None:
    """Configure global shared-ingestor cache (LRU size and TTL)."""
    _INGESTOR_CACHE.set_params(maxsize=maxsize, ttl_seconds=ttl_seconds)


def clear_ingestor_cache() -> None:
    """Clear all cached ingestors (helpful for tests)."""
    _INGESTOR_CACHE.clear()


def ingestor_cache_stats() -> dict[str, Any]:
    """Return internal cache stats for diagnostics."""
    return _INGESTOR_CACHE.stats()


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
        dynamic_model_fallback: bool = False,
        strict: bool = False,
    ):
        """
        Initialize a schema-aware XML ingestor.

        Args:
            schema_files: Paths to XSD files used to parse/resolve types.
            app_label: Django app label expected to contain installed models.
            dynamic_model_fallback: Controls behavior when an installed model is missing.
                - True: fall back to dynamically generated stand-ins from the
                  in-memory registry, allowing unsaved/ephemeral workflows.
                - False (default): raise ModelResolutionError when a discovered complex type cannot be
                  resolved to an installed Django model, avoiding implicit usage of stand-ins.

        Behavior:
            - Regardless of the fallback flag, the ingestor will always prefer installed Django
              models and pre-cache them for all discovered complex types to ensure consistent
              resolution across root and nested elements.
        """
        try:  # Validate dependency early
            import lxml.etree  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("lxml is required for XML ingestion. Install with: pip install lxml") from exc

        self.app_label = app_label
        self._save_objects: bool = True
        self._dynamic_model_fallback: bool = bool(dynamic_model_fallback)
        self._strict: bool = bool(strict)
        self.created_instances: list[Any] = []

        discovery = XmlSchemaDiscovery()
        discovery.discover_models(packages=[str(p) for p in schema_files], app_label=app_label)
        # Keep references for mapping
        self._schemas: list[XmlSchemaDefinition] = list(discovery.parsed_schemas)

        # Pre-resolve and cache installed Django model classes for all discovered complex types.
        # This guarantees we consistently use concrete models (with managers) when available.
        self._model_resolution_cache: dict[str, type] = {}
        try:
            discovered_names: set[str] = set()
            for schema in self._schemas:
                for ct in schema.get_all_complex_types():
                    discovered_names.add(ct.name)
            for model_name in discovered_names:
                try:
                    model_cls = django_apps.get_model(f"{self.app_label}.{model_name}")
                except Exception:
                    model_cls = None
                if model_cls is not None:
                    self._model_resolution_cache[model_name] = model_cls
        except Exception:
            # Cache population is best-effort; fall back to on-demand lookups.
            self._model_resolution_cache = {}

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

    def validate_models(self, *, strict: bool = True) -> list[ContractIssue]:
        """
        Validate that discovered schema types align with installed Django models.

        - Ensures expected fields from attributes, simple elements, and single nested complex
          elements exist on the corresponding Django model.
        - Flags Timescale-based models that are missing the canonical 'time' field.

        Returns a list of ContractIssue. If strict=True and issues exist, raises SchemaSyncError
        with a concise remediation message.
        """
        issues: list[ContractIssue] = []
        for schema in self._schemas:
            try:
                complex_types = list(schema.get_all_complex_types())
            except Exception:
                continue
            for ct in complex_types:
                model_cls = self._get_model_for_complex_type(ct)
                if model_cls is None:
                    issues.append(
                        ContractIssue(
                            complex_type_name=ct.name,
                            model_name=f"{self.app_label}.{ct.name}",
                            missing_fields=["<installed model not found>"],
                            extra_fields=[],
                            problems=["No installed Django model resolved for complex type"],
                        )
                    )
                    continue

                try:
                    model_field_names = {f.name for f in model_cls._meta.fields}
                except Exception:
                    model_field_names = set()

                expected_fields: set[str] = set()
                # Attributes
                for attr_name in getattr(ct, "attributes", {}).keys():
                    expected_fields.add(self._xml_name_to_django_field(attr_name))
                # Elements (simple + parent links for single nested complex)
                for el in getattr(ct, "elements", []):
                    if getattr(el, "type_name", None) and getattr(el, "base_type", None) is None:
                        if not getattr(el, "is_list", False):
                            expected_fields.add(self._xml_name_to_django_field(el.name))
                        continue
                    expected_fields.add(self._xml_name_to_django_field(el.name))

                missing = sorted([fn for fn in expected_fields if fn not in model_field_names])
                problems: list[str] = []
                if self._is_timescale_model(model_cls) and "time" not in model_field_names:
                    problems.append("Timescale model missing required 'time' field")

                if missing or problems:
                    issues.append(
                        ContractIssue(
                            complex_type_name=ct.name,
                            model_name=f"{self.app_label}.{model_cls.__name__}",
                            missing_fields=missing,
                            extra_fields=[],
                            problems=problems,
                        )
                    )

        if strict and issues:
            details = "; ".join(
                f"[{i.model_name}] missing={i.missing_fields or '-'} problems={i.problems or '-'}" for i in issues[:5]
            )
            raise SchemaSyncError(
                f"Schema and static model are out of sync; verify schema and/or regenerate static Django models and re-migrate. Details: {details}"
            )
        return issues

    # --- Core ingestion ---
    def _ingest_root_element(self, elem: Any) -> Any:
        local_name = self._local_name(elem.tag)
        complex_type = self._resolve_root_complex_type(local_name)
        if complex_type is None:
            known_types = []
            try:
                for s in self._schemas:
                    known_types.extend([ct.name for ct in s.get_all_complex_types()])
            except Exception:
                pass
            raise ValueError(
                "Could not resolve complex type for root element "
                f"'{local_name}'. Known discovered types (sample): "
                f"{known_types[:10]}"
            )

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
        parent_instance: Any | None,
        parent_link_field: str | None = None,
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

        # Strict mode: detect unexpected XML child elements and attributes not declared in the schema
        if self._strict:
            expected_child_names = {e.name for e in getattr(complex_type, "elements", [])}
            unexpected_children = sorted([n for n in children_by_local.keys() if n not in expected_child_names])
            if unexpected_children:
                raise SchemaSyncError(
                    "Schema and static model are out of sync; verify schema and/or regenerate static Django models and re-migrate. "
                    f"Unexpected XML elements for type '{complex_type.name}': {unexpected_children}"
                )
            expected_attr_names = set(getattr(complex_type, "attributes", {}).keys())
            xml_attr_local = {self._local_name(k) for k in getattr(elem, "attrib", {}).keys()}
            unexpected_attrs = sorted([n for n in xml_attr_local if n not in expected_attr_names])
            if unexpected_attrs:
                raise SchemaSyncError(
                    "Schema and static model are out of sync; verify schema and/or regenerate static Django models and re-migrate. "
                    f"Unmapped XML attributes for type '{complex_type.name}': {unexpected_attrs}"
                )

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

        # If this element is a repeated child (child_fk strategy), and we know the FK field name
        # on the child model, include it in initial field values to satisfy NOT NULL constraints.
        if parent_instance is not None and parent_link_field:
            try:
                model_field_names = {f.name for f in model_cls._meta.fields}
                if parent_link_field in model_field_names:
                    field_values.setdefault(parent_link_field, parent_instance)
            except Exception:
                pass

        # Instantiate without other relationships first
        instance: Any
        if self._save_objects:
            # Timescale-aware timestamp remapping: if model expects a 'time' field,
            # remap common XML timestamp attributes (e.g., creationTime → creation_time)
            # into 'time' when not already provided.
            try:
                model_field_names = {f.name for f in model_cls._meta.fields}
            except Exception:
                model_field_names = set()
            if has_timescale_time_field(model_field_names) and "time" not in field_values:
                # Always attempt a helpful alias remap; only enforce requiredness for Timescale models
                map_time_alias_into_time(field_values, aliases=TIMESERIES_TIME_ALIASES)
                if self._is_timescale_model(model_cls) and "time" not in field_values:
                    raise TimeseriesTimestampMissingError(
                        model_cls.__name__, attempted_aliases=list(TIMESERIES_TIME_ALIASES)
                    )

            instance = model_cls.objects.create(**field_values)
        else:
            try:
                # Mirror the same remapping for unsaved construction flows
                try:
                    model_field_names = {f.name for f in model_cls._meta.fields}
                except Exception:
                    model_field_names = set()
                if has_timescale_time_field(model_field_names) and "time" not in field_values:
                    # For unsaved flows, remap aliases but do not enforce a hard requirement
                    map_time_alias_into_time(field_values, aliases=TIMESERIES_TIME_ALIASES)
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
            unmapped_dynamic: list[str] = []
            for attr_name, attr_val in getattr(elem, "attrib", {}).items():
                dj_name = self._xml_name_to_django_field(attr_name)
                if dj_name not in field_values and dj_name not in model_field_names:
                    if self._strict:
                        unmapped_dynamic.append(dj_name)
                    else:
                        setattr(instance, dj_name, attr_val)
            if self._strict and unmapped_dynamic:
                raise SchemaSyncError(
                    "Schema and static model are out of sync; verify schema and/or regenerate static Django models and re-migrate. "
                    f"Unmapped XML attributes for type '{complex_type.name}': {unmapped_dynamic}"
                )
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
                # If the parent model exposes GenericRelation('entries'), persist as GenericEntry rows
                try:
                    has_entries = hasattr(instance, "entries")
                except Exception:
                    has_entries = False
                if has_entries:
                    try:
                        GenericEntry = django_apps.get_model(f"{self.app_label}.GenericEntry")
                    except Exception:
                        GenericEntry = None
                    if GenericEntry is not None:
                        for idx, child_elem in enumerate(elements):
                            entry_data_kwargs = self._build_generic_entry_kwargs(
                                instance=instance,
                                el_name=el_def.name,
                                target_type_name=target_type_name,
                                child_elem=child_elem,
                                GenericEntry=GenericEntry,
                                order_index=idx,
                            )
                            # Create entry row
                            if self._save_objects:
                                GenericEntry.objects.create(**entry_data_kwargs)
                        # Skip concrete child instance creation when using GFK
                        continue
                # Default generation style 'child_fk': inject FK on child named after parent class in lowercase
                parent_fk_field = instance.__class__.__name__.lower()
                for child_elem in elements:
                    child_instance: Any = self._build_instance_from_element(
                        child_elem,
                        target_complex_type,
                        target_model_cls,
                        parent_instance=instance,
                        parent_link_field=parent_fk_field,
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

            # Single nested complex element: either FK on parent or GFK entry (all_nested policy)
            parent_fk_name = self._xml_name_to_django_field(el_def.name)
            # If parent exposes entries and does NOT define a concrete FK field for this element, persist as GenericEntry
            use_gfk_single = False
            try:
                has_entries = hasattr(instance, "entries")
            except Exception:
                has_entries = False
            if has_entries:
                try:
                    model_field_names = {f.name for f in model_cls._meta.fields}
                except Exception:
                    model_field_names = set()
                if parent_fk_name not in model_field_names:
                    use_gfk_single = True

            if use_gfk_single:
                try:
                    GenericEntry = django_apps.get_model(f"{self.app_label}.GenericEntry")
                except Exception:
                    GenericEntry = None
                if GenericEntry is not None:
                    child_elem = elements[0]
                    single_entry_kwargs = self._build_generic_entry_kwargs(
                        instance=instance,
                        el_name=el_def.name,
                        target_type_name=target_type_name,
                        child_elem=child_elem,
                        GenericEntry=GenericEntry,
                        order_index=0,
                    )
                    if self._save_objects:
                        GenericEntry.objects.create(**single_entry_kwargs)
                # Skip concrete child instance creation
                continue

            # Default: create child instance and set FK on parent
            child_elem = elements[0]
            child_instance = self._build_instance_from_element(
                child_elem, target_complex_type, target_model_cls, parent_instance=instance
            )

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

    @staticmethod
    def _build_generic_entry_kwargs(
        *,
        instance: Any,
        el_name: str,
        target_type_name: str | None,
        child_elem: Any,
        GenericEntry: Any,
        order_index: int,
    ) -> dict[str, Any]:
        """Build keyword args for creating a GenericEntry from an XML child element.

        Parses attributes and element text. When the GenericEntry model defines typed
        value columns, attempts to parse numeric and ISO8601 datetime values.
        """
        entry_attrs_json: dict[str, Any] = {}
        try:
            for k, v in getattr(child_elem, "attrib", {}).items():
                entry_attrs_json[k] = v
        except Exception:
            pass

        entry_kwargs: dict[str, Any] = {
            "content_object": instance,
            "element_qname": el_name,
            "type_qname": target_type_name or None,
            "attrs_json": entry_attrs_json,
            "order_index": order_index,
            "path_hint": el_name,
        }

        # Handle text value
        try:
            text_val = getattr(child_elem, "text", None)
        except Exception:
            text_val = None
        if text_val and str(text_val).strip() != "":
            raw = str(text_val).strip()
            # If GenericEntry defines typed columns, attempt parse to numeric/time
            has_text_col = False
            try:
                has_text_col = hasattr(GenericEntry, "_meta") and any(
                    f.name == "text_value" for f in GenericEntry._meta.fields
                )
            except Exception:
                has_text_col = False
            if has_text_col:
                entry_kwargs["text_value"] = raw
                try:
                    entry_kwargs["num_value"] = Decimal(raw)
                except Exception:
                    pass
                try:
                    iso = raw.replace("Z", "+00:00")
                    entry_kwargs["time_value"] = _dt.fromisoformat(iso)
                except Exception:
                    pass
            else:
                entry_attrs_json["value"] = raw

        return entry_kwargs

    # --- Helpers ---
    def _resolve_root_complex_type(self, root_local_name: str) -> XmlSchemaComplexType | None:
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

    def _find_complex_type(self, type_name: str) -> XmlSchemaComplexType | None:
        for schema in self._schemas:
            ct = schema.find_complex_type(type_name, namespace=schema.target_namespace)
            if ct:
                return ct
        return None

    def _get_model_for_complex_type(self, complex_type: XmlSchemaComplexType) -> type | None:
        model_name = complex_type.name
        # Prefer cache of installed models first for determinism across root and nested types
        cached = getattr(self, "_model_resolution_cache", {}).get(model_name)
        if cached is not None:
            return cached

        # Otherwise, attempt to resolve from app registry, then fall back to generated stand-ins
        try:
            model_cls = django_apps.get_model(f"{self.app_label}.{model_name}")
            # Populate cache for subsequent lookups
            try:
                self._model_resolution_cache[model_name] = model_cls
            except Exception:
                pass
            return model_cls
        except Exception as exc:
            # Optionally fallback to in-memory registry for dynamically generated classes
            if not getattr(self, "_dynamic_model_fallback", True):
                # Provide a detailed error to aid configuration
                raise ModelResolutionError(
                    app_label=self.app_label,
                    model_name=model_name,
                    message=(
                        "Installed Django model not found and dynamic model fallback is disabled. "
                        "Ensure your concrete models are installed under the correct app label, or enable "
                        "dynamic fallback explicitly."
                    ),
                ) from exc
            try:
                return get_generated_model(self.app_label, model_name)
            except Exception:
                return None

    @staticmethod
    def _is_timescale_model(model_cls: type) -> bool:
        try:
            from pydantic2django.django.models import TimescaleModel as _TsModel
        except Exception:
            return False
        try:
            return issubclass(model_cls, _TsModel)
        except Exception:
            return False

    @staticmethod
    def _local_name(qname: str) -> str:
        if "}" in qname:
            return qname.split("}", 1)[1]
        if ":" in qname:
            return qname.split(":", 1)[1]
        return qname

    @staticmethod
    def _xml_name_to_django_field(xml_name: str) -> str:
        # Use shared normalization so all ingestion paths stay consistent
        return sanitize_field_identifier(xml_name)


class ModelResolutionError(Exception):
    def __init__(self, *, app_label: str, model_name: str, message: str) -> None:
        super().__init__(f"[{app_label}.{model_name}] {message}")
        self.app_label = app_label
        self.model_name = model_name


# TimeseriesTimestampMissingError is provided by pydantic2django.core.utils.timescale


@dataclass
class ContractIssue:
    complex_type_name: str
    model_name: str
    missing_fields: list[str]
    extra_fields: list[str]
    problems: list[str]


class SchemaSyncError(Exception):
    """Raised when schema-to-model contract validation fails (strict mode)."""
