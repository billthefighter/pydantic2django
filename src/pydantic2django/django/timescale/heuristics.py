"""Heuristics and helpers to classify models for TimescaleDB usage.

Primary goals:
- Decide whether a model should be a hypertable (Timescale-enabled) or a regular table
- Enforce safe FK behavior: hypertable → hypertable FKs are not allowed (soft references instead)

Scoring overview for XML complex types (default threshold = 3):
- +2 if a time-like element/attribute exists (e.g., ``time``, ``timestamp``, ``sequence``, ``effectiveTime``, ``sampleRate``)
- +2 if the type name matches observation/event patterns (e.g., ``Samples``, ``Events``, ``Condition``, ``*Changed``, ``*Removed``, ``*Added``, ``Streams``)
- +1 if the schema suggests unbounded/high-cardinality growth (e.g., an element with ``is_list=True``)
- −2 if the name matches definition/metadata categories (e.g., ``*Definition*``, ``*Definitions*``, ``*Parameters*``, ``Header``, ``Counts``, ``Configuration``, ``Description``, ``Location``, ``Limits``, ``Reference``, ``Relationships``)

Types with score ≥ threshold are treated as hypertables; otherwise they are dimensions.
Soft references are realized as indexed scalar fields (e.g., ``UUIDField(db_index=True, null=True, blank=True)``) in place of illegal hypertable→hypertable FKs.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, get_origin


class TimescaleRole(str, Enum):
    HYPERTABLE = "hypertable"
    DIMENSION = "dimension"


@dataclass(frozen=True)
class TimescaleConfig:
    """Configuration knobs for the heuristics.

    Attributes:
        threshold: Total score required to classify as a hypertable (default: 3).
        default_soft_ref_field: Name of the Django field type to use for soft references.
            Defaults to ``UUIDField``; you can layer PK-specific soft refs if needed.
    """

    threshold: int = 3
    default_soft_ref_field: str = "UUIDField"


def _score_name(name: str) -> int:
    """Score based on the complex type's name.

    Heuristics:
    - +2 if name looks like an observation/event (``Samples``, ``Events``, ``Condition``, ``*Changed``,
      ``*Removed``, ``*Added``, ``Streams``)
    - −2 if name looks like a definition/metadata container (``*Definition*``, ``*Definitions*``,
      ``Constraints``, ``Properties``, ``Parameters``, ``Header``, ``Counts``, ``Configuration``,
      ``Description``, ``Location``, ``Limits``, ``Reference``, ``Relationships``)
    """
    score = 0
    # Hypertable-ish names
    if re.search(r"(Samples|Events|Condition|Changed|Removed|Added|Streams)", name, re.IGNORECASE):
        score += 2
    # Dimension-ish names
    if re.search(
        r"(Definition|Definitions|Constraints|Properties|Parameters|Header|Counts|Configuration|Description|Location|Limits|Reference|Relationships)",
        name,
        re.IGNORECASE,
    ):
        score -= 2
    return score


def _score_xml_features(xml_complex_type) -> int:
    """Score based on XML structure for ``XmlSchemaComplexType``.

    Heuristics:
    - +2 if a time/timestamp/sequence-like field exists in attributes OR elements
    - +1 if any element indicates list/unbounded growth (``is_list=True``)

    Relies on basic attributes available on the in-memory XML model.
    """
    score = 0
    # time-ish fields
    time_keys = {"time", "timestamp", "sequence", "effectiveTime", "sampleRate", "date", "datetime"}
    try:
        for attr_name in getattr(xml_complex_type, "attributes", {}).keys():
            if any(k.lower() in attr_name.lower() for k in time_keys):
                score += 2
        for el in getattr(xml_complex_type, "elements", []) or []:
            if any(k.lower() in getattr(el, "name", "").lower() for k in time_keys):
                score += 2
            if getattr(el, "is_list", False):
                score += 1  # unbounded growth
    except Exception:
        pass
    return score


def classify_xml_complex_types(
    models: Iterable,
    overrides: dict[str, TimescaleRole] | None = None,
    *,
    config: TimescaleConfig | None = None,
) -> dict[str, TimescaleRole]:
    """Classify XML complex types into hypertable vs dimension.

    Scoring (defaults shown; adjust with :class:`TimescaleConfig`):
    - +2 for time-like fields
    - +2 for observation/event-like names
    - +1 for unbounded/list growth
    - −2 for definition/metadata-like names

    Any type with total score ≥ ``threshold`` (default 3) is a hypertable; others are dimensions.

    Args:
        models: Iterable of complex types (objects providing ``name``, ``elements``, and ``attributes``).
        overrides: Optional mapping of type name to explicit :class:`TimescaleRole` to force classification.
        config: Optional configuration (``threshold``, soft ref defaults).

    Returns:
        Mapping ``{type_name: TimescaleRole}``.
    """
    cfg = config or TimescaleConfig()
    result: dict[str, TimescaleRole] = {}
    overrides = overrides or {}

    # Build a quick lookup of complex types by name for descendant analysis
    name_to_model: dict[str, Any] = {}
    for m in models:
        try:
            n = getattr(m, "name", None) or getattr(m, "__name__", str(m))
            name_to_model[n] = m
        except Exception:
            continue

    def _has_direct_time_feature(xct: Any) -> bool:
        try:
            time_keys = {"time", "timestamp", "sequence", "effectiveTime", "sampleRate"}
            for attr_name in getattr(xct, "attributes", {}).keys():
                if any(k.lower() in attr_name.lower() for k in time_keys):
                    return True
            for el in getattr(xct, "elements", []) or []:
                if any(k.lower() in getattr(el, "name", "").lower() for k in time_keys):
                    return True
            return False
        except Exception:
            return False

    def _child_complex_types(xct: Any) -> list[Any]:
        children: list[Any] = []
        try:
            for el in getattr(xct, "elements", []) or []:
                # Prefer an inline complex_type, otherwise resolve by type_name
                child = getattr(el, "complex_type", None)
                if child is None:
                    tname = getattr(el, "type_name", None)
                    if tname and tname in name_to_model:
                        child = name_to_model[tname]
                if child is not None:
                    children.append(child)
        except Exception:
            pass
        return children

    # First pass: base scoring
    for m in models:
        name = getattr(m, "name", None) or getattr(m, "__name__", str(m))
        if name in overrides:
            result[name] = overrides[name]
            continue

        score = _score_name(name) + _score_xml_features(m)
        role = TimescaleRole.HYPERTABLE if score >= cfg.threshold else TimescaleRole.DIMENSION
        result[name] = role

    # Second pass: demote container types without direct time fields and promote leaf children with time fields
    for m in models:
        name = getattr(m, "name", None) or getattr(m, "__name__", str(m))
        try:
            # Skip if this type already has direct time features
            if _has_direct_time_feature(m):
                continue

            # Find descendant complex types that have direct time features
            first_children = _child_complex_types(m)
            stack = list(first_children)
            leaf_time_types: set[str] = set()
            seen: set[int] = set()
            while stack:
                node = stack.pop()
                if id(node) in seen:
                    continue
                seen.add(id(node))
                if _has_direct_time_feature(node):
                    leaf_time_types.add(getattr(node, "name", getattr(node, "__name__", str(node))))
                else:
                    stack.extend(_child_complex_types(node))

            if leaf_time_types:
                # If this container was initially classified hypertable, demote it
                if result.get(name) == TimescaleRole.HYPERTABLE:
                    result[name] = TimescaleRole.DIMENSION
                # Ensure all leaf time-bearing types are hypertables
                for tname in leaf_time_types:
                    result[tname] = TimescaleRole.HYPERTABLE
            else:
                # No leaf time types found. If it clearly acts as a container (has any child complex types)
                # and lacks direct time, classify as dimension defensively to avoid container hypertables
                if first_children and result.get(name) == TimescaleRole.HYPERTABLE:
                    result[name] = TimescaleRole.DIMENSION
        except Exception:
            # Be defensive: never fail classification due to structure issues
            continue

    return result


def _score_time_fields_in_mapping(field_names: Iterable[str]) -> int:
    score = 0
    time_keys = {"time", "timestamp", "sequence", "effectiveTime", "sampleRate"}
    try:
        for name in field_names:
            if any(k.lower() in str(name).lower() for k in time_keys):
                score += 2
                break
    except Exception:
        pass
    return score


def _has_list_annotation(type_obj: Any) -> bool:
    try:
        origin = get_origin(type_obj)
        return origin in (list, set, tuple)
    except Exception:
        return False


def classify_pydantic_models(
    models: Iterable[Any], overrides: dict[str, TimescaleRole] | None = None, *, config: TimescaleConfig | None = None
) -> dict[str, TimescaleRole]:
    """Classify Pydantic ``BaseModel`` types using name/field heuristics.

    Signals:
    - +2 if any field name is time-like
    - +2 if class name matches observation/event patterns
    - +1 if any field annotation is a list/collection (append-only growth signal)
    - −2 for definition/metadata-like class names
    """
    cfg = config or TimescaleConfig()
    result: dict[str, TimescaleRole] = {}
    overrides = overrides or {}

    for cls in models:
        name = getattr(cls, "__name__", str(cls))
        if name in overrides:
            result[name] = overrides[name]
            continue

        score = _score_name(name)
        # time-like fields by name
        try:
            model_fields = getattr(cls, "model_fields", {}) or {}
            score += _score_time_fields_in_mapping(model_fields.keys())
            # list/collection fields
            for f in getattr(cls, "__annotations__", {}).values():
                if _has_list_annotation(f):
                    score += 1
                    break
        except Exception:
            pass

        role = TimescaleRole.HYPERTABLE if score >= cfg.threshold else TimescaleRole.DIMENSION
        result[name] = role

    return result


def classify_dataclass_types(
    models: Iterable[Any], overrides: dict[str, TimescaleRole] | None = None, *, config: TimescaleConfig | None = None
) -> dict[str, TimescaleRole]:
    """Classify Python dataclasses using name/field heuristics."""
    import dataclasses as _dc

    cfg = config or TimescaleConfig()
    result: dict[str, TimescaleRole] = {}
    overrides = overrides or {}

    for cls in models:
        name = getattr(cls, "__name__", str(cls))
        if name in overrides:
            result[name] = overrides[name]
            continue

        score = _score_name(name)
        # time-like fields by name
        try:
            if _dc.is_dataclass(cls):
                score += _score_time_fields_in_mapping(f.name for f in _dc.fields(cls))
                # list/collection fields via annotations
                for f in getattr(cls, "__annotations__", {}).values():
                    if _has_list_annotation(f):
                        score += 1
                        break
        except Exception:
            pass

        role = TimescaleRole.HYPERTABLE if score >= cfg.threshold else TimescaleRole.DIMENSION
        result[name] = role

    return result


def is_hypertable(model_name: str, roles: dict[str, TimescaleRole]) -> bool:
    """Return ``True`` if ``model_name`` is classified as a hypertable in ``roles``."""
    return roles.get(model_name) == TimescaleRole.HYPERTABLE


def should_soft_reference(source_model_name: str, target_model_name: str, roles: dict[str, TimescaleRole]) -> bool:
    """Return ``True`` if an FK from ``source`` to ``target`` would be hypertable→hypertable.

    TimescaleDB disallows hypertable→hypertable FKs. Callers should emit an indexed scalar field
    (e.g., ``UUIDField(db_index=True, null=True, blank=True)``) instead of a ``ForeignKey``.
    """
    return is_hypertable(source_model_name, roles) and is_hypertable(target_model_name, roles)


def should_invert_fk(source_model_name: str, target_model_name: str, roles: dict[str, TimescaleRole]) -> bool:
    """Return True if FK should be inverted to point from hypertable -> dimension.

    Policy:
    - If source is a dimension (not hypertable) and target is a hypertable, invert.
    - This avoids illegal/fragile FKs to hypertables (PK dropped on hypertable creation),
      while preserving joinability by referencing the dimension from the hypertable instead.
    - Hypertable -> hypertable remains disallowed (handled by should_soft_reference).
    """
    return (not is_hypertable(source_model_name, roles)) and is_hypertable(target_model_name, roles)


def should_use_timescale_base(model_name: str, roles: dict[str, TimescaleRole]) -> bool:
    """Return ``True`` if the model should inherit from a Timescale-enabled base.

    Used by generators to pick an appropriate base class (e.g., ``XmlTimescaleBase``) for hypertables.
    """
    return is_hypertable(model_name, roles)
