"""TimescaleDB integration helpers and bases.

This namespace centralizes Timescale-specific support so it can be reused by
XML, Pydantic, Dataclass, and other generators without duplicating logic.
"""

from .bases import DataclassTimescaleBase, PydanticTimescaleBase, XmlTimescaleBase
from .heuristics import (
    TimescaleRole,
    classify_xml_complex_types,
    is_hypertable,
    should_soft_reference,
    should_use_timescale_base,
)

__all__ = [
    "XmlTimescaleBase",
    "PydanticTimescaleBase",
    "DataclassTimescaleBase",
    "TimescaleRole",
    "classify_xml_complex_types",
    "is_hypertable",
    "should_use_timescale_base",
    "should_soft_reference",
]
