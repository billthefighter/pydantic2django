import pytest

from pydantic2django.django.timescale.heuristics import (
    TimescaleRole,
    classify_xml_complex_types,
    is_hypertable,
    should_soft_reference,
    should_use_timescale_base,
)
from pydantic2django.xmlschema.models import (
    XmlSchemaAttribute,
    XmlSchemaComplexType,
    XmlSchemaElement,
    XmlSchemaDefinition,
    XmlSchemaType,
)


def _mk_schema_with_types(*types: XmlSchemaComplexType) -> XmlSchemaDefinition:
    schema = XmlSchemaDefinition(schema_location="mem.xsd", target_namespace="tns")
    for t in types:
        schema.add_complex_type(t)
    return schema


def test_classification_hypertable_signals_timestamp_and_list():
    # SamplesType with timestamp and list signals => hypertable
    samples = XmlSchemaComplexType(
        name="SamplesType",
        elements=[
            XmlSchemaElement(name="timestamp", base_type=XmlSchemaType.DATETIME),
            XmlSchemaElement(name="reading", base_type=XmlSchemaType.DECIMAL, is_list=True),
        ],
        attributes={"sequence": XmlSchemaAttribute(name="sequence", base_type=XmlSchemaType.LONG)},
    )
    _mk_schema_with_types(samples)

    roles = classify_xml_complex_types([samples])
    assert roles.get("SamplesType") in {TimescaleRole.HYPERTABLE, TimescaleRole.DIMENSION}
    assert is_hypertable("SamplesType", roles)
    assert should_use_timescale_base("SamplesType", roles) is True


def test_classification_dimension_signals_definition_name():
    # Definitions-like name with no time signals => dimension
    defs = XmlSchemaComplexType(
        name="DeviceDefinitionsType",
        elements=[XmlSchemaElement(name="config", base_type=XmlSchemaType.STRING)],
        attributes={},
    )
    _mk_schema_with_types(defs)

    roles = classify_xml_complex_types([defs])
    assert roles.get("DeviceDefinitionsType") in {TimescaleRole.HYPERTABLE, TimescaleRole.DIMENSION}
    assert is_hypertable("DeviceDefinitionsType", roles) is False
    assert should_use_timescale_base("DeviceDefinitionsType", roles) is False


def test_soft_reference_only_when_both_hypertables():
    roles = {
        "SamplesType": TimescaleRole.HYPERTABLE,
        "EventsType": TimescaleRole.HYPERTABLE,
        "HeaderType": TimescaleRole.DIMENSION,
    }

    # hypertable -> hypertable: soft reference required
    assert should_soft_reference("SamplesType", "EventsType", roles) is True
    # hypertable -> dimension: normal FK allowed
    assert should_soft_reference("SamplesType", "HeaderType", roles) is False
    # dimension -> hypertable: current policy does not force soft ref here
    assert should_soft_reference("HeaderType", "SamplesType", roles) is False
