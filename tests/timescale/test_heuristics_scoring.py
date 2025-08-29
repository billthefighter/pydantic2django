import pytest

from pydantic2django.django.timescale.heuristics import (
    TimescaleRole,
    TimescaleConfig,
    classify_xml_complex_types,
)
from pydantic2django.xmlschema.models import (
    XmlSchemaComplexType,
    XmlSchemaDefinition,
    XmlSchemaElement,
    XmlSchemaType,
)


@pytest.mark.parametrize(
    "elements,attributes,expected",
    [
        # Direct time-like element names
        ([XmlSchemaElement(name="time", base_type=XmlSchemaType.DATETIME)], {}, TimescaleRole.HYPERTABLE),
        ([XmlSchemaElement(name="timestamp", base_type=XmlSchemaType.DATETIME)], {}, TimescaleRole.HYPERTABLE),
        # New: date and datetime keywords should promote to hypertable
        ([XmlSchemaElement(name="date", base_type=XmlSchemaType.DATE)], {}, TimescaleRole.HYPERTABLE),
        ([XmlSchemaElement(name="sampleDate", base_type=XmlSchemaType.DATE)], {}, TimescaleRole.HYPERTABLE),
        ([XmlSchemaElement(name="datetime", base_type=XmlSchemaType.DATETIME)], {}, TimescaleRole.HYPERTABLE),
        # Attribute containing 'date' should also count
        ([], {"eventDate": None}, TimescaleRole.HYPERTABLE),
        # No time-like signals, just name bump (+2) is insufficient (threshold 3) -> dimension
        ([], {}, TimescaleRole.DIMENSION),
    ],
)
def test_scoring_time_and_date_keywords(elements, attributes, expected):
    schema = XmlSchemaDefinition(schema_location="mem.xsd", target_namespace="tns")
    ct = XmlSchemaComplexType(name="StreamsType")
    for el in elements:
        ct.add_element(el)
    # attributes in schema model are objects; use names only for heuristic check
    for attr_name in attributes.keys():
        ct.attributes[attr_name] = None  # type: ignore[assignment]
    schema.add_complex_type(ct)

    roles = classify_xml_complex_types([ct], config=TimescaleConfig(threshold=3))
    assert roles.get("StreamsType") == expected


def test_container_demoted_and_leaf_with_time_promoted():
    schema = XmlSchemaDefinition(schema_location="mem.xsd", target_namespace="tns")
    container = XmlSchemaComplexType(name="StreamsType")
    # Child complex type with direct time
    child = XmlSchemaComplexType(name="ComponentStreamType")
    child.add_element(XmlSchemaElement(name="time", base_type=XmlSchemaType.DATETIME))
    # Reference child from container
    container.add_element(XmlSchemaElement(name="ComponentStream", type_name="ComponentStreamType"))
    schema.add_complex_type(container)
    schema.add_complex_type(child)

    roles = classify_xml_complex_types([container, child])
    assert roles.get("StreamsType") == TimescaleRole.DIMENSION
    assert roles.get("ComponentStreamType") == TimescaleRole.HYPERTABLE


def test_container_with_own_time_not_demoted():
    schema = XmlSchemaDefinition(schema_location="mem.xsd", target_namespace="tns")
    container = XmlSchemaComplexType(name="StreamsType")
    container.add_element(XmlSchemaElement(name="effectiveTime", base_type=XmlSchemaType.DATETIME))
    child = XmlSchemaComplexType(name="ComponentStreamType")
    schema.add_complex_type(container)
    schema.add_complex_type(child)

    roles = classify_xml_complex_types([container, child])
    assert roles.get("StreamsType") == TimescaleRole.HYPERTABLE


def test_list_growth_still_counts():
    schema = XmlSchemaDefinition(schema_location="mem.xsd", target_namespace="tns")
    ct = XmlSchemaComplexType(name="DevicesType")
    ct.add_element(XmlSchemaElement(name="Device", base_type=None, type_name="DeviceType", max_occurs="unbounded"))
    schema.add_complex_type(ct)
    # With name scoring 0 and list +1, below threshold 3 -> dimension
    roles = classify_xml_complex_types([ct])
    assert roles.get("DevicesType") == TimescaleRole.DIMENSION


def test_overrides_respected():
    schema = XmlSchemaDefinition(schema_location="mem.xsd", target_namespace="tns")
    ct = XmlSchemaComplexType(name="ArbitraryType")
    schema.add_complex_type(ct)
    roles = classify_xml_complex_types([ct], overrides={"ArbitraryType": TimescaleRole.HYPERTABLE})
    assert roles.get("ArbitraryType") == TimescaleRole.HYPERTABLE
