import pytest

from django.db import models

from pydantic2django.core.factories import ConversionCarrier
from pydantic2django.xmlschema.factory import XmlSchemaFieldFactory, XmlSchemaModelFactory
from pydantic2django.xmlschema.models import (
    XmlSchemaComplexType,
    XmlSchemaDefinition,
    XmlSchemaElement,
)


def _make_schema_with_types(*types: XmlSchemaComplexType) -> XmlSchemaDefinition:
    schema = XmlSchemaDefinition(schema_location="mem.xsd", target_namespace="tns")
    for t in types:
        schema.add_complex_type(t)
    return schema


def test_element_soft_reference_for_hypertable_to_hypertable():
    # Build two complex types A (source) and B (target)
    src = XmlSchemaComplexType(name="SamplesType")
    tgt = XmlSchemaComplexType(name="EventsType")
    schema = _make_schema_with_types(src, tgt)

    # Element on source referencing target complex type
    element = XmlSchemaElement(name="event", type_name="EventsType", nillable=True)

    # Carrier configured with timescale roles marking both as hypertables
    carrier = ConversionCarrier(
        source_model=src,
        meta_app_label="tests",
        base_django_model=models.Model,
        class_name_prefix="",
        strict=False,
    )
    carrier.source_model.schema_def = schema
    carrier.context_data["_timescale_roles"] = {"SamplesType": "hypertable", "EventsType": "hypertable"}

    factory = XmlSchemaFieldFactory()
    field_class, kwargs = factory._create_element_field(element, src.name, carrier)

    assert field_class is models.UUIDField
    assert kwargs.get("db_index") is True
    assert kwargs.get("null") is True and kwargs.get("blank") is True


def test_element_fk_for_hypertable_to_dimension():
    src = XmlSchemaComplexType(name="SamplesType")
    tgt = XmlSchemaComplexType(name="HeaderType")
    schema = _make_schema_with_types(src, tgt)
    element = XmlSchemaElement(name="header", type_name="HeaderType", nillable=True)

    carrier = ConversionCarrier(
        source_model=src,
        meta_app_label="tests",
        base_django_model=models.Model,
        class_name_prefix="",
        strict=False,
    )
    carrier.source_model.schema_def = schema
    carrier.context_data["_timescale_roles"] = {"SamplesType": "hypertable", "HeaderType": "dimension"}

    factory = XmlSchemaFieldFactory()
    field_class, kwargs = factory._create_element_field(element, src.name, carrier)

    assert field_class is models.ForeignKey
    assert kwargs.get("to") == "tests.HeaderType"


def test_finalize_relationships_soft_reference_child_to_parent_for_hypertables():
    # Prepare carriers for child/parent
    parent = XmlSchemaComplexType(name="SamplesType")
    child = XmlSchemaComplexType(name="EventsType")
    child_carrier = ConversionCarrier(
        source_model=child,
        meta_app_label="tests",
        base_django_model=models.Model,
        class_name_prefix="",
        strict=False,
    )
    # Django model placeholder for child to pass the check
    ChildDjango = type("ChildDjango", (models.Model,), {"__module__": "tests", "Meta": type("Meta", (), {"app_label": "tests"})})
    child_carrier.django_model = ChildDjango
    child_carrier.context_data["_timescale_roles"] = {"SamplesType": "hypertable", "EventsType": "hypertable"}

    carriers_by_name = {"EventsType": child_carrier}

    model_factory = XmlSchemaModelFactory(app_label="tests")
    # Inject a pending child FK relation equivalent
    model_factory._pending_child_fk = [
        {"child": "EventsType", "parent": "SamplesType", "element_name": "events"}
    ]

    model_factory.finalize_relationships(carriers_by_name, app_label="tests")

    # Expect a UUIDField definition rather than a ForeignKey for child->parent
    defs = child_carrier.django_field_definitions
    assert "samplestype" in defs  # field name used
    assert "UUIDField(" in defs["samplestype"], defs["samplestype"]
    assert "ForeignKey(" not in defs["samplestype"], defs["samplestype"]
