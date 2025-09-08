import pytest

from django.db import models

from pydantic2django.django.models import Xml2DjangoBaseClass
from pydantic2django.django.timescale.bases import XmlTimescaleBase
from pydantic2django.django.timescale.heuristics import TimescaleRole
from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator
from pydantic2django.xmlschema.models import (
    XmlSchemaComplexType,
    XmlSchemaDefinition,
    XmlSchemaElement,
    XmlSchemaType,
)
from pydantic2django.core.factories import ConversionCarrier
from pydantic2django.xmlschema.factory import XmlSchemaModelFactory
from pydantic2django.django.timescale.heuristics import TimescaleRole
from pydantic2django.django.models import Xml2DjangoBaseClass


def test_generator_uses_timescale_base_for_hypertables(monkeypatch):
    gen = XmlSchemaDjangoModelGenerator(schema_files=["dummy.xsd"], output_path="/tmp/out.py", app_label="tests")

    # Inject roles and call setup_django_model directly
    gen._timescale_roles = {"SamplesType": TimescaleRole.HYPERTABLE, "HeaderType": TimescaleRole.DIMENSION}

    src_hyper = XmlSchemaComplexType(name="SamplesType")
    src_dim = XmlSchemaComplexType(name="HeaderType")

    carrier_h = gen.setup_django_model(src_hyper)
    carrier_d = gen.setup_django_model(src_dim)

    assert carrier_h is not None
    assert carrier_d is not None
    # Base for hypertable is XmlTimescaleBase
    assert carrier_h.base_django_model is XmlTimescaleBase
    # Base for dimension falls back to generator default base
    assert carrier_d.base_django_model is gen.base_model_class
    assert carrier_d.base_django_model is Xml2DjangoBaseClass


def test_timescale_base_does_not_add_redundant_unique_constraint_on_pk():
    class StreamsType(XmlTimescaleBase):
        class Meta:  # type: ignore[misc]
            app_label = "test_app"

        time = models.DateTimeField()

    # Ensure NO redundant UniqueConstraint on id is added by the base
    print("Constraints on StreamsType:", StreamsType._meta.constraints)
    unique_constraints = [c for c in StreamsType._meta.constraints if isinstance(c, models.UniqueConstraint)]
    assert not any(list(getattr(c, "fields", [])) == ["id"] for c in unique_constraints), (
        "Timescale base should not emit UniqueConstraint on primary key 'id'."
    )


def test_invert_fk_dimension_to_hypertable_and_add_indexes():
    # Build a minimal schema with a dimension (DeviceStreamType) referencing a hypertable (StreamsType)
    schema = XmlSchemaDefinition(schema_location="mem.xsd", target_namespace="tns")

    streams = XmlSchemaComplexType(name="StreamsType")
    # Ensure time field exists on hypertable
    streams.add_element(XmlSchemaElement(name="time", base_type=XmlSchemaType.DATETIME))
    schema.add_complex_type(streams)

    device_stream = XmlSchemaComplexType(name="DeviceStreamType")
    # Single nested reference to StreamsType
    device_stream.add_element(XmlSchemaElement(name="Streams", base_type=None, type_name="StreamsType"))
    schema.add_complex_type(device_stream)

    # Prepare model factory and carriers
    factory = XmlSchemaModelFactory(app_label="test_app")

    # Hypertable carrier
    carrier_h = ConversionCarrier(
        source_model=streams,
        meta_app_label="test_app",
        base_django_model=XmlTimescaleBase,
        class_name_prefix="",
        strict=False,
    )
    # Dimension carrier
    carrier_d = ConversionCarrier(
        source_model=device_stream,
        meta_app_label="test_app",
        base_django_model=Xml2DjangoBaseClass,
        class_name_prefix="",
        strict=False,
    )
    # Timescale roles
    roles = {"StreamsType": TimescaleRole.HYPERTABLE, "DeviceStreamType": TimescaleRole.DIMENSION}
    carrier_h.context_data["_timescale_roles"] = roles
    carrier_d.context_data["_timescale_roles"] = roles

    # Make models
    factory.make_django_model(carrier_h)
    factory.make_django_model(carrier_d)

    carriers_by_name = {"StreamsType": carrier_h, "DeviceStreamType": carrier_d}
    print("Pending inverted on dimension:", carrier_d.context_data.get("_pending_inverted_fk"))
    factory.finalize_relationships(carriers_by_name, "test_app")
    print("Hypertable field defs:", carrier_h.django_field_definitions)
    print("Hypertable meta indexes:", carrier_h.context_data.get("meta_indexes"))

    # Assert no FK on dimension -> hypertable
    dim_defs = carrier_d.django_field_definitions
    assert not any("StreamsType" in v for v in dim_defs.values())

    # Assert FK exists on hypertable -> dimension with SET_NULL
    hyper_defs = carrier_h.django_field_definitions
    fk_def = hyper_defs.get("devicestreamtype")
    assert fk_def and "models.ForeignKey(" in fk_def and "to='test_app.DeviceStreamType'" in fk_def
    assert "on_delete=models.SET_NULL" in fk_def and "null=True" in fk_def and "blank=True" in fk_def

    # Assert indexes include FK and composite with -time
    meta_indexes = carrier_h.context_data.get("meta_indexes", [])
    assert any("fields=['devicestreamtype']" in idx for idx in meta_indexes)
    assert any("fields=['devicestreamtype', '-time']" in idx for idx in meta_indexes)
