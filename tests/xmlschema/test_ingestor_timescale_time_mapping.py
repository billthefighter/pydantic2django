from types import SimpleNamespace
from pathlib import Path

import lxml.etree as etree

from pydantic2django.xmlschema import XmlInstanceIngestor
from pydantic2django.xmlschema.ingestor import TimeseriesTimestampMissingError
from pydantic2django.django.models import TimescaleModel
from pydantic2django.xmlschema.models import XmlSchemaComplexType, XmlSchemaAttribute


def test_timescale_maps_creation_time_to_time(tmp_path: Path) -> None:
    # Minimal XSD file just to satisfy ingestor initialization
    xsd_content = """
    <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="urn:test:Samples" xmlns="urn:test:Samples" elementFormDefault="qualified">
      <xs:complexType name="Dummy"/>
      <xs:element name="Root" type="Dummy"/>
    </xs:schema>
    """.strip()
    xsd_path = tmp_path / "samples.xsd"
    xsd_path.write_text(xsd_content)

    ingestor = XmlInstanceIngestor(
        schema_files=[str(xsd_path)], app_label="samples_timescale_test", dynamic_model_fallback=True
    )
    ingestor._save_objects = False

    # Stub Django-like model class with a 'time' field in _meta and an __init__ that sets attrs
    class DummyModel:
        class _Meta:
            # Fields list items need 'name' attributes
            fields = [SimpleNamespace(name="time"), SimpleNamespace(name="creation_time")]

        _meta = _Meta()

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    # Case 1: creationTime → creation_time → time
    ct1 = XmlSchemaComplexType(name="Dummy")
    ct1.attributes["creationTime"] = XmlSchemaAttribute(name="creationTime")
    elem1 = etree.Element("Root")
    elem1.set("creationTime", "2025-09-18T17:40:02Z")
    inst1 = ingestor._build_instance_from_element(elem1, ct1, DummyModel, parent_instance=None)
    assert getattr(inst1, "time", None) == "2025-09-18T17:40:02Z"

    # Case 2: timeStamp → time_stamp → time
    ct2 = XmlSchemaComplexType(name="Dummy")
    ct2.attributes["timeStamp"] = XmlSchemaAttribute(name="timeStamp")
    elem2 = etree.Element("Root")
    elem2.set("timeStamp", "2025-09-18T17:40:03Z")
    inst2 = ingestor._build_instance_from_element(elem2, ct2, DummyModel, parent_instance=None)
    assert getattr(inst2, "time", None) == "2025-09-18T17:40:03Z"

    # Case 3: time-stamp → time_stamp → time (hyphen normalization)
    ct3 = XmlSchemaComplexType(name="Dummy")
    ct3.attributes["time-stamp"] = XmlSchemaAttribute(name="time-stamp")
    elem3 = etree.Element("Root")
    elem3.set("time-stamp", "2025-09-18T17:40:04Z")
    inst3 = ingestor._build_instance_from_element(elem3, ct3, DummyModel, parent_instance=None)
    assert getattr(inst3, "time", None) == "2025-09-18T17:40:04Z"

    # Case 4a: missing timestamp attributes on NON-Timescale model → no error on save path
    ingestor._save_objects = True
    class NonTsModelWithManager(DummyModel):
        class objects:  # type: ignore[override]
            @staticmethod
            def create(**kwargs):
                return DummyModel(**kwargs)

    ct4a = XmlSchemaComplexType(name="Dummy")
    elem4a = etree.Element("Root")
    inst4a = ingestor._build_instance_from_element(elem4a, ct4a, NonTsModelWithManager, parent_instance=None)
    assert getattr(inst4a, "time", None) is None

    # Case 4b: missing timestamp attributes on Timescale model → error on save path
    class TsModelWithManager(TimescaleModel, DummyModel):
        class objects:  # type: ignore[override]
            @staticmethod
            def create(**kwargs):
                # Should not be reached because error is raised first
                raise AssertionError("create() should not be called when 'time' is missing")

    ct4b = XmlSchemaComplexType(name="Dummy")
    elem4b = etree.Element("Root")
    try:
        ingestor._build_instance_from_element(elem4b, ct4b, TsModelWithManager, parent_instance=None)
        assert False, "Expected TimeseriesTimestampMissingError to be raised"
    except TimeseriesTimestampMissingError as exc:
        assert "requires a non-null 'time' value" in str(exc)
