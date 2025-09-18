from types import SimpleNamespace
from pathlib import Path

import lxml.etree as etree
import pytest

from pydantic2django.xmlschema import XmlInstanceIngestor
from pydantic2django.xmlschema.ingestor import SchemaSyncError
from pydantic2django.xmlschema.models import XmlSchemaComplexType, XmlSchemaAttribute, XmlSchemaElement
from pydantic2django.django.models import TimescaleModel


def _mk_ingestor(tmp_path: Path) -> XmlInstanceIngestor:
    xsd_content = (
        """
        <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="urn:test:Strict" xmlns="urn:test:Strict" elementFormDefault="qualified">
          <xs:complexType name="Dummy"/>
          <xs:element name="Root" type="Dummy"/>
        </xs:schema>
        """
    ).strip()
    xsd_path = tmp_path / "strict.xsd"
    xsd_path.write_text(xsd_content)
    return XmlInstanceIngestor(schema_files=[str(xsd_path)], app_label="strict_test", dynamic_model_fallback=True, strict=True)


def test_strict_unexpected_child_element_raises(tmp_path: Path) -> None:
    ingestor = _mk_ingestor(tmp_path)
    # Model with no special behavior
    class DummyModel:
        class _Meta:
            fields = []

        _meta = _Meta()

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    # Complex type expects no children
    ct = XmlSchemaComplexType(name="Dummy")
    elem = etree.Element("Root")
    etree.SubElement(elem, "Extra")

    with pytest.raises(SchemaSyncError):
        ingestor._build_instance_from_element(elem, ct, DummyModel, parent_instance=None)


def test_strict_unmapped_attribute_raises(tmp_path: Path) -> None:
    ingestor = _mk_ingestor(tmp_path)

    class DummyModel:
        class _Meta:
            fields = []

        _meta = _Meta()

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    # Complex type expects no attributes
    ct = XmlSchemaComplexType(name="Dummy")
    elem = etree.Element("Root")
    elem.set("unknownAttr", "value")

    with pytest.raises(SchemaSyncError):
        ingestor._build_instance_from_element(elem, ct, DummyModel, parent_instance=None)


def test_validate_models_detects_missing_model_and_missing_fields(tmp_path: Path) -> None:
    ingestor = _mk_ingestor(tmp_path)

    # Craft a CT that expects attribute 'serialNumber' and simple element 'label'
    ct = XmlSchemaComplexType(name="Device")
    ct.attributes["serialNumber"] = XmlSchemaAttribute(name="serialNumber")
    ct.elements.append(XmlSchemaElement(name="label"))

    # Bypass real discovery; inject our schema with just this CT
    ingestor._schemas = [SimpleNamespace(get_all_complex_types=lambda: [ct])]

    # Case 1: no model resolved
    ingestor._get_model_for_complex_type = lambda _ct: None  # type: ignore[assignment]
    issues = ingestor.validate_models(strict=False)
    assert issues and "<installed model not found>" in issues[0].missing_fields

    # Case 2: model resolved but missing expected fields
    class DeviceModel:
        class _Meta:
            fields = [SimpleNamespace(name="id")]  # Missing 'serial_number' and 'label'

        _meta = _Meta()

    ingestor._get_model_for_complex_type = lambda _ct: DeviceModel  # type: ignore[assignment]
    issues = ingestor.validate_models(strict=False)
    assert issues and ("serial_number" in issues[0].missing_fields)

    with pytest.raises(SchemaSyncError):
        ingestor.validate_models(strict=True)


def test_validate_models_timescale_requires_time(tmp_path: Path) -> None:
    ingestor = _mk_ingestor(tmp_path)

    ct = XmlSchemaComplexType(name="Metric")
    # No explicit fields; the check is Timescale-only time presence

    class TsMetricModel(TimescaleModel):
        class _Meta:
            fields = [SimpleNamespace(name="id")]  # Missing 'time'

        _meta = _Meta()

    ingestor._schemas = [SimpleNamespace(get_all_complex_types=lambda: [ct])]
    ingestor._get_model_for_complex_type = lambda _ct: TsMetricModel  # type: ignore[assignment]

    issues = ingestor.validate_models(strict=False)
    field_names = {f.name for f in TsMetricModel._meta.fields}
    assert issues and ("time" not in field_names) and ("Timescale model missing required 'time' field" in issues[0].problems)

    with pytest.raises(SchemaSyncError):
        ingestor.validate_models(strict=True)
