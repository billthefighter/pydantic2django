"""Unit test for typed value parsing into GenericEntry kwargs builder."""
from dataclasses import dataclass


class _F:
    def __init__(self, name: str):
        self.name = name


class DummyMeta:
    def __init__(self, field_names: list[str]):
        self.fields = [_F(n) for n in field_names]


class DummyGenericEntry:
    def __init__(self, field_names: list[str]):
        self._meta = DummyMeta(field_names)


def test_build_generic_entry_kwargs_typed_columns_parsing():
    # Build a dummy GenericEntry that declares typed columns
    GE = DummyGenericEntry(["id", "text_value", "num_value", "time_value"])  # _meta.fields with names

    # Minimal element stub
    class E:
        def __init__(self):
            self.attrib = {"unit": "C"}
            self.text = "2024-01-01T00:00:00Z"

    from pydantic2django.xmlschema.ingestor import XmlInstanceIngestor

    kwargs = XmlInstanceIngestor._build_generic_entry_kwargs(
        instance=object(),
        el_name="Temperature",
        target_type_name="TempType",
        child_elem=E(),
        GenericEntry=GE,
        order_index=5,
    )

    assert kwargs["element_qname"] == "Temperature"
    assert kwargs["type_qname"] == "TempType"
    assert kwargs["order_index"] == 5
    # Typed extraction
    assert kwargs["text_value"] == "2024-01-01T00:00:00Z"
    assert kwargs.get("num_value") is None  # not numeric
    assert kwargs.get("time_value") is not None
    # Attributes preserved
    assert kwargs["attrs_json"]["unit"] == "C"
