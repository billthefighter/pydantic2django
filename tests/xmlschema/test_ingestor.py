import pytest

from django.apps import apps
from django.db import models, connection
from pydantic2django.xmlschema import XmlInstanceIngestor
from tests.fixtures.fixtures import ParentType, ChildType, ItemType


def test_ingest_nested_xml(tmp_path):
    # Ensure models are discoverable via app registry
    assert apps.get_model("tests", "ParentType") is ParentType
    assert apps.get_model("tests", "ChildType") is ChildType
    assert apps.get_model("tests", "ItemType") is ItemType

    # Ensure DB tables exist for our test models
    # Avoid database writes in this test; validate ingestion logic with unsaved instances

    # Use provided nested schema fixture
    xsd_path = apps.get_app_config("tests").path + "/xmlschema/fixtures/nested_schema.xsd"

    xml = (
        """
        <ParentType xmlns="http://www.example.com/nested">
            <owner>Alice</owner>
            <child code="C1"><value>V1</value></child>
            <items><price>10.50</price></items>
            <items><price>20.00</price></items>
        </ParentType>
        """
    )

    ingestor = XmlInstanceIngestor(schema_files=[xsd_path], app_label="tests")
    root = ingestor.ingest_from_string(xml, save=False)

    # Validate root
    assert isinstance(root, ParentType)
    assert root.owner == "Alice"
    assert root.child is not None
    assert root.child.value == "V1"
    assert root.child.code == "C1"

    # Validate children created and linked via child_fk (unsaved instances)
    created_items = [obj for obj in ingestor.created_instances if isinstance(obj, ItemType)]
    assert len(created_items) == 2
    for it in created_items:
        assert hasattr(it, "parenttype")
        assert it.parenttype is root
