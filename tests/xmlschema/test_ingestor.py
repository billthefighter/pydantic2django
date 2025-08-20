import pytest

from django.apps import apps
from django.db import models, connection

from pydantic2django.django.models import Xml2DjangoBaseClass
from pydantic2django.xmlschema import XmlInstanceIngestor


class ParentType(Xml2DjangoBaseClass):
    owner = models.CharField(max_length=255)
    # FK to child will be on parent for single nested complex element
    child = models.ForeignKey("tests.ChildType", on_delete=models.SET_NULL, null=True, blank=True)


class ChildType(Xml2DjangoBaseClass):
    value = models.CharField(max_length=255)
    code = models.CharField(max_length=255, null=True, blank=True)


class ItemType(Xml2DjangoBaseClass):
    price = models.DecimalField(max_digits=10, decimal_places=2)
    # For repeated nested complex elements, generator defaults to child_fk strategy
    parenttype = models.ForeignKey("tests.ParentType", on_delete=models.CASCADE, related_name="items", null=True)


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
