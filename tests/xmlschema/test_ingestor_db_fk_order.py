import pytest
from django.apps import apps
from django.db import IntegrityError

from pydantic2django.xmlschema import XmlInstanceIngestor


@pytest.mark.django_db
def test_ingest_saves_parent_before_children_fk_not_null():
    # Use the same fixture schema as the unsaved test; models exist via migrations
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

    # Before the fix, creating ItemType children without the parent FK set would raise IntegrityError
    # due to the non-null constraint on tests.ItemType.parenttype (set in migration 0009).
    # After the fix, the ingestor sets the FK at creation time and no error is raised.
    try:
        ingestor.ingest_from_string(xml, save=True)
    except IntegrityError:
        # Explicitly fail with a clear message to surface the ordering bug
        pytest.fail("Child rows were inserted before setting required parent FK (ordering bug)")

    # Sanity-check: two ItemType records exist and are linked to the parent
    ParentType = apps.get_model("tests", "ParentType")
    ItemType = apps.get_model("tests", "ItemType")

    assert ParentType.objects.count() == 1
    assert ItemType.objects.count() == 2
    parent = ParentType.objects.first()
    assert parent is not None
    assert ItemType.objects.filter(parenttype=parent).count() == 2
