import pytest

from django.apps import apps


@pytest.mark.django_db
def test_ingestor_prefers_installed_models_over_generated(monkeypatch):
    # Arrange: ensure installed concrete models exist
    ParentType = apps.get_model("tests", "ParentType")
    assert ParentType is not None

    # Prepare XML and XSD fixtures from the tests app
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

    # Create a fake generated stand-in class with the same name but no manager
    FakeParentType = type("ParentType", (), {})

    # Inject a temporary generated-models registry containing the fake class
    import pydantic2django.xmlschema as xmlschema_mod

    monkeypatch.setattr(xmlschema_mod, "_GENERATED_MODELS_REGISTRY", {"tests": {"ParentType": FakeParentType}})

    # Act: ingest with save=True; should use installed model (with .objects), not the fake
    from pydantic2django.xmlschema import XmlInstanceIngestor

    ingestor = XmlInstanceIngestor(schema_files=[xsd_path], app_label="tests")
    root = ingestor.ingest_from_string(xml, save=True)

    # Assert: root is instance of the installed concrete model, and save worked
    assert isinstance(root, ParentType)
    assert ParentType.objects.count() == 1



@pytest.mark.django_db
def test_ingestor_resolves_nested_types_to_installed_models(monkeypatch):
    # Arrange
    ParentType = apps.get_model("tests", "ParentType")
    ChildType = apps.get_model("tests", "ChildType")
    ItemType = apps.get_model("tests", "ItemType")
    assert ParentType is not None and ChildType is not None and ItemType is not None

    xsd_path = apps.get_app_config("tests").path + "/xmlschema/fixtures/nested_schema.xsd"
    xml = (
        """
        <ParentType xmlns="http://www.example.com/nested">
            <owner>Bob</owner>
            <child code="C2"><value>V2</value></child>
            <items><price>1.00</price></items>
            <items><price>2.00</price></items>
        </ParentType>
        """
    )

    # Create a fake generated stand-in classes to ensure installed models are preferred
    FakeParentType = type("ParentType", (), {})
    FakeChildType = type("ChildType", (), {})
    FakeItemType = type("ItemType", (), {})

    import pydantic2django.xmlschema as xmlschema_mod

    monkeypatch.setattr(
        xmlschema_mod,
        "_GENERATED_MODELS_REGISTRY",
        {"tests": {"ParentType": FakeParentType, "ChildType": FakeChildType, "ItemType": FakeItemType}},
    )

    # Act
    from pydantic2django.xmlschema import XmlInstanceIngestor

    ingestor = XmlInstanceIngestor(schema_files=[xsd_path], app_label="tests")
    root = ingestor.ingest_from_string(xml, save=True)

    # Assert root and nested instances use installed models and were saved
    assert isinstance(root, ParentType)
    assert isinstance(root.child, ChildType)
    assert ParentType.objects.count() >= 1
    assert ChildType.objects.count() >= 1
    assert ItemType.objects.count() == 2
    # Items back-link should point to root
    for item in ItemType.objects.all():
        assert item.parenttype_id == root.id
