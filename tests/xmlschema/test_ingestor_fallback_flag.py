import pytest

from django.apps import apps


@pytest.mark.django_db
def test_default_uses_installed_models():
    # With installed models (tests app), disabling fallback should still work
    xsd_path = apps.get_app_config("tests").path + "/xmlschema/fixtures/nested_schema.xsd"

    from pydantic2django.xmlschema import XmlInstanceIngestor

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

    # Default dynamic_model_fallback is False; should use installed models
    ing = XmlInstanceIngestor(schema_files=[xsd_path], app_label="tests")
    root = ing.ingest_from_string(xml, save=True)

    ParentType = apps.get_model("tests", "ParentType")
    assert isinstance(root, ParentType)
    assert ParentType.objects.count() == 1


def test_default_raises_when_no_installed_models(tmp_path):
    # For a generated-only app_label, disabling fallback should raise a clear error
    # Use the streams example which generates dynamic classes under a unique app_label
    repo_root = apps.get_app_config("tests").path.rsplit("/tests", 1)[0]
    xsd_path = repo_root + "/tests/xmlschema/example_xml/MTConnectStreams_1.7.xsd"

    from pydantic2django.xmlschema import XmlSchemaDjangoModelGenerator, XmlInstanceIngestor
    from pydantic2django.xmlschema.ingestor import ModelResolutionError

    app_label = "streams_test_no_install"

    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[xsd_path],
        output_path=str(tmp_path / "models.py"),
        app_label=app_label,
        verbose=False,
    )
    gen.generate()  # populate generated registry only

    # Default (False) should raise when no installed models exist
    ing = XmlInstanceIngestor(schema_files=[xsd_path], app_label=app_label)

    xml = (
        """
        <MTConnectStreams xmlns="urn:mtconnect.org:MTConnectStreams:1.7">
            <Header/>
            <Streams/>
        </MTConnectStreams>
        """
    )

    with pytest.raises(ModelResolutionError) as ei:
        ing.ingest_from_string(xml, save=False)

    assert app_label in str(ei.value)


def test_explicit_true_allows_dynamic_fallback(tmp_path):
    # When explicitly enabled, fallback to generated stand-ins should work without raising
    repo_root = apps.get_app_config("tests").path.rsplit("/tests", 1)[0]
    xsd_path = repo_root + "/tests/xmlschema/example_xml/MTConnectStreams_1.7.xsd"

    from pydantic2django.xmlschema import XmlSchemaDjangoModelGenerator, XmlInstanceIngestor

    app_label = "streams_test_explicit_fallback"

    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[xsd_path],
        output_path=str(tmp_path / "models.py"),
        app_label=app_label,
        verbose=False,
    )
    gen.generate()

    ing = XmlInstanceIngestor(schema_files=[xsd_path], app_label=app_label, dynamic_model_fallback=True)

    xml = (
        """
        <MTConnectStreams xmlns="urn:mtconnect.org:MTConnectStreams:1.7">
            <Header/>
            <Streams/>
        </MTConnectStreams>
        """
    )

    # Should not raise
    ing.ingest_from_string(xml, save=False)
