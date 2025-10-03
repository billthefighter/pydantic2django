from pathlib import Path

from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator
from pydantic2django.xmlschema.ingestor import XmlInstanceIngestor


def test_streams_gfk_ingestion_skips_observation_children(tmp_path: Path):
    # Use bundled Streams 1.7 XSD and example XML
    xsd_path = Path(__file__).parent / "example_xml" / "MTConnectStreams_1.7.xsd"
    xml_path = Path(__file__).parent / "example_xml" / "current.xml"
    out_file = tmp_path / "models.py"

    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[str(xsd_path)],
        output_path=str(out_file),
        app_label="streams_gfk",
        enable_gfk=True,
        gfk_policy="all_nested",
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
    )
    gen.generate()

    ing = XmlInstanceIngestor(schema_files=[str(xsd_path)], app_label="streams_gfk", dynamic_model_fallback=True)
    root = ing.ingest_from_file(str(xml_path), save=False)

    # Root should exist
    assert root is not None
    created = [obj.__class__.__name__ for obj in ing.created_instances]

    # Containers like StreamsType / ComponentStreamType should be present
    assert any("Streams" in n for n in created) or any("Stream" in n for n in created)

    # Observation-like concrete children should not be instantiated under GFK
    # Heuristic: members often end with "EntryType" or contain common observation names (Angle, Position, etc.)
    assert all(not n.endswith("EntryType") for n in created)
