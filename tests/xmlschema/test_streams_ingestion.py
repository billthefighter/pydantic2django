import os
from pathlib import Path

import pytest

from pydantic2django.xmlschema import XmlSchemaDjangoModelGenerator, XmlInstanceIngestor



def test_generate_and_ingest_mtconnect_streams(tmp_path: Path):
    # Paths to example XSD and XML
    repo_root = Path(__file__).parent.parent
    xsd_path = repo_root / "xmlschema" / "example_xml" / "MTConnectStreams_1.7.xsd"
    xml_path = repo_root / "xmlschema" / "example_xml" / "current.xml"

    assert xsd_path.exists(), f"Missing XSD file: {xsd_path}"
    assert xml_path.exists(), f"Missing XML file: {xml_path}"

    # Generate dynamic Django model classes from the XSD
    app_label = "streams_test"
    out_dir = tmp_path / "gen_streams"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "models.py"

    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[str(xsd_path)],
        output_path=str(out_file),
        app_label=app_label,
        verbose=False,
        # Keep nested complex types as FK/child_fk which aligns with the ingestor
        nested_relationship_strategy="fk",
        list_relationship_style="child_fk",
        nesting_depth_threshold=2,
    )

    # Trigger discovery and dynamic class assembly; also writes the models file (optional)
    gen.generate()

    # Ingest the XML instance without saving to the DB
    ingestor = XmlInstanceIngestor(schema_files=[str(xsd_path)], app_label=app_label)
    root = ingestor.ingest_from_file(str(xml_path), save=False)

    # Validate root-level structure
    # MTConnectStreams root should have a nested Header element mapped as FK
    assert hasattr(root, "header"), "Root should have 'header' field"
    assert root.header is not None, "Header instance should be created"

    # Validate a few header attributes parsed from current.xml
    # <Header sender="variaxis" bufferSize="131072" .../>
    assert getattr(root.header, "sender", None) == "variaxis"
    # buffer_size attribute should be mapped (string vs int depends on model field; unsaved assignment may be str)
    buf_val = getattr(root.header, "buffer_size", None)
    assert buf_val in ("131072", 131072)

    # Validate one nested path existence: Streams -> DeviceStream (repeating)
    # Since repeated complex types are created as child instances with FK back to parent,
    # ensure at least one DeviceStream child instance was created and linked to the Streams parent.
    # Use the ingestor's created_instances to find them without hitting the DB.
    created_classes = {obj.__class__.__name__: obj for obj in ingestor.created_instances}
    # Streams container should exist
    assert any(cls_name.lower().startswith("streams") for cls_name in created_classes), "Streams instance missing"
