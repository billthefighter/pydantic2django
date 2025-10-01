from __future__ import annotations

import pathlib

import pytest


def test_attribute_group_fields_present_on_observations(tmp_path: pathlib.Path):
    """
    Expectations (MTConnect Streams 2.4):
    - Observation elements (e.g., Position, Temperature) include common attributes
      via attribute groups such as timestamp, subType, dataItemId.
    - After generation, concrete observation models should expose these as fields.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    # Paths used by the example generator
    xsd_streams = repo_root / "examples/mtconnect_example/schemas/MTConnectStreams_2.4.xsd"
    assert xsd_streams.exists(), "Streams XSD not found; run the example downloader first."

    # Generate fresh models to a temp file using the built-in generator
    # Then import the generated module
    import sys
    from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    out_file = tmp_path / "models.py"
    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[str(xsd_streams)],
        output_path=str(out_file),
        app_label="mtconnect_streams_v24",
        verbose=False,
    )
    gen.generate()

    import importlib.util
    spec = importlib.util.spec_from_file_location("mtc_streams_models_tmp", str(out_file))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    # Heuristic: at least one observation-like model should have timestamp/subtype fields
    # This will fail until attributeGroup and substitutionGroup are implemented.
    candidate_class_names = [
        name for name in dir(module) if name.endswith("DataSetType") or name.endswith("TableEntryType") or name.endswith("Type")
    ]
    assert candidate_class_names, "No candidate models discovered in generated streams models."

    found_with_attrs = False
    for name in candidate_class_names:
        cls = getattr(module, name, None)
        if not hasattr(cls, "_meta"):
            continue
        field_names = {f.name for f in getattr(cls._meta, "fields", [])}
        if {"timestamp", "sub_type"}.issubset(field_names) or {"timestamp", "subtype"}.issubset(field_names):
            found_with_attrs = True
            break

    assert found_with_attrs, (
        "Expected at least one observation model to include 'timestamp' and 'subType' fields from attribute groups."
    )


def test_substitution_group_creates_concrete_observation_models(tmp_path: pathlib.Path):
    """
    Expectations:
    - Substitution groups under Samples/Events/Condition should produce concrete
      models (e.g., Position, Temperature) rather than a single 'sample' field.
    - There should be distinct Django models for several observation names.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    # Generate fresh models to temp file
    import sys
    from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator
    xsd_streams = repo_root / "examples/mtconnect_example/schemas/MTConnectStreams_2.4.xsd"
    assert xsd_streams.exists(), "Streams XSD not found; run the example downloader first."
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    out_file = tmp_path / "models.py"
    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[str(xsd_streams)],
        output_path=str(out_file),
        app_label="mtconnect_streams_v24",
        verbose=False,
    )
    gen.generate()

    import importlib.util
    spec = importlib.util.spec_from_file_location("mtc_streams_models_tmp", str(out_file))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    # Expect presence of common concrete observation models
    required = [
        "PositionDataSetType",  # typical sample type in streams
        "TemperatureDataSetType",
        "VoltageDataSetType",
    ]

    missing = [name for name in required if not hasattr(module, name)]
    assert not missing, f"Missing concrete observation models due to unexpanded substitution groups: {missing}"
