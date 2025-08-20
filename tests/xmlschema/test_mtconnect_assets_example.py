"""
MTConnect Assets 2.4 schema integration tests.

Validates that generated Django models:
- use correctly cased relation targets (app_label.ModelName)
- include null=True, blank=True whenever on_delete=SET_NULL is used
- do not emit max_length on IntegerField
- emit enums and choices properly
- avoid plain 'id' non-PK field conflicts (rename to xml_id)
- include RegexValidator import when used
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator


def _assert_field(code: str, model_name: str, field_name: str, expected_type: str, expected_kwargs: dict[str, str]):
    # Find class block first to scope search (optional robustness)
    class_pattern = re.compile(rf"class\s+{re.escape(model_name)}\(.*?\):([\s\S]*?)(?:\nclass\s+|\Z)")
    class_match = class_pattern.search(code)
    assert class_match, f"Model class {model_name} not found."
    class_body = class_match.group(1)

    # Find field definition line
    field_pattern = re.compile(rf"^\s*{re.escape(field_name)}\s*=\s*models\.{re.escape(expected_type)}\((.*)\)", re.MULTILINE)
    m = field_pattern.search(class_body)
    assert m, f"Field '{field_name}: models.{expected_type}' not found in {model_name}."
    kwargs_str = m.group(1)

    for key, value in expected_kwargs.items():
        # Accept both quoted and unquoted representations
        val_pat = f"(?:{re.escape(value)}|'" + re.escape(value) + "'|\"" + re.escape(value) + "\")"
        pat = re.compile(rf"\b{re.escape(key)}\s*=\s*{val_pat}")
        assert pat.search(kwargs_str), f"Expected kwarg {key}={value} in {model_name}.{field_name}. Got: {kwargs_str}"



def test_mtconnect_assets_v24_generation(tmp_path: Path):
    xsd_path = Path(__file__).parent / "example_xml" / "MTConnectAssets_2.4.xsd"
    assert xsd_path.exists(), "Example MTConnectAssets_2.4.xsd missing in tests/xmlschema/example_xml/"

    out_file = tmp_path / "models.py"
    app_label = "mtconnect_assets_v24"

    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[str(xsd_path)],
        output_path=str(out_file),
        app_label=app_label,
        verbose=False,
        # Leave defaults for nested_relationship_strategy / list_relationship_style
    )
    gen.generate()

    code = out_file.read_text()

    # 1) Relation casing: ensure all 'to' kwargs reference 'app_label.ModelName' (ModelName is class-cased)
    # Must not have any lowercased model refs like mtconnect_assets_v24.dataitemdescriptiontype
    assert not re.search(r"to='mtconnect_assets_v24\.[a-z]", code), "Found a lowercased lazy ref in 'to='"
    # And should have at least some correctly cased relations present
    assert re.search(r"to='mtconnect_assets_v24\.[A-Z]\w+'", code), "No correctly-cased relation targets found"

    # 2) SET_NULL implies null/blank
    # 2) SET_NULL implies null/blank for all fields that use it
    for m in re.finditer(r"^\s*\w+\s*=\s*models\.(ForeignKey|OneToOneField)\(([^\)]*)\)", code, re.MULTILINE):
        kwargs_str = m.group(2)
        if "on_delete=models.SET_NULL" in kwargs_str:
            assert "null=True" in kwargs_str and "blank=True" in kwargs_str, f"Missing null/blank on SET_NULL: {kwargs_str}"

    # 3) No max_length on IntegerField (use any representative integer field if present)
    # ProgramToolNumber appears in several types
    int_field_line = re.search(r"ProgramToolNumber\s*=\s*models\.IntegerField\(([^\)]*)\)", code)
    if int_field_line:  # If present in this schema version
        assert "max_length" not in int_field_line.group(1)

    # 4) Enum and choices usage: at least one TextChoices enum class and at least one usage
    assert re.search(r"class\s+[A-Z]\w*\(models\.TextChoices\):", code), "Expected at least one TextChoices enum"
    assert re.search(r"choices=[A-Z]\w*\.choices", code), "Expected at least one field using <Enum>.choices"

    # 5) id handling – ensure non-PK 'id' is renamed if present; RawMaterialMaterialType.xml_id expected
    # We don't rely on the exact field set for every schema; check if RawMaterialMaterialType exists then assert xml_id
    if "class RawMaterialMaterialType(" in code:
        assert re.search(r"\n\s*xml_id\s*=\s*models\.CharField\(", code), "Expected xml_id CharField in RawMaterialMaterialType"

    # 6) RegexValidator import when used (indices pattern in CuttingItemType)
    if re.search(r"RegexValidator\(", code):
        assert re.search(r"from\s+django\.core\.validators\s+import\s+RegexValidator", code), "Missing RegexValidator import"

    # 7) Guard: ensure no lowercase lazy refs appear (e.g. '.dataitemdescriptiontype')
    assert ".dataitemdescriptiontype'" not in code
    assert ".celldefinitionstype'" not in code
