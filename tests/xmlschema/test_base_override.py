"""
Tests for overriding the base model class in generated code.

Covers the guidance in docs/how_to_use/integrate_into_django_app.md under
"Override the generated base class".
"""

from pathlib import Path

import pydantic2django.django.models as dj_models
from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator


def _simple_xsd() -> Path:
    return Path(__file__).parent / "fixtures" / "simple_schema.xsd"


def test_xmlschema_base_override_via_attribute(tmp_path):
    output_file = tmp_path / "models.py"
    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[str(_simple_xsd())],
        output_path=str(output_file),
        app_label="test_app",
        verbose=True,
    )
    # Override after construction (as documented)
    # Resolve at runtime to avoid import/type issues in environments without Timescale
    XmlTimescaleBase = getattr(dj_models, "XmlTimescaleBase")
    gen.base_model_class = XmlTimescaleBase
    gen.generate()

    code = output_file.read_text()
    assert "from pydantic2django.django.models import XmlTimescaleBase" in code
    assert "class BookType(XmlTimescaleBase):" in code
    assert "class BookType(Xml2DjangoBaseClass):" not in code

