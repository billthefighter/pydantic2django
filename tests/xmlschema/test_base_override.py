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

import re
from pathlib import Path

import pytest

from pydantic2django.django.models import XmlTimescaleBase
from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator


@pytest.fixture(scope="module")
def simple_xsd_path() -> Path:
    return Path(__file__).parent / "fixtures" / "simple_schema.xsd"


def _assert_class_inherits_base(code: str, model_name: str, base_name: str):
    pattern = re.compile(rf"class\s+{re.escape(model_name)}\({re.escape(base_name)}\):")
    assert pattern.search(code), f"Expected class '{model_name}({base_name})' in generated code.\n{code}"


def test_override_base_class_via_constructor(simple_xsd_path, tmp_path):
    output_file = tmp_path / "models_constructor.py"
    app_label = "override_ctor"

    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[str(simple_xsd_path)],
        output_path=str(output_file),
        app_label=app_label,
        base_model_class=XmlTimescaleBase,
        verbose=False,
    )
    gen.generate()

    code = output_file.read_text()

    # Import should reference the overridden base
    assert "from pydantic2django.django.models import XmlTimescaleBase" in code
    # The simple schema defines BookType
    _assert_class_inherits_base(code, "BookType", "XmlTimescaleBase")


def test_override_base_class_via_assignment(simple_xsd_path, tmp_path):
    output_file = tmp_path / "models_assignment.py"
    app_label = "override_assign"

    gen = XmlSchemaDjangoModelGenerator(
        schema_files=[str(simple_xsd_path)],
        output_path=str(output_file),
        app_label=app_label,
        verbose=False,
    )
    # Override after construction
    gen.base_model_class = XmlTimescaleBase
    gen.generate()

    code = output_file.read_text()

    # Import should reference the overridden base
    assert "from pydantic2django.django.models import XmlTimescaleBase" in code
    _assert_class_inherits_base(code, "BookType", "XmlTimescaleBase")
