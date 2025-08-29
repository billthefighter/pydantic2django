import pytest

from django.db import models

from pydantic2django.django.models import Xml2DjangoBaseClass
from pydantic2django.django.timescale.bases import XmlTimescaleBase
from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator
from pydantic2django.xmlschema.models import XmlSchemaComplexType


def test_generator_uses_timescale_base_for_hypertables(monkeypatch):
    gen = XmlSchemaDjangoModelGenerator(schema_files=["dummy.xsd"], output_path="/tmp/out.py", app_label="tests")

    # Inject roles and call setup_django_model directly
    gen._timescale_roles = {"SamplesType": "hypertable", "HeaderType": "dimension"}

    src_hyper = XmlSchemaComplexType(name="SamplesType")
    src_dim = XmlSchemaComplexType(name="HeaderType")

    carrier_h = gen.setup_django_model(src_hyper)
    carrier_d = gen.setup_django_model(src_dim)

    assert carrier_h is not None
    assert carrier_d is not None
    # Base for hypertable is XmlTimescaleBase
    assert carrier_h.base_django_model is XmlTimescaleBase
    # Base for dimension falls back to generator default base
    assert carrier_d.base_django_model is gen.base_model_class
    assert carrier_d.base_django_model is Xml2DjangoBaseClass
