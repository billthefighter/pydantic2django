import pytest
from django.db import models


def test_pydantic_generator_uses_pydantic_base():
    from typed2django.django.models import Pydantic2DjangoBaseClass
    from pydantic2django.pydantic.generator import StaticPydanticModelGenerator

    gen = StaticPydanticModelGenerator(output_path="/tmp/models.py", app_label="app", packages=["tests"])
    assert issubclass(gen.base_model_class, models.Model)
    assert gen.base_model_class is Pydantic2DjangoBaseClass


def test_dataclass_generator_uses_dataclass_base():
    from typed2django.django.models import Dataclass2DjangoBaseClass
    from pydantic2django.dataclass.generator import DataclassDjangoModelGenerator

    gen = DataclassDjangoModelGenerator(
        output_path="/tmp/models.py",
        app_label="app",
        filter_function=None,
        verbose=False,
        packages=[],
    )
    assert issubclass(gen.base_model_class, models.Model)
    assert gen.base_model_class is Dataclass2DjangoBaseClass


def test_xmlschema_generator_uses_xml_base():
    from typed2django.django.models import Xml2DjangoBaseClass
    from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator

    gen = XmlSchemaDjangoModelGenerator(schema_files=[], output_path="/tmp/models.py", app_label="xml")
    assert issubclass(gen.base_model_class, models.Model)
    assert gen.base_model_class is Xml2DjangoBaseClass
