"""
Functional tests for Django model generation from a comprehensive XML Schema,
focusing on relationships, data types, and constraints.
"""

import pytest
import re
from pathlib import Path
import logging

from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator
from .test_generation import assert_field_definition_xml

# Configure logging to see parser output during tests
parser_logger = logging.getLogger("pydantic2django.xmlschema.parser")
parser_logger.setLevel(logging.DEBUG)
if not parser_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    parser_logger.addHandler(handler)
    parser_logger.propagate = False

# Configure logging for discovery
discovery_logger = logging.getLogger("pydantic2django.xmlschema.discovery")
discovery_logger.setLevel(logging.INFO)
if not discovery_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    discovery_logger.addHandler(handler)
    discovery_logger.propagate = False


@pytest.fixture(scope="module")
def comprehensive_xsd_path() -> Path:
    """Provides the path to the comprehensive test XSD file."""
    return Path(__file__).parent / "fixtures" / "comprehensive_schema.xsd"


@pytest.fixture(scope="module")
def generated_code(comprehensive_xsd_path, tmp_path_factory) -> str:
    """
    Runs the generator for the comprehensive schema once per test module
    and returns the generated code as a string.
    """
    output_file = tmp_path_factory.mktemp("generated") / "models.py"
    app_label = "test_app"

    generator = XmlSchemaDjangoModelGenerator(
        schema_files=[str(comprehensive_xsd_path)],
        output_path=str(output_file),
        app_label=app_label,
    )
    generator.generate()

    assert output_file.exists()
    code = output_file.read_text()
    print(f"\\n--- Generated Code from comprehensive_schema.xsd ---")
    print(code)
    print("----------------------------------------------------")
    return code


class TestComprehensiveSchemaGeneration:
    """Groups tests for the comprehensive_schema.xsd."""

    def test_model_classes_generated(self, generated_code: str):
        """Tests that both AuthorType and BookType models are generated."""
        assert "class AuthorType(Xml2DjangoBaseClass):" in generated_code
        assert "class BookType(Xml2DjangoBaseClass):" in generated_code

    def test_meta_class_generated(self, generated_code: str):
        """Tests that the Meta class with app_label is present in both models."""
        assert "class Meta:" in generated_code
        assert "app_label = 'test_app'" in generated_code
        # Ensure it appears for both models
        assert generated_code.count("app_label = 'test_app'") == 2

    # --- Author Model Tests ---

    @pytest.mark.parametrize(
        "field_name, expected_type, expected_kwargs, absent_kwargs",
        [
            ("author_id", "CharField", {"max_length": "255", "primary_key": "True"}, ["null", "blank"]),
            ("name", "CharField", {"max_length": "255"}, ["null", "blank"]),
            ("email", "CharField", {"max_length": "255", "null": "True", "blank": "True"}, []),
            ("bio", "TextField", {"null": "True", "blank": "True"}, []),
            (
                "status",
                "CharField",
                {
                    "max_length": "8", # length of 'deceased'
                    "choices": "Status.choices",
                    "default": "Status.ACTIVE",
                },
                ["null", "blank"],
            ),
        ],
        ids=["author_id-pk", "name-required", "email-optional-pattern", "bio-optional-text", "status-enum-default"],
    )
    def test_author_model_fields(self, generated_code: str, field_name: str, expected_type: str, expected_kwargs: dict, absent_kwargs: list):
        """Parameterized test for fields in the generated AuthorType model."""
        assert_field_definition_xml(
            generated_code, field_name, expected_type, expected_kwargs, absent_kwargs, model_name="AuthorType"
        )

    # @pytest.mark.xfail(reason="Validators from simpleType restrictions not yet implemented")
    def test_author_email_field_has_validator(self, generated_code: str):
        """Tests that the email field has a RegexValidator from the simpleType pattern."""
        # This test needs to be more robust, checking for the actual validator object.
        # For now, we check the generated code string.
        assert_field_definition_xml(
            generated_code,
            "email",
            "CharField",
            {"validators": "[RegexValidator"},  # Check if validators list starts
            model_name="AuthorType",
        )

    # @pytest.mark.xfail(reason="Enum class generation from simpleType not yet implemented")
    def test_author_status_enum_class_generated(self, generated_code: str):
        """Tests that the TextChoices enum for AuthorStatus is generated correctly."""
        assert "class Status(models.TextChoices):" in generated_code
        assert 'ACTIVE = "active", "Active"' in generated_code

    # --- Book Model Tests ---

    @pytest.mark.parametrize(
        "field_name, expected_type, expected_kwargs, absent_kwargs",
        [
            ("isbn", "CharField", {"max_length": "255", "primary_key": "True"}, ["null", "blank"]),
            # ("title", "CharField", {"max_length": "255"}, ["null", "blank"]),
            # ("publication_date", "IntegerField", {}, ["null", "blank"]), # xs:gYear -> IntegerField
            # ("pages", "PositiveIntegerField", {}, ["null", "blank"]),
            # (
            #     "genre",
            #     "CharField",
            #     {
            #         "max_length": "15", # length of 'science-fiction'
            #         "choices": "Genre.choices",
            #         "default": "Genre.FICTION",
            #     },
            #     ["null", "blank"],
            # ),
            # ("summary", "TextField", {"null": "True", "blank": "True"}, []),
        ],
        ids=["isbn-pk"], #, "title-required", "publication_date-gYear", "pages-positiveInt", "genre-enum-default", "summary-nillable"],
    )
    def test_book_model_fields(self, generated_code: str, field_name: str, expected_type: str, expected_kwargs: dict, absent_kwargs: list):
        """Parameterized test for fields in the generated BookType model."""
        assert_field_definition_xml(
            generated_code, field_name, expected_type, expected_kwargs, absent_kwargs, model_name="BookType"
        )

    # @pytest.mark.xfail(reason="Enum class generation from simpleType not yet implemented")
    def test_book_genre_enum_class_generated(self, generated_code: str):
        """Tests that the TextChoices enum for BookGenre is generated correctly."""
        assert "class Genre(models.TextChoices):" in generated_code
        assert 'FANTASY = "fantasy", "Fantasy"' in generated_code

    # @pytest.mark.xfail(reason="keyref relationship to ForeignKey not yet implemented")
    def test_foreign_key_relationship_from_book_to_author(self, generated_code: str):
        """Tests that the xs:keyref has created a ForeignKey field."""
        expected_kwargs = {
            "to": "'test_app.AuthorType'",
            "on_delete": "models.CASCADE",
            "related_name": "'books'",
        }
        assert_field_definition_xml(
            generated_code,
            "author_ref",
            "ForeignKey",
            expected_kwargs,
            model_name="BookType",
        )
