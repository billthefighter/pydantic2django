"""Tests for GenericForeignKey GenericEntry generation via Pydantic path."""
from pathlib import Path

import pytest


def test_pydantic_gfk_generates_generic_entry_and_relation(tmp_path: Path):
    """Enabling GFK with policy=all_nested should emit GenericEntry and a GenericRelation on parent."""
    # Create a temporary package with Pydantic models
    pkg_dir = tmp_path / "pyd_gfk_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "models.py").write_text(
        """
from typing import Union
from pydantic import BaseModel


class A(BaseModel):
    v: int


class B(BaseModel):
    v: str


class Parent(BaseModel):
    items: list[Union[A, B]]
        """
    )

    # Ensure the temporary package can be imported by discovery
    import sys

    sys.path.insert(0, str(tmp_path))

    # Generate models
    from pydantic2django.pydantic.generator import StaticPydanticModelGenerator

    output_file = tmp_path / "models_gfk.py"
    gen = StaticPydanticModelGenerator(
        output_path=str(output_file),
        packages=["pyd_gfk_pkg"],
        app_label="tests",
        enable_gfk=True,
        gfk_policy="all_nested",
        gfk_value_mode="typed_columns",
    )
    gen.generate()

    code = output_file.read_text()

    assert "class GenericEntry(" in code
    assert "entries = GenericRelation('GenericEntry', related_query_name='entries')" in code
    # Parent model class name should be prefixed with default 'Django'
    assert "class DjangoParent(" in code
