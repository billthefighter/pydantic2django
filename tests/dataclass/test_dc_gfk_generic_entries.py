"""Tests for GenericForeignKey GenericEntry generation via Dataclass path."""
from pathlib import Path


def test_dataclass_gfk_generates_generic_entry_and_relation(tmp_path: Path):
    pkg_dir = tmp_path / "dc_gfk_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "models.py").write_text(
        """
from dataclasses import dataclass
from typing import Union


@dataclass
class A:
    v: int


@dataclass
class B:
    v: str


@dataclass
class Parent:
    items: list[Union[A, B]]
        """
    )

    import sys

    sys.path.insert(0, str(tmp_path))

    from pydantic2django.dataclass.generator import DataclassDjangoModelGenerator

    output_file = tmp_path / "models_gfk.py"
    gen = DataclassDjangoModelGenerator(
        output_path=str(output_file),
        packages=["dc_gfk_pkg"],
        app_label="tests",
        filter_function=None,
        verbose=False,
        enable_gfk=True,
        gfk_policy="all_nested",
        gfk_value_mode="typed_columns",
    )
    gen.generate()

    code = output_file.read_text()

    assert "class GenericEntry(" in code
    assert "entries = GenericRelation('GenericEntry', related_query_name='entries')" in code
    # Dataclass parent model name should be prefixed with default 'Django'
    assert "class DjangoParent(" in code
