"""
Tests for migration detection functionality.
"""
import pytest
from django.db import models
from pydantic import BaseModel

from pydantic2django import make_django_model
from pydantic2django.migrations import check_model_migrations, get_migration_operations


@pytest.mark.django_db
def test_model_without_migrations():
    """Test migration detection for a model without existing migrations."""

    class User(BaseModel):
        name: str
        age: int

    django_model, operations = make_django_model(User, check_migrations=False)

    # Since we're not checking migrations, operations should be None
    assert operations is None

    # Now check migrations manually
    _, operations = check_model_migrations(django_model)
    assert operations is not None
    assert any("CreateModel" in op for op in operations)


@pytest.mark.django_db
def test_model_with_matching_migrations(db):
    """Test migration detection for a model that matches its migrations."""

    # First create and migrate a model
    class Product(BaseModel):
        name: str
        price: float

    django_model, operations = make_django_model(Product, check_migrations=False)

    # Check migrations manually
    _, operations = check_model_migrations(django_model)
    assert operations is not None
    assert any("CreateModel" in op for op in operations)


@pytest.mark.django_db
def test_model_with_field_changes(db):
    """Test migration detection when model fields change."""

    # First create and migrate a model
    class Article(BaseModel):
        title: str

    django_model, _ = make_django_model(Article, check_migrations=False)

    # Check migrations manually
    _, operations = check_model_migrations(django_model)
    assert operations is not None
    assert any("CreateModel" in op for op in operations)

    # Now modify the model with a new field
    class ArticleWithContent(BaseModel):
        title: str
        content: str  # New field

    django_model2, _ = make_django_model(ArticleWithContent, check_migrations=False)

    # Check migrations manually
    _, operations2 = check_model_migrations(django_model2)
    assert operations2 is not None
    assert any("AddField" in op for op in operations2)


def test_migration_check_disabled():
    """Test that migration checking can be disabled."""

    class Comment(BaseModel):
        text: str
        author: str

    django_model, operations = make_django_model(Comment, check_migrations=False)

    # Should return None when checking is disabled
    assert operations is None
