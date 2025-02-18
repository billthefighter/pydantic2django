"""
Tests for relationship field mapping functionality.
"""
from django.db import models

from pydantic2django import make_django_model
from .fixtures import get_model_fields


def test_foreign_key_relationship(relationship_models):
    """Test mapping of foreign key relationships."""
    Address = relationship_models["Address"]
    User = relationship_models["User"]

    DjangoAddress = make_django_model(Address)
    DjangoUser = make_django_model(User)

    fields = get_model_fields(DjangoUser)

    assert isinstance(fields["address"], models.ForeignKey)
    assert fields["address"].remote_field.model == "Address"
    assert fields["address"].on_delete == models.CASCADE


def test_one_to_one_relationship(relationship_models):
    """Test mapping of one-to-one relationships."""
    Profile = relationship_models["Profile"]
    User = relationship_models["User"]

    DjangoProfile = make_django_model(Profile)
    DjangoUser = make_django_model(User)

    fields = get_model_fields(DjangoUser)

    assert isinstance(fields["profile"], models.OneToOneField)
    assert fields["profile"].remote_field.model == "Profile"
    assert fields["profile"].on_delete == models.CASCADE


def test_many_to_many_relationship(relationship_models):
    """Test mapping of many-to-many relationships."""
    Tag = relationship_models["Tag"]
    User = relationship_models["User"]

    DjangoTag = make_django_model(Tag)
    DjangoUser = make_django_model(User)

    fields = get_model_fields(DjangoUser)

    assert isinstance(fields["tags"], models.ManyToManyField)
    assert fields["tags"].remote_field.model == "Tag"


def test_optional_relationship():
    """Test mapping of optional relationships."""
    from typing import Optional
    from pydantic import BaseModel, Field

    class Category(BaseModel):
        name: str

    class Post(BaseModel):
        title: str
        category: Optional[Category] = Field(on_delete=models.SET_NULL)

    DjangoCategory = make_django_model(Category)
    DjangoPost = make_django_model(Post)

    fields = get_model_fields(DjangoPost)

    assert isinstance(fields["category"], models.ForeignKey)
    assert fields["category"].null
    assert fields["category"].blank
    assert fields["category"].on_delete == models.SET_NULL


def test_relationship_with_related_name():
    """Test relationships with related_name."""
    from pydantic import BaseModel, Field

    class Author(BaseModel):
        name: str

    class Book(BaseModel):
        title: str
        author: Author = Field(related_name="books")

    DjangoAuthor = make_django_model(Author)
    DjangoBook = make_django_model(Book)

    fields = get_model_fields(DjangoBook)

    assert isinstance(fields["author"], models.ForeignKey)
    assert fields["author"].remote_field.related_name == "books"


def test_set_relationship():
    """Test mapping of set relationships."""
    from pydantic import BaseModel

    class Student(BaseModel):
        name: str

    class Course(BaseModel):
        name: str
        students: set[Student]  # Many-to-many using Set

    DjangoStudent = make_django_model(Student)
    DjangoCourse = make_django_model(Course)

    fields = get_model_fields(DjangoCourse)

    assert isinstance(fields["students"], models.ManyToManyField)
    assert fields["students"].remote_field.model == "Student"
