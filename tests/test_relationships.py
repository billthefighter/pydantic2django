"""
Tests for relationship field mapping functionality.
"""
from typing import Optional

from django.db import models
from pydantic import BaseModel, Field

from pydantic2django import make_django_model


def test_foreign_key_relationship():
    """Test mapping of foreign key relationships."""

    class Address(BaseModel):
        street: str
        city: str
        country: str

    class User(BaseModel):
        name: str
        address: Address  # One-to-many relationship

    DjangoAddress = make_django_model(Address)
    DjangoUser = make_django_model(User)

    fields = {f.name: f for f in DjangoUser._meta.get_fields()}

    assert isinstance(fields["address"], models.ForeignKey)
    assert fields["address"].remote_field.model == "Address"
    assert fields["address"].on_delete == models.CASCADE


def test_one_to_one_relationship():
    """Test mapping of one-to-one relationships."""

    class Profile(BaseModel):
        bio: str
        website: str

    class User(BaseModel):
        name: str
        profile: Profile = Field(one_to_one=True)  # One-to-one relationship

    DjangoProfile = make_django_model(Profile)
    DjangoUser = make_django_model(User)

    fields = {f.name: f for f in DjangoUser._meta.get_fields()}

    assert isinstance(fields["profile"], models.OneToOneField)
    assert fields["profile"].remote_field.model == "Profile"
    assert fields["profile"].on_delete == models.CASCADE


def test_many_to_many_relationship():
    """Test mapping of many-to-many relationships."""

    class Tag(BaseModel):
        name: str

    class Post(BaseModel):
        title: str
        content: str
        tags: list[Tag]  # Many-to-many relationship

    DjangoTag = make_django_model(Tag)
    DjangoPost = make_django_model(Post)

    fields = {f.name: f for f in DjangoPost._meta.get_fields()}

    assert isinstance(fields["tags"], models.ManyToManyField)
    assert fields["tags"].remote_field.model == "Tag"


def test_optional_relationship():
    """Test mapping of optional relationships."""

    class Category(BaseModel):
        name: str

    class Post(BaseModel):
        title: str
        category: Optional[Category] = Field(on_delete=models.SET_NULL)

    DjangoCategory = make_django_model(Category)
    DjangoPost = make_django_model(Post)

    fields = {f.name: f for f in DjangoPost._meta.get_fields()}

    assert isinstance(fields["category"], models.ForeignKey)
    assert fields["category"].null
    assert fields["category"].blank
    assert fields["category"].on_delete == models.SET_NULL


def test_relationship_with_related_name():
    """Test relationships with related_name."""

    class Author(BaseModel):
        name: str

    class Book(BaseModel):
        title: str
        author: Author = Field(related_name="books")

    DjangoAuthor = make_django_model(Author)
    DjangoBook = make_django_model(Book)

    fields = {f.name: f for f in DjangoBook._meta.get_fields()}

    assert isinstance(fields["author"], models.ForeignKey)
    assert fields["author"].remote_field.related_name == "books"


def test_set_relationship():
    """Test mapping of set relationships."""

    class Student(BaseModel):
        name: str

    class Course(BaseModel):
        name: str
        students: set[Student]  # Many-to-many using Set

    DjangoStudent = make_django_model(Student)
    DjangoCourse = make_django_model(Course)

    fields = {f.name: f for f in DjangoCourse._meta.get_fields()}

    assert isinstance(fields["students"], models.ManyToManyField)
    assert fields["students"].remote_field.model == "Student"
