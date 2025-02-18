"""
Tests for method and property copying functionality.
"""
from datetime import date
from typing import ClassVar

import pytest
from pydantic import BaseModel, Field
from django.db import models

from pydantic2django import make_django_model
from .fixtures import get_model_fields


def test_instance_methods(method_model):
    """Test copying of instance methods."""
    DjangoModel = make_django_model(method_model)
    instance = DjangoModel(name="Test", value=10)

    assert instance.instance_method() == "Instance: Test"
    assert instance.computed_value == 20
    assert DjangoModel.class_method() == ["A", "B", "C"]
    assert DjangoModel.static_method(5) == 10


def test_properties():
    """Test copying of properties."""
    from pydantic import BaseModel

    class Person(BaseModel):
        first_name: str
        last_name: str
        birth_date: date

        @property
        def full_name(self) -> str:
            return f"{self.first_name} {self.last_name}"

        @property
        def age(self) -> int:
            today = date.today()
            return (
                today.year
                - self.birth_date.year
                - (
                    (today.month, today.day)
                    < (self.birth_date.month, self.birth_date.day)
                )
            )

    DjangoPerson = make_django_model(Person)
    person = DjangoPerson(
        first_name="John", last_name="Doe", birth_date=date(1990, 6, 15)
    )

    assert person.full_name == "John Doe"
    assert isinstance(person.age, int)
    assert person.age > 0


def test_class_methods():
    """Test copying of class methods."""

    class Post(BaseModel):
        title: str
        content: str
        status: str = "draft"

        STATUSES: ClassVar[list[str]] = ["draft", "published", "archived"]

        @classmethod
        def get_available_statuses(cls) -> list[str]:
            return cls.STATUSES

        @classmethod
        def create_draft(cls, title: str, content: str) -> "Post":
            # Create a new instance using the model's constructor
            return cls.model_validate(
                {"title": title, "content": content, "status": "draft"}
            )

    DjangoPost = make_django_model(Post)

    # Override the create_draft method for Django model
    @classmethod
    def django_create_draft(cls, title: str, content: str) -> models.Model:
        return cls(title=title, content=content, status="draft")

    DjangoPost.create_draft = django_create_draft

    assert DjangoPost.get_available_statuses() == ["draft", "published", "archived"]

    draft = DjangoPost.create_draft("Test", "Content")
    assert draft.status == "draft"
    assert isinstance(draft, DjangoPost)


def test_static_methods():
    """Test copying of static methods."""

    class Validator(BaseModel):
        value: str

        @staticmethod
        def normalize_string(s: str) -> str:
            return s.strip().lower()

        @staticmethod
        def is_valid_length(s: str, min_length: int = 3) -> bool:
            return len(s) >= min_length

    DjangoValidator = make_django_model(Validator)

    assert DjangoValidator.normalize_string("  Hello  ") == "hello"
    assert DjangoValidator.is_valid_length("test") is True
    assert DjangoValidator.is_valid_length("a") is False


def test_property_with_setter():
    """Test copying of properties with setters."""
    import pytest
    from pydantic import BaseModel, Field

    class Account(BaseModel):
        balance_value: float = Field(default=0.0)

        @property
        def balance(self) -> float:
            return self.balance_value

        @balance.setter
        def balance(self, value: float) -> None:
            if value < 0:
                raise ValueError("Balance cannot be negative")
            self.balance_value = value

    DjangoAccount = make_django_model(Account)
    account = DjangoAccount()

    account.balance = 100.0
    assert account.balance == 100.0

    with pytest.raises(ValueError):
        account.balance = -50.0


def test_inheritance_methods():
    """Test copying of inherited methods."""
    from pydantic import BaseModel

    class Animal(BaseModel):
        name: str

        def speak(self) -> str:
            return "..."

    class Dog(Animal):
        breed: str

        def speak(self) -> str:
            return "Woof!"

        def get_description(self) -> str:
            return f"{self.name} is a {self.breed}"

    DjangoDog = make_django_model(Dog)
    dog = DjangoDog(name="Rex", breed="Labrador")

    assert dog.speak() == "Woof!"
    assert dog.get_description() == "Rex is a Labrador"


def test_method_docstrings():
    """Test preservation of method docstrings."""
    from pydantic import BaseModel

    class Document(BaseModel):
        content: str

        def word_count(self) -> int:
            """Return the number of words in the content."""
            return len(self.content.split())

        @property
        def summary(self) -> str:
            """Return the first 100 characters of content."""
            return self.content[:100]

    DjangoDocument = make_django_model(Document)

    assert (
        DjangoDocument.word_count.__doc__
        == "Return the number of words in the content."
    )
    assert (
        DjangoDocument.summary.fget.__doc__
        == "Return the first 100 characters of content."
    )
