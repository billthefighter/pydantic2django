"""
Tests for advanced relationship field mapping functionality.
"""
from typing import Optional

from django.db import models
from pydantic import BaseModel, Field

from pydantic2django import make_django_model


def test_many_to_many_through():
    """Test many-to-many relationship with through model."""

    class Student(BaseModel):
        name: str

    class Course(BaseModel):
        name: str

    class Enrollment(BaseModel):
        student: Student
        course: Course
        date_enrolled: str
        grade: Optional[str] = None

    class StudentWithCourses(BaseModel):
        name: str
        courses: list[Course] = Field(through="Enrollment", through_fields=("student", "course"))

    DjangoStudent = make_django_model(Student)
    DjangoCourse = make_django_model(Course)
    DjangoEnrollment = make_django_model(Enrollment)
    DjangoStudentWithCourses = make_django_model(StudentWithCourses)

    fields = {f.name: f for f in DjangoStudentWithCourses._meta.get_fields()}

    assert isinstance(fields["courses"], models.ManyToManyField)
    assert fields["courses"].remote_field.through == "Enrollment"
    assert fields["courses"].remote_field.through_fields == ("student", "course")


def test_self_referential_foreign_key():
    """Test self-referential foreign key relationship."""

    class Category(BaseModel):
        name: str
        parent: Optional["Category"] = Field(self=True, null=True)

    DjangoCategory = make_django_model(Category)
    fields = {f.name: f for f in DjangoCategory._meta.get_fields()}

    assert isinstance(fields["parent"], models.ForeignKey)
    assert fields["parent"].remote_field.model == "self"
    assert fields["parent"].null is True


def test_self_referential_many_to_many():
    """Test self-referential many-to-many relationship."""

    class Person(BaseModel):
        name: str
        friends: list["Person"] = Field(self=True, symmetrical=True)
        followers: list["Person"] = Field(self=True, symmetrical=False, related_name="following")

    DjangoPerson = make_django_model(Person)
    fields = {f.name: f for f in DjangoPerson._meta.get_fields()}

    # Test friends field (symmetrical)
    assert isinstance(fields["friends"], models.ManyToManyField)
    assert fields["friends"].remote_field.model == "self"
    assert fields["friends"].remote_field.symmetrical is True

    # Test followers field (asymmetrical)
    assert isinstance(fields["followers"], models.ManyToManyField)
    assert fields["followers"].remote_field.model == "self"
    assert fields["followers"].remote_field.symmetrical is False
    assert fields["followers"].remote_field.related_name == "following"


def test_complex_through_model():
    """Test complex many-to-many relationship with through model."""

    class User(BaseModel):
        name: str

    class Group(BaseModel):
        name: str

    class Membership(BaseModel):
        user: User
        group: Group
        date_joined: str
        role: str = "member"
        is_active: bool = True

    class UserWithGroups(BaseModel):
        name: str
        groups: list[Group] = Field(through="Membership", through_fields=("user", "group"), related_name="members")

    DjangoUser = make_django_model(User)
    DjangoGroup = make_django_model(Group)
    DjangoMembership = make_django_model(Membership)
    DjangoUserWithGroups = make_django_model(UserWithGroups)

    fields = {f.name: f for f in DjangoUserWithGroups._meta.get_fields()}

    assert isinstance(fields["groups"], models.ManyToManyField)
    assert fields["groups"].remote_field.through == "Membership"
    assert fields["groups"].remote_field.through_fields == ("user", "group")
    assert fields["groups"].remote_field.related_name == "members"


def test_recursive_relationship():
    """Test recursive relationships with different types."""

    class Employee(BaseModel):
        name: str
        manager: Optional["Employee"] = Field(self=True, null=True, related_name="subordinates")
        mentor: Optional["Employee"] = Field(self=True, null=True, related_name="mentees", one_to_one=True)
        peers: list["Employee"] = Field(
            self=True,
            symmetrical=True,
            related_name="+",  # No reverse accessor
        )

    DjangoEmployee = make_django_model(Employee)
    fields = {f.name: f for f in DjangoEmployee._meta.get_fields()}

    # Test manager field (ForeignKey)
    assert isinstance(fields["manager"], models.ForeignKey)
    assert fields["manager"].remote_field.model == "self"
    assert fields["manager"].null is True
    assert fields["manager"].remote_field.related_name == "subordinates"

    # Test mentor field (OneToOneField)
    assert isinstance(fields["mentor"], models.OneToOneField)
    assert fields["mentor"].remote_field.model == "self"
    assert fields["mentor"].null is True
    assert fields["mentor"].remote_field.related_name == "mentees"

    # Test peers field (ManyToManyField)
    assert isinstance(fields["peers"], models.ManyToManyField)
    assert fields["peers"].remote_field.model == "self"
    assert fields["peers"].remote_field.symmetrical is True
    assert fields["peers"].remote_field.related_name == "+"
