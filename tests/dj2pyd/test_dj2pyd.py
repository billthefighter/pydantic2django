import uuid
import pytest
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field, EmailStr, Json, HttpUrl, IPvAnyAddress
from pydantic_core import PydanticUndefined
from django.db import models  # Import models for type hints in fixtures if needed

# Import the function to test
from pydantic2django.django.conversion import django_to_pydantic

# Import Django Fixtures (ensure pytest path allows this)
# These fixtures return the *class*, not an instance
from ..fixtures.fixtures import (
    related_model,
    abstract_model,
    concrete_model,
    all_fields_model,
    membership_model,
    StatusChoices,  # Assuming StatusChoices is defined in fixtures
)


# --- Pydantic Models Mirroring Django Fixtures ---
# These Pydantic models should correspond to the structure expected
# after conversion from their Django counterparts.


class PydanticRelated(BaseModel):
    id: int
    name: str

    model_config = {"from_attributes": True}  # Enable ORM mode


class PydanticConcrete(BaseModel):
    id: int
    name: str  # Inherited from AbstractModel fixture definition

    model_config = {"from_attributes": True}


# Define Pydantic equivalent for the Enum
class PydanticStatusEnum(str, Enum):
    PENDING = StatusChoices.PENDING.value
    COMPLETED = StatusChoices.COMPLETED.value


# Removed unused PydanticMembership model
# class PydanticMembership(BaseModel):
#     id: int
#     # We expect the converted fields to hold related Pydantic models or IDs
#     # Depending on how we want conversion to work. For depth testing,
#     # let's assume they might convert recursively.
#     # all_fields_model: PydanticAllFields # Causes circular dependency - handle carefully
#     # related_model: PydanticRelated
#     all_fields_model_id: int  # More likely outcome if not deeply converting Membership itself
#     related_model_id: int
#     date_joined: date
#
#     model_config = {"from_attributes": True}


class PydanticAllFields(BaseModel):
    # Match fields in the all_fields_model fixture
    auto_field: int
    boolean_field: bool
    null_boolean_field: Optional[bool] = None
    char_field: str
    char_field_choices: PydanticStatusEnum  # Expect enum value
    text_field: Optional[str] = None
    slug_field: str
    email_field: Optional[EmailStr] = None
    url_field: HttpUrl
    ip_address_field: IPvAnyAddress
    uuid_field: uuid.UUID
    integer_field: int
    big_integer_field: Optional[int] = None
    small_integer_field: int
    positive_integer_field: int
    positive_small_integer_field: int
    positive_big_integer_field: int
    float_field: Optional[float] = None
    decimal_field: Decimal
    date_field: date
    datetime_field: datetime
    time_field: Optional[time] = None
    duration_field: timedelta
    binary_field: bytes  # Pydantic handles bytes
    # File/Image fields are tricky - often represented as URL strings or just names in Pydantic
    file_field: Optional[str] = None  # Assuming conversion results in path/URL string
    image_field: Optional[str] = None
    # image_height: Optional[int] = None # These are usually read-only on Django side
    # image_width: Optional[int] = None
    json_field: Optional[Json[Any]] = None  # Use Json[Any]

    # Relationships - expect nested Pydantic models
    foreign_key_field: Optional[PydanticRelated] = None
    one_to_one_field: Optional[PydanticRelated] = None
    many_to_many_field: List[PydanticRelated] = []

    model_config = {"from_attributes": True}


# --- Test Cases ---


@pytest.mark.django_db
def test_convert_related_model(related_model):
    """Test converting a simple Django model to Pydantic."""
    RelatedModel = related_model  # Use injected fixture value directly
    dj_instance = RelatedModel.objects.create(name="Test Related")

    pydantic_instance = django_to_pydantic(dj_instance, PydanticRelated)

    assert isinstance(pydantic_instance, PydanticRelated)
    assert pydantic_instance.id == dj_instance.id
    assert pydantic_instance.name == "Test Related"


@pytest.mark.django_db
def test_convert_concrete_model(concrete_model):
    """Test converting a concrete Django model inheriting from an abstract one."""
    dj_instance = concrete_model.objects.create(name="Test Concrete")

    pydantic_instance = django_to_pydantic(dj_instance, PydanticConcrete)

    assert isinstance(pydantic_instance, PydanticConcrete)
    assert pydantic_instance.id == dj_instance.id
    assert pydantic_instance.name == "Test Concrete"


@pytest.mark.django_db
def test_convert_all_fields_comprehensive(related_model, all_fields_model, membership_model):
    """Test converting the comprehensive AllFieldsModel with various field types and relationships."""
    RelatedModel = related_model
    AllFieldsModel = all_fields_model
    Membership = membership_model

    # 1. Create related objects
    related_fk = RelatedModel.objects.create(name="RelatedFK")
    related_o2o = RelatedModel.objects.create(name="RelatedO2O")
    related_m2m_1 = RelatedModel.objects.create(name="RelatedM2M_1")
    related_m2m_2 = RelatedModel.objects.create(name="RelatedM2M_2Z")

    # 2. Create the main AllFieldsModel instance
    # Use specific values to test various types
    test_uuid = uuid.uuid4()
    test_decimal = Decimal("12345.67")
    test_datetime = datetime.now()
    test_date = date.today()
    test_time = time(12, 30, 15)
    test_duration = timedelta(days=1, hours=2)
    test_json = {"key": "value", "number": 123}
    test_binary = b"\x01\x02\x03\x04"

    dj_instance = AllFieldsModel.objects.create(
        boolean_field=True,
        null_boolean_field=None,
        char_field="Test Char",
        char_field_choices=StatusChoices.COMPLETED,
        text_field="Long text here",
        slug_field="test-slug-123",
        email_field=f"test-{uuid.uuid4()}@example.com",
        url_field="https://example.com",
        ip_address_field="192.168.1.1",
        uuid_field=test_uuid,
        integer_field=100,
        big_integer_field=999_999_999_999,
        small_integer_field=5,
        positive_integer_field=200,
        positive_small_integer_field=10,
        positive_big_integer_field=1_000_000_000_000,
        float_field=123.45,
        decimal_field=test_decimal,
        # date_field has auto_now_add=True in fixture, value set by DB
        # datetime_field has auto_now=True in fixture, value set by DB
        time_field=test_time,
        duration_field=test_duration,
        binary_field=test_binary,
        # file_field / image_field require file handling setup - skip direct value assertion
        json_field=test_json,
        foreign_key_field=related_fk,
        one_to_one_field=related_o2o,
    )

    # Need to refresh auto fields like date/datetime
    dj_instance.refresh_from_db()

    # 3. Create M2M relationship via the through model
    Membership.objects.create(all_fields_model=dj_instance, related_model=related_m2m_1)
    Membership.objects.create(all_fields_model=dj_instance, related_model=related_m2m_2)

    # 4. Perform conversion
    pydantic_instance = django_to_pydantic(dj_instance, PydanticAllFields)

    # 5. Assertions
    assert isinstance(pydantic_instance, PydanticAllFields)

    # Assert simple types
    assert pydantic_instance.auto_field == dj_instance.auto_field
    assert pydantic_instance.boolean_field is True
    assert pydantic_instance.null_boolean_field is None
    assert pydantic_instance.char_field == "Test Char"
    assert pydantic_instance.char_field_choices == PydanticStatusEnum.COMPLETED
    assert pydantic_instance.text_field == "Long text here"
    assert pydantic_instance.slug_field == "test-slug-123"
    assert pydantic_instance.email_field == dj_instance.email_field
    assert str(pydantic_instance.url_field) == "https://example.com/"
    assert str(pydantic_instance.ip_address_field) == "192.168.1.1"
    assert pydantic_instance.uuid_field == test_uuid
    assert pydantic_instance.integer_field == 100
    assert pydantic_instance.big_integer_field == 999_999_999_999
    assert pydantic_instance.small_integer_field == 5
    assert pydantic_instance.positive_integer_field == 200
    assert pydantic_instance.positive_small_integer_field == 10
    assert pydantic_instance.positive_big_integer_field == 1_000_000_000_000
    assert pydantic_instance.float_field == 123.45
    assert pydantic_instance.decimal_field == test_decimal
    assert pydantic_instance.date_field == dj_instance.date_field  # Compare with refreshed value
    assert pydantic_instance.datetime_field == dj_instance.datetime_field
    assert pydantic_instance.time_field == test_time
    assert pydantic_instance.duration_field == test_duration
    assert pydantic_instance.binary_field == test_binary
    assert pydantic_instance.json_field == test_json
    # Skip file/image field assertions as they depend on storage/conversion logic
    assert pydantic_instance.file_field is None
    assert pydantic_instance.image_field is None

    # Assert relationships
    assert isinstance(pydantic_instance.foreign_key_field, PydanticRelated)
    assert pydantic_instance.foreign_key_field.id == related_fk.id
    assert pydantic_instance.foreign_key_field.name == related_fk.name

    assert isinstance(pydantic_instance.one_to_one_field, PydanticRelated)
    assert pydantic_instance.one_to_one_field.id == related_o2o.id
    assert pydantic_instance.one_to_one_field.name == related_o2o.name

    assert isinstance(pydantic_instance.many_to_many_field, list)
    assert len(pydantic_instance.many_to_many_field) == 2
    m2m_ids = {m.id for m in pydantic_instance.many_to_many_field}
    m2m_names = {m.name for m in pydantic_instance.many_to_many_field}
    assert m2m_ids == {related_m2m_1.id, related_m2m_2.id}
    assert m2m_names == {related_m2m_1.name, related_m2m_2.name}
    for item in pydantic_instance.many_to_many_field:
        assert isinstance(item, PydanticRelated)


@pytest.mark.django_db
def test_convert_all_fields_with_exclusion(related_model, all_fields_model, membership_model):
    """Test the exclude parameter prevents specific fields from being converted."""
    RelatedModel = related_model
    AllFieldsModel = all_fields_model
    Membership = membership_model

    # Minimal setup needed for existence
    related = RelatedModel.objects.create(name="ExcludeRelated")
    dj_instance = AllFieldsModel.objects.create(
        char_field="Keep Me",
        integer_field=555,
        email_field=f"exclude-{uuid.uuid4()}@example.com",
        slug_field=f"exclude-slug-{uuid.uuid4()}",
        url_field="https://exclude.example.com",
        ip_address_field="192.168.1.10",
        uuid_field=uuid.uuid4(),
        decimal_field=Decimal("1.00"),
        foreign_key_field=related,
        boolean_field=True,
        char_field_choices=StatusChoices.PENDING,
        small_integer_field=1,
        positive_integer_field=1,
        positive_small_integer_field=1,
        positive_big_integer_field=1,
        date_field=date.today(),
        datetime_field=datetime.now(),
        duration_field=timedelta(seconds=10),
        binary_field=b"\x00",
    )
    Membership.objects.create(all_fields_model=dj_instance, related_model=related)

    exclude_fields = {"email_field", "many_to_many_field", "foreign_key_field"}
    pydantic_instance = django_to_pydantic(dj_instance, PydanticAllFields, exclude=exclude_fields)

    assert isinstance(pydantic_instance, PydanticAllFields)

    # Check included fields are present and have values
    assert pydantic_instance.char_field == "Keep Me"
    assert pydantic_instance.integer_field == 555
    assert "char_field" in pydantic_instance.model_fields_set
    assert "integer_field" in pydantic_instance.model_fields_set

    # Check excluded fields are not in the set of populated fields
    # Note: They might still exist on the model with default values (like None or [])
    assert "email_field" not in pydantic_instance.model_fields_set
    assert "many_to_many_field" not in pydantic_instance.model_fields_set
    assert "foreign_key_field" not in pydantic_instance.model_fields_set

    # Verify they have default values if defined on Pydantic model
    assert pydantic_instance.email_field is None  # Now matches Optional[EmailStr] = None
    assert pydantic_instance.many_to_many_field == []  # Assuming Pydantic field defaults to []
    assert pydantic_instance.foreign_key_field is None  # Assuming Pydantic field is Optional


@pytest.mark.django_db
def test_convert_max_depth(related_model, all_fields_model, membership_model):
    """Test the max_depth parameter limits recursion."""
    RelatedModel = related_model
    AllFieldsModel = all_fields_model
    Membership = membership_model

    # Setup: AllFields -> Related (Depth 1)
    related1 = RelatedModel.objects.create(name="R1")
    dj_instance = AllFieldsModel.objects.create(
        # Provide required fields that lack defaults
        email_field=f"depth-test-{uuid.uuid4()}@example.com",  # Ensure unique email
        slug_field=f"depth-slug-{uuid.uuid4()}",  # Ensure unique slug
        url_field="https://depth.example.com",
        ip_address_field="192.168.1.11",
        uuid_field=uuid.uuid4(),
        decimal_field=Decimal("1.00"),
        # Provide fields needed for relationships
        foreign_key_field=related1,
        one_to_one_field=related1,
        # Add other required fields from AllFieldsModel if they lack defaults
        boolean_field=True,
        char_field_choices=StatusChoices.PENDING,
        integer_field=1,
        small_integer_field=1,
        positive_integer_field=1,
        positive_small_integer_field=1,
        positive_big_integer_field=1,
        date_field=date.today(),
        datetime_field=datetime.now(),
        duration_field=timedelta(seconds=10),
        binary_field=b"\x00",
    )
    Membership.objects.create(all_fields_model=dj_instance, related_model=related1)

    # Test depth 0: No relationships should be converted
    pydantic_depth_0 = django_to_pydantic(dj_instance, PydanticAllFields, max_depth=0)
    assert isinstance(pydantic_depth_0, PydanticAllFields)
    assert pydantic_depth_0.foreign_key_field is None
    assert pydantic_depth_0.one_to_one_field is None
    assert pydantic_depth_0.many_to_many_field == []  # Conversion of items failed due to depth

    # Test depth 1: Direct relationships should be converted, but not nested ones within them
    # (Our related model has no further relations, so depth 1 is sufficient here)
    pydantic_depth_1 = django_to_pydantic(dj_instance, PydanticAllFields, max_depth=1)
    assert isinstance(pydantic_depth_1, PydanticAllFields)
    assert isinstance(pydantic_depth_1.foreign_key_field, PydanticRelated)
    assert pydantic_depth_1.foreign_key_field.id == related1.id
    assert isinstance(pydantic_depth_1.one_to_one_field, PydanticRelated)
    assert pydantic_depth_1.one_to_one_field.id == related1.id
    assert isinstance(pydantic_depth_1.many_to_many_field, list)
    assert len(pydantic_depth_1.many_to_many_field) == 1
    assert isinstance(pydantic_depth_1.many_to_many_field[0], PydanticRelated)
    assert pydantic_depth_1.many_to_many_field[0].id == related1.id
