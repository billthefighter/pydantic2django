import logging
import uuid
import pytest
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, ForwardRef, cast, get_args

from pydantic import BaseModel, Field, EmailStr, Json, HttpUrl, IPvAnyAddress
from pydantic_core import PydanticUndefined
from django.db import models  # Import models for type hints in fixtures if needed
from django.utils.functional import Promise  # Import Promise for type checking lazy objects
from django.utils.translation import gettext_lazy as _

# Configure logging FOR THE CONVERSION MODULE specifically
conv_logger = logging.getLogger("pydantic2django.django.conversion")
conv_logger.setLevel(logging.DEBUG)
# Ensure handlers are set up (add a handler if none exist)
if not conv_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    conv_logger.addHandler(handler)
    conv_logger.propagate = False  # Prevent duplication if root logger also has handlers

# Configure logger for the test file itself
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    test_handler = logging.StreamHandler()
    test_formatter = logging.Formatter("%(asctime)s - TEST - %(levelname)s - %(message)s")
    test_handler.setFormatter(test_formatter)
    logger.addHandler(test_handler)
    logger.propagate = False

# Import the function to test
from pydantic2django.django.conversion import django_to_pydantic, generate_pydantic_class
from pydantic2django.core.bidirectional_mapper import BidirectionalTypeMapper
from pydantic2django.core.relationships import RelationshipConversionAccessor

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
    null_boolean_field: bool | None = None
    char_field: str
    char_field_choices: PydanticStatusEnum  # Expect enum value
    text_field: str | None = None
    slug_field: str
    email_field: EmailStr | None = None
    url_field: HttpUrl
    ip_address_field: IPvAnyAddress
    uuid_field: uuid.UUID
    integer_field: int
    big_integer_field: int | None = None
    small_integer_field: int
    positive_integer_field: int
    positive_small_integer_field: int
    positive_big_integer_field: int
    float_field: float | None = None
    decimal_field: Decimal
    date_field: date
    datetime_field: datetime
    time_field: time | None = None
    duration_field: timedelta
    binary_field: bytes  # Pydantic handles bytes
    # File/Image fields are tricky - often represented as URL strings or just names in Pydantic
    file_field: str | None = None  # Assuming conversion results in path/URL string
    image_field: str | None = None
    # image_height: Optional[int] = None # These are usually read-only on Django side
    # image_width: Optional[int] = None
    json_field: Json[Any] | None = None  # Use Json[Any]

    # Relationships - expect nested Pydantic models
    foreign_key_field: PydanticRelated | None = None
    one_to_one_field: PydanticRelated | None = None
    many_to_many_field: list[PydanticRelated] = []

    model_config = {"from_attributes": True}


# --- Helper Functions for Comparisons ---


def compare_m2m(actual_pyd_list: list[PydanticRelated], expected_dj_qs):
    """Compare M2M list: check type, length, and content (IDs and names)."""
    assert isinstance(actual_pyd_list, list)
    expected_ids = {obj.id for obj in expected_dj_qs}
    expected_names = {obj.name for obj in expected_dj_qs}
    actual_ids = {item.id for item in actual_pyd_list}
    actual_names = {item.name for item in actual_pyd_list}

    assert len(actual_pyd_list) == len(expected_dj_qs)
    assert actual_ids == expected_ids
    assert actual_names == expected_names
    for item in actual_pyd_list:
        assert isinstance(item, PydanticRelated)


def compare_related(actual_pyd_obj: PydanticRelated | None, expected_dj_obj):
    """Compare FK/O2O: check type and key attributes."""
    if expected_dj_obj is None:
        assert actual_pyd_obj is None
    else:
        assert isinstance(actual_pyd_obj, PydanticRelated)
        assert actual_pyd_obj.id == expected_dj_obj.id
        assert actual_pyd_obj.name == expected_dj_obj.name


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


# --- Fixture for Comprehensive Test Setup ---


@pytest.fixture(scope="function")
def comprehensive_data(related_model, all_fields_model, membership_model):
    """Sets up data for comprehensive field tests and yields instances."""
    RelatedModel = related_model
    AllFieldsModel = all_fields_model
    Membership = membership_model

    # 1. Create related objects
    related_fk = RelatedModel.objects.create(name="RelatedFK")
    related_o2o = RelatedModel.objects.create(name="RelatedO2O")
    related_m2m_1 = RelatedModel.objects.create(name="RelatedM2M_1")
    related_m2m_2 = RelatedModel.objects.create(name="RelatedM2M_2Z")
    m2m_qs = [related_m2m_1, related_m2m_2]

    # 2. Create the main AllFieldsModel instance
    test_uuid = uuid.uuid4()
    test_decimal = Decimal("12345.67")
    # test_datetime = datetime.now() # Handled by auto_now
    # test_date = date.today() # Handled by auto_now_add
    test_time = time(12, 30, 15)
    test_duration = timedelta(days=1, hours=2)
    test_json = {"key": "value", "number": 123}
    test_binary = b""

    dj_instance = AllFieldsModel.objects.create(
        boolean_field=True,
        null_boolean_field=None,
        char_field="Test Char",
        char_field_choices=StatusChoices.COMPLETED,
        text_field="Long text here",
        slug_field="test-slug-123",
        email_field=f"test-{uuid.uuid4()}@example.com",  # Generate dynamically
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
        time_field=test_time,
        duration_field=test_duration,
        binary_field=test_binary,
        json_field=test_json,
        foreign_key_field=related_fk,
        one_to_one_field=related_o2o,
    )
    # Need to refresh auto fields like date/datetime after create
    dj_instance.refresh_from_db()

    # 3. Create M2M relationship via the through model
    Membership.objects.create(all_fields_model=dj_instance, related_model=related_m2m_1)
    Membership.objects.create(all_fields_model=dj_instance, related_model=related_m2m_2)

    # 4. Perform conversion
    pydantic_instance = django_to_pydantic(dj_instance, PydanticAllFields)

    # Yield data needed for tests
    yield dj_instance, pydantic_instance, {
        "related_fk": related_fk,
        "related_o2o": related_o2o,
        "m2m_qs": m2m_qs,
        "test_uuid": test_uuid,
        "test_decimal": test_decimal,
        "test_time": test_time,
        "test_duration": test_duration,
        "test_json": test_json,
        "test_binary": test_binary,
    }
    # Teardown is handled by django_db fixture's transaction rollback


# --- Parameterized Test Cases for All Fields ---

# Define test parameters: (field_name, expected_value_or_getter, comparison_func=None)
# Getter can be a lambda accessing dj_instance or a direct value from setup_data
comprehensive_test_params = [
    ("auto_field", lambda dj: dj.auto_field),
    ("boolean_field", True),
    ("null_boolean_field", None),
    ("char_field", "Test Char"),
    ("char_field_choices", PydanticStatusEnum.COMPLETED),
    ("text_field", "Long text here"),
    ("slug_field", "test-slug-123"),
    ("email_field", lambda dj: dj.email_field),
    ("url_field", "https://example.com/", lambda a, e: str(a) == e),  # Compare string representation
    ("ip_address_field", "192.168.1.1", lambda a, e: str(a) == e),  # Compare string representation
    ("uuid_field", lambda _, setup: setup["test_uuid"]),
    ("integer_field", 100),
    ("big_integer_field", 999_999_999_999),
    ("small_integer_field", 5),
    ("positive_integer_field", 200),
    ("positive_small_integer_field", 10),
    ("positive_big_integer_field", 1_000_000_000_000),
    ("float_field", lambda _, __: pytest.approx(123.45)),  # Use pytest.approx for float
    ("decimal_field", lambda _, setup: setup["test_decimal"]),
    ("date_field", lambda dj: dj.date_field),  # Compare refreshed value
    ("datetime_field", lambda dj: dj.datetime_field),  # Compare refreshed value
    ("time_field", lambda _, setup: setup["test_time"]),
    ("duration_field", lambda _, setup: setup["test_duration"]),
    ("binary_field", lambda _, setup: setup["test_binary"]),
    ("json_field", lambda _, setup: setup["test_json"]),
    ("file_field", None),  # Skipped in original test
    ("image_field", None),  # Skipped in original test
    ("foreign_key_field", lambda _, setup: setup["related_fk"], compare_related),
    ("one_to_one_field", lambda _, setup: setup["related_o2o"], compare_related),
    ("many_to_many_field", lambda _, setup: setup["m2m_qs"], compare_m2m),
]


@pytest.mark.parametrize(
    "field_name, expected_value_or_getter, comparator",
    [(p[0], p[1], p[2] if len(p) > 2 else None) for p in comprehensive_test_params],
    ids=[p[0] for p in comprehensive_test_params],  # Use field names for test IDs
)
@pytest.mark.django_db  # Still needed for the test function itself if it interacts
def test_all_fields_parameterized(comprehensive_data, field_name, expected_value_or_getter, comparator):
    """Tests individual fields of the converted PydanticAllFields model."""
    dj_instance, pydantic_instance, setup_data = comprehensive_data

    assert isinstance(pydantic_instance, PydanticAllFields)

    actual_value = getattr(pydantic_instance, field_name)

    # Determine the expected value
    if callable(expected_value_or_getter):
        # If it's a lambda/function, call it with instances/setup data
        try:
            expected_value = expected_value_or_getter(dj_instance, setup_data)
        except TypeError:  # Handle functions that only need one arg
            expected_value = expected_value_or_getter(dj_instance)
    else:
        # Otherwise, it's a direct value
        expected_value = expected_value_or_getter

    # Perform comparison
    if comparator:
        comparator(actual_value, expected_value)
    elif isinstance(expected_value, float):  # Default check for approx if no comparator given
        assert actual_value == pytest.approx(expected_value)
    else:
        assert actual_value == expected_value


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
        json_field=None,  # Explicitly set to None
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


# --- Tests for DjangoPydanticConverter ---

from pydantic2django.django.conversion import DjangoPydanticConverter


@pytest.mark.django_db
class TestDjangoPydanticConverter:
    """Tests for the DjangoPydanticConverter class."""

    def test_init_and_generate(self, all_fields_model):
        """Test converter initialization and dynamic Pydantic model generation."""
        AllFieldsModel = all_fields_model
        converter = DjangoPydanticConverter(AllFieldsModel)

        assert converter.django_model_cls == AllFieldsModel
        assert converter.initial_django_instance is None
        # Ensure the generated model is actually a BaseModel type, not ForwardRef for this test
        GeneratedModel = converter.generated_pydantic_model
        assert isinstance(GeneratedModel, type) and issubclass(GeneratedModel, BaseModel)
        # Check if the generated model name looks correct (optional)
        assert GeneratedModel.__name__.startswith(AllFieldsModel.__name__)

        # Test init with instance
        dj_instance = AllFieldsModel.objects.create(
            # Provide required fields
            email_field=f"init-test-{uuid.uuid4()}@example.com",
            slug_field=f"init-slug-{uuid.uuid4()}",
            url_field="https://init.example.com",
            ip_address_field="192.168.1.12",
            uuid_field=uuid.uuid4(),
            decimal_field=Decimal("1.00"),
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
        converter_with_instance = DjangoPydanticConverter(dj_instance)
        assert converter_with_instance.django_model_cls == AllFieldsModel
        assert converter_with_instance.initial_django_instance == dj_instance

    def test_to_pydantic_conversion(self, related_model, all_fields_model):
        """Test converting Django instance to Pydantic via the converter."""
        RelatedModel = related_model
        AllFieldsModel = all_fields_model
        related = RelatedModel.objects.create(name="ConverterRelated")
        dj_instance = AllFieldsModel.objects.create(
            char_field="To Pydantic",
            integer_field=123,
            foreign_key_field=related,
            # Add other required fields
            email_field=f"tpyd-{uuid.uuid4()}@example.com",
            slug_field=f"tpyd-slug-{uuid.uuid4()}",
            url_field="https://tpyd.example.com",
            ip_address_field="192.168.1.13",
            uuid_field=uuid.uuid4(),
            decimal_field=Decimal("1.23"),
            boolean_field=False,
            char_field_choices=StatusChoices.COMPLETED,
            small_integer_field=2,
            positive_integer_field=2,
            positive_small_integer_field=2,
            positive_big_integer_field=2,
            date_field=date.today(),
            datetime_field=datetime.now(),
            duration_field=timedelta(hours=1),
            binary_field=b"\x01\x02",
        )

        # Test conversion using instance provided at init
        converter1 = DjangoPydanticConverter(dj_instance)
        pyd_instance1 = converter1.to_pydantic()
        # Ensure the model used for isinstance is the resolved type
        GeneratedModel1 = converter1.generated_pydantic_model
        assert isinstance(GeneratedModel1, type) and issubclass(GeneratedModel1, BaseModel)
        assert isinstance(pyd_instance1, GeneratedModel1)
        assert pyd_instance1.char_field == "To Pydantic"  # type: ignore[attr-defined]
        assert pyd_instance1.integer_field == 123  # type: ignore[attr-defined]
        # Check that the foreign key field is a BaseModel and has the expected ID
        fk_field_value = getattr(pyd_instance1, "foreign_key_field", None)
        assert isinstance(fk_field_value, BaseModel)
        assert getattr(fk_field_value, "id", None) == related.id

        # Test conversion using instance passed to method
        converter2 = DjangoPydanticConverter(AllFieldsModel)
        pyd_instance2 = converter2.to_pydantic(dj_instance)
        # Ensure the model used for isinstance is the resolved type
        GeneratedModel2 = converter2.generated_pydantic_model
        assert isinstance(GeneratedModel2, type) and issubclass(GeneratedModel2, BaseModel)
        assert isinstance(pyd_instance2, GeneratedModel2)
        assert pyd_instance1.model_dump() == pyd_instance2.model_dump()

    def test_to_django_creation(self, related_model, all_fields_model, membership_model):
        """Test creating a new Django instance from a Pydantic model via the converter."""
        RelatedModel = related_model
        AllFieldsModel = all_fields_model
        Membership = membership_model

        # Create related instances needed for FK/M2M in Pydantic data
        related_for_fk = RelatedModel.objects.create(name="TargetFK")
        related_for_m2m1 = RelatedModel.objects.create(name="TargetM2M_1")
        related_for_m2m2 = RelatedModel.objects.create(name="TargetM2M_2")

        # Create a Pydantic instance (use the generated type from converter)
        converter = DjangoPydanticConverter(AllFieldsModel)
        PydanticGenerated = converter.generated_pydantic_model
        # Ensure the generated model is actually a BaseModel type before instantiation
        assert isinstance(PydanticGenerated, type) and issubclass(PydanticGenerated, BaseModel)

        test_uuid = uuid.uuid4()
        pyd_data = {
            # Provide values for fields required by Pydantic model
            "auto_field": None,  # PK field, should be optional or None for creation data
            "boolean_field": True,
            "null_boolean_field": None,
            "char_field": "Created via Converter",
            "char_field_choices": "PENDING",  # Use string value matching StatusChoices.PENDING.value
            "text_field": "Some text",
            "slug_field": f"created-slug-{uuid.uuid4()}",
            "email_field": f"created-{uuid.uuid4()}@example.com",
            "url_field": "https://created.example.com",
            "ip_address_field": "10.0.0.1",
            "uuid_field": test_uuid,
            "integer_field": 987,
            "big_integer_field": 1_000_000_000,
            "small_integer_field": -1,
            "positive_integer_field": 50,
            "positive_small_integer_field": 5,
            "positive_big_integer_field": 5_000_000_000,
            "float_field": 99.9,
            "decimal_field": Decimal("9876.54"),
            "date_field": date(2023, 1, 1),
            "datetime_field": datetime(2023, 1, 1, 10, 30, 0),
            "time_field": time(14, 0, 0),
            "duration_field": timedelta(minutes=30),
            "binary_field": b"CREATED",
            "file_field": None,  # Provide None as it's nullable
            "image_field": None,  # Provide None as it's nullable
            "image_height": None,  # Provide None as it's nullable
            "image_width": None,  # Provide None as it's nullable
            "json_field": {"status": "new"},  # Assuming generated Pydantic handles dict for JSONField
            # Provide dicts for relationships matching nested Pydantic model structure
            "foreign_key_field": {"id": related_for_fk.id, "name": related_for_fk.name} if related_for_fk else None,
            "one_to_one_field": None,  # Test setting FK to None
            "many_to_many_field": [{"id": m.id, "name": m.name} for m in [related_for_m2m1, related_for_m2m2]],
        }
        # Now it's safe to instantiate
        pyd_instance = PydanticGenerated(**pyd_data)

        # Perform conversion to Django
        created_dj_instance = converter.to_django(pyd_instance)

        # Assertions: Check the created Django instance
        assert isinstance(created_dj_instance, AllFieldsModel)
        assert created_dj_instance.pk is not None  # Should have been saved

        # Verify fields were set correctly
        assert created_dj_instance.char_field == "Created via Converter"
        assert created_dj_instance.integer_field == 987
        # assert created_dj_instance.uuid_field == test_uuid # Removed: uuid_field is editable=False, default is used
        assert created_dj_instance.char_field_choices == StatusChoices.PENDING.value
        assert created_dj_instance.foreign_key_field_id == related_for_fk.id
        assert created_dj_instance.one_to_one_field is None
        assert created_dj_instance.json_field == {"status": "new"}

        # Verify M2M relationships (need to query the manager)
        m2m_ids = set(created_dj_instance.many_to_many_field.values_list("id", flat=True))
        assert m2m_ids == {related_for_m2m1.id, related_for_m2m2.id}

    def test_to_django_update(self, related_model, all_fields_model, membership_model):
        """Test updating an existing Django instance from Pydantic via the converter."""
        RelatedModel = related_model
        AllFieldsModel = all_fields_model
        Membership = membership_model

        # 1. Create initial Django instance and related objects
        related1 = RelatedModel.objects.create(name="InitialFK")
        related2 = RelatedModel.objects.create(name="UpdatedFK")
        m2m1 = RelatedModel.objects.create(name="InitialM2M")
        m2m2 = RelatedModel.objects.create(name="KeepM2M")
        m2m3 = RelatedModel.objects.create(name="AddNewM2M")

        dj_instance = AllFieldsModel.objects.create(
            char_field="Initial Value",
            integer_field=100,
            foreign_key_field=related1,
            # Add other required fields
            email_field=f"update-{uuid.uuid4()}@example.com",
            slug_field=f"update-slug-{uuid.uuid4()}",
            url_field="https://update.example.com",
            ip_address_field="192.168.1.14",
            uuid_field=uuid.uuid4(),
            decimal_field=Decimal("100.00"),
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
        Membership.objects.create(all_fields_model=dj_instance, related_model=m2m1)
        Membership.objects.create(all_fields_model=dj_instance, related_model=m2m2)
        initial_pk = dj_instance.pk

        # 2. Create Pydantic instance with updated data (including PK)
        converter = DjangoPydanticConverter(AllFieldsModel)
        PydanticGenerated = converter.generated_pydantic_model
        # Ensure the generated model is actually a BaseModel type before instantiation
        assert isinstance(PydanticGenerated, type) and issubclass(PydanticGenerated, BaseModel)

        pyd_data_update = {
            "auto_field": initial_pk,  # Include PK for update identification
            "char_field": "Updated Value",
            "integer_field": 200,
            # Provide dict for FK matching expected nested Pydantic structure
            "foreign_key_field": {"id": related2.id, "name": related2.name},
            # Provide list of dicts for M2M
            "many_to_many_field": [{"id": m2m2.id, "name": m2m2.name}, {"id": m2m3.id, "name": m2m3.name}],
            # Include other fields that might be required by model validation
            "boolean_field": dj_instance.boolean_field,
            "null_boolean_field": dj_instance.null_boolean_field,
            "char_field_choices": dj_instance.char_field_choices,
            "text_field": dj_instance.text_field,
            "slug_field": dj_instance.slug_field,
            "email_field": dj_instance.email_field,
            "url_field": dj_instance.url_field,
            "ip_address_field": dj_instance.ip_address_field,
            "uuid_field": dj_instance.uuid_field,
            "big_integer_field": dj_instance.big_integer_field,
            "small_integer_field": dj_instance.small_integer_field,
            "positive_integer_field": dj_instance.positive_integer_field,
            "positive_small_integer_field": dj_instance.positive_small_integer_field,
            "positive_big_integer_field": dj_instance.positive_big_integer_field,
            "float_field": dj_instance.float_field,
            "decimal_field": dj_instance.decimal_field,
            "date_field": dj_instance.date_field,
            "datetime_field": dj_instance.datetime_field,
            "time_field": dj_instance.time_field,
            "duration_field": dj_instance.duration_field,
            "binary_field": dj_instance.binary_field,
            "file_field": None,  # Provide None explicitly
            "image_field": None,  # Provide None explicitly
            "image_height": None,  # Provide None explicitly
            "image_width": None,  # Provide None explicitly
            "json_field": dj_instance.json_field,
            "one_to_one_field": None,
        }
        # Now it's safe to instantiate
        pyd_instance_update = PydanticGenerated(**pyd_data_update)

        # 3. Perform update using to_django
        # Option 1: Pass initial instance to converter
        # converter_with_instance = DjangoPydanticConverter(dj_instance)
        # updated_dj_instance = converter_with_instance.to_django(pyd_instance_update)

        # Option 2: Use converter initialized with class
        updated_dj_instance = converter.to_django(pyd_instance_update)

        # Option 3: Pass update_instance explicitly
        # updated_dj_instance = converter.to_django(pyd_instance_update, update_instance=dj_instance)

        # 4. Assertions: Fetch the instance and check fields
        assert updated_dj_instance.pk == initial_pk
        updated_dj_instance.refresh_from_db()  # Ensure we have latest data

        assert updated_dj_instance.char_field == "Updated Value"
        assert updated_dj_instance.integer_field == 200
        assert updated_dj_instance.foreign_key_field_id == related2.id

        # Verify M2M relationships were updated correctly
        updated_m2m_ids = set(updated_dj_instance.many_to_many_field.values_list("id", flat=True))
        assert updated_m2m_ids == {m2m2.id, m2m3.id}

    def test_converter_exclude(self, all_fields_model):
        """Test exclude parameter in DjangoPydanticConverter."""
        AllFieldsModel = all_fields_model
        dj_instance = AllFieldsModel.objects.create(
            char_field="Exclude Test",
            integer_field=999,
            # Add other required fields
            email_field=f"exclude-{uuid.uuid4()}@example.com",
            slug_field=f"exclude-slug-{uuid.uuid4()}",
            url_field="https://exclude.example.com",
            ip_address_field="192.168.1.15",
            uuid_field=uuid.uuid4(),
            decimal_field=Decimal("9.99"),
            boolean_field=True,
            char_field_choices=StatusChoices.PENDING,
            small_integer_field=1,
            positive_integer_field=1,
            positive_small_integer_field=1,
            positive_big_integer_field=1,
            date_field=date.today(),
            datetime_field=datetime.now(),
            duration_field=timedelta(seconds=5),
            binary_field=b"x",
        )

        exclude_fields = {"integer_field", "char_field"}
        converter = DjangoPydanticConverter(dj_instance, exclude=exclude_fields)
        pyd_instance = converter.to_pydantic()

        # Ensure the model used for isinstance is the resolved type
        GeneratedModel = converter.generated_pydantic_model
        assert isinstance(GeneratedModel, type) and issubclass(GeneratedModel, BaseModel)
        assert isinstance(pyd_instance, GeneratedModel)
        assert "integer_field" not in pyd_instance.model_fields_set
        assert "char_field" not in pyd_instance.model_fields_set
        assert "email_field" in pyd_instance.model_fields_set  # Check others are present

    def test_converter_max_depth(self, related_model, all_fields_model, membership_model):
        """Test max_depth parameter in DjangoPydanticConverter."""
        RelatedModel = related_model
        AllFieldsModel = all_fields_model
        Membership = membership_model

        related = RelatedModel.objects.create(name="DepthRelated")
        dj_instance = AllFieldsModel.objects.create(
            foreign_key_field=related,
            # Add other required fields
            email_field=f"depth-{uuid.uuid4()}@example.com",
            slug_field=f"depth-slug-{uuid.uuid4()}",
            url_field="https://depth.example.com",
            ip_address_field="192.168.1.16",
            uuid_field=uuid.uuid4(),
            decimal_field=Decimal("1.11"),
            boolean_field=True,
            char_field_choices=StatusChoices.PENDING,
            integer_field=1,
            small_integer_field=1,
            positive_integer_field=1,
            positive_small_integer_field=1,
            positive_big_integer_field=1,
            date_field=date.today(),
            datetime_field=datetime.now(),
            duration_field=timedelta(seconds=1),
            binary_field=b"y",
        )
        Membership.objects.create(all_fields_model=dj_instance, related_model=related)

        # Test depth 0
        converter0 = DjangoPydanticConverter(dj_instance, max_depth=0)
        pyd_instance0 = converter0.to_pydantic()
        # We expect the generated model to be valid even with depth 0
        GeneratedModel0 = converter0.generated_pydantic_model
        assert isinstance(GeneratedModel0, type) and issubclass(GeneratedModel0, BaseModel)
        assert isinstance(pyd_instance0, GeneratedModel0)
        # Access attributes safely after checking type
        assert getattr(pyd_instance0, "foreign_key_field", PydanticUndefined) is None
        assert getattr(pyd_instance0, "many_to_many_field", PydanticUndefined) == []

        # Test depth 1
        converter1 = DjangoPydanticConverter(dj_instance, max_depth=1)
        pyd_instance1 = converter1.to_pydantic()
        GeneratedModel1 = converter1.generated_pydantic_model
        assert isinstance(GeneratedModel1, type) and issubclass(GeneratedModel1, BaseModel)
        assert isinstance(pyd_instance1, GeneratedModel1)
        # Note: Accessing dynamically generated fields below
        assert isinstance(getattr(pyd_instance1, "foreign_key_field", None), BaseModel)
        m2m_field_val = getattr(pyd_instance1, "many_to_many_field", [])
        assert isinstance(m2m_field_val, list) and len(m2m_field_val) == 1
        assert isinstance(m2m_field_val[0], BaseModel)


# --- Test for Lazy Choices Serialization ---


@pytest.mark.django_db
def test_lazy_choices_serialization_in_generated_schema(lazy_choice_model):
    """
    Tests that json_schema_extra in generated Pydantic models contains
    resolved strings for lazy choice labels, not Django lazy proxies.
    Also tests that the generated model's JSON schema can be created without error.
    """
    LazyChoiceModel = lazy_choice_model
    converter = DjangoPydanticConverter(LazyChoiceModel)

    # Get the generated model class (ensure it's resolved)
    GeneratedModel = converter.generated_pydantic_model
    assert isinstance(GeneratedModel, type) and issubclass(GeneratedModel, BaseModel)

    # Inspect the 'lazy_choice_field'
    lazy_field_info = GeneratedModel.model_fields.get("lazy_choice_field")
    assert lazy_field_info is not None, "lazy_choice_field not found in generated model"

    # Check json_schema_extra for 'choices'
    schema_extra = lazy_field_info.json_schema_extra
    assert isinstance(schema_extra, dict), "json_schema_extra should be a dict"
    choices = schema_extra.get("choices")
    assert isinstance(choices, list), "'choices' not found or not a list in json_schema_extra"
    assert len(choices) == 2, "Expected 2 choices"

    # Crucially, check that labels are strings, not lazy proxies
    for choice_item in choices:
        assert (
            isinstance(choice_item, (list, tuple)) and len(choice_item) == 2
        ), f"Choice item '{choice_item}' is not a list/tuple of length 2"
        value, label = choice_item  # Unpack after verifying structure
        # Log the label type for clarity during testing
        logger.debug(f"Checking choice in test: Value={value!r}, Label={label!r} (Type: {type(label)})")
        assert isinstance(
            label, str
        ), f"Choice label '{label}' for value '{value}' is not a string (type: {type(label)})"
        assert not isinstance(label, Promise), f"Choice label '{label}' is still a lazy proxy!"

    # Verify that the full JSON schema generation does not raise the serialization error
    try:
        schema = GeneratedModel.model_json_schema()
        assert isinstance(schema, dict)
        # Optionally, check the schema content for the choices enum/description
        props = schema.get("properties", {})
        lazy_choice_prop = props.get("lazy_choice_field")
        assert lazy_choice_prop is not None
        # Pydantic might represent choices as enum in the schema
        assert "enum" in lazy_choice_prop or "anyOf" in lazy_choice_prop or "allOf" in lazy_choice_prop
        # Check description if present might have resolved label text
        # assert "Choice 1" in lazy_choice_prop.get("description", "")

    except Exception as e:
        pytest.fail(f"model_json_schema() raised an unexpected error: {e}")


@pytest.mark.xfail(
    reason=(
        "The DjangoPydanticConverter now proactively resolves lazy choices "
        "during model generation, preventing the PydanticSerializationError. "
        "This test correctly fails its assertion that lazy proxies *remain* after conversion."
    )
)
@pytest.mark.django_db
def test_lazy_choices_fail_serialization_without_translation_active(lazy_choice_model):
    """
    Tests that generating the JSON schema FAILs with PydanticSerializationError
    if translations are not active when the converter generates the model,
    as the lazy proxies in choices will not be resolved.
    """
    from django.utils import translation
    from pydantic_core import PydanticSerializationError
    from django.utils.functional import Promise  # Ensure Promise is imported here too

    LazyChoiceModel = lazy_choice_model

    # --- Step 1: Deactivate translations --- Ensure no active context
    translation.deactivate_all()
    logger.info("Deactivated translations before converter initialization.")

    # --- Step 2: Instantiate Converter --- This generates the model internally
    converter = DjangoPydanticConverter(LazyChoiceModel)
    GeneratedModel = converter.generated_pydantic_model
    assert isinstance(GeneratedModel, type) and issubclass(GeneratedModel, BaseModel)

    # Optional: Verify proxies are still in json_schema_extra at this point
    lazy_field_info = GeneratedModel.model_fields.get("lazy_choice_field")
    assert lazy_field_info is not None
    schema_extra = lazy_field_info.json_schema_extra
    assert isinstance(schema_extra, dict)
    choices = schema_extra.get("choices", [])
    # Ensure choices is a list of pairs before iterating
    valid_choices_structure = False
    if isinstance(choices, list):
        valid_choices_structure = True  # Assume true initially
        for item in choices:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                valid_choices_structure = False
                break
    assert valid_choices_structure, "Choices structure is invalid before checking for Promise."
    # This assertion is expected to fail because the converter resolves proxies
    assert valid_choices_structure and any(
        isinstance(label, Promise) for _, label in choices
    ), "Lazy proxies were unexpectedly resolved even with translations deactivated."

    logger.info("Confirmed lazy proxies exist in json_schema_extra before schema generation.")

    # --- Step 3 & 4: Generate Schema and Assert Failure --- #
    # This part is less relevant now as the primary assertion above will fail
    with pytest.raises(PydanticSerializationError) as excinfo:
        _ = GeneratedModel.model_json_schema()  # Attempt schema generation

    # Check that the error message contains the expected unserializable type
    assert "django.utils.functional.lazy.<locals>.__proxy__" in str(excinfo.value)
    logger.info("Successfully caught PydanticSerializationError due to unresolved lazy proxy.")

    # --- Step 5 (Optional Contrast): Activate translations and assert success --- #
    try:
        # Activate a default language (e.g., 'en')
        with translation.override("en"):
            logger.info("Activated translation context ('en')")
            # We might need to re-trigger the choice handling if it happened only during init.
            # Let's regenerate the converter *within* the context for a clearer test.
            converter_active = DjangoPydanticConverter(LazyChoiceModel)
            GeneratedModelActive = converter_active.generated_pydantic_model
            assert isinstance(GeneratedModelActive, type) and issubclass(GeneratedModelActive, BaseModel)

            # Verify labels are resolved now
            lazy_field_info_active = GeneratedModelActive.model_fields.get("lazy_choice_field")
            assert lazy_field_info_active is not None
            schema_extra_active = lazy_field_info_active.json_schema_extra
            assert isinstance(schema_extra_active, dict)
            choices_active = schema_extra_active.get("choices", [])
            # Ensure choices_active is a list of pairs before iterating
            valid_choices_active_structure = False
            if isinstance(choices_active, list):
                valid_choices_active_structure = True  # Assume true initially
                for item in choices_active:
                    if not (isinstance(item, (list, tuple)) and len(item) == 2):
                        valid_choices_active_structure = False
                        break
            assert valid_choices_active_structure, "Choices structure is invalid before checking for resolved strings."
            # This assertion should pass because converter resolves proxies AND translation is active
            assert valid_choices_active_structure and all(
                isinstance(label, str) for _, label in choices_active
            ), "Lazy proxies were NOT resolved even with translations active."
            logger.info("Confirmed lazy proxies are resolved with active translation context.")

            # Now, schema generation should succeed
            schema_active = GeneratedModelActive.model_json_schema()
            assert isinstance(schema_active, dict)
            logger.info("Successfully generated JSON schema with active translation context.")

    except Exception as e:
        pytest.fail(f"Schema generation unexpectedly failed even with active translation context: {e}")


@pytest.mark.django_db
def test_pydantic_serialization_error_with_raw_lazy_choices(lazy_choice_model):
    """
    Tests that directly providing unresolved lazy choices to Pydantic's
    schema generation (bypassing the converter's fix) raises the expected
    PydanticSerializationError when translations are inactive.
    """
    from django.utils import translation
    from django.utils.functional import Promise
    from pydantic import BaseModel, Field
    from pydantic_core import PydanticSerializationError
    from pydantic.fields import FieldInfo

    LazyChoiceModel = lazy_choice_model

    # --- Step 1: Deactivate translations --- Ensure no active context
    translation.deactivate_all()
    logger.info("Deactivated translations before attempting direct schema generation.")

    # --- Step 2: Get Raw Choices with Lazy Proxies ---
    django_field = LazyChoiceModel._meta.get_field("lazy_choice_field")
    raw_choices = django_field.choices
    # Verify that we actually have Promise objects in the raw choices
    assert isinstance(raw_choices, list)
    assert any(
        isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], Promise) for item in raw_choices
    ), "Raw choices did not contain lazy proxies as expected."
    logger.info("Confirmed raw choices contain lazy proxies.")

    # --- Step 3: Manually Create FieldInfo with Raw Choices ---
    faulty_field_info = FieldInfo(
        annotation=str,  # Type doesn't matter much here, focus is on schema extra
        json_schema_extra={"choices": raw_choices},
    )

    # --- Step 4: Create a Model and Trigger Schema Generation ---
    class ModelWithRawLazyChoices(BaseModel):
        faulty_field: str = Field(json_schema_extra={"choices": raw_choices})

    # --- Step 5: Assert Failure ---
    with pytest.raises(PydanticSerializationError) as excinfo:
        _ = ModelWithRawLazyChoices.model_json_schema()

    # Check that the error message contains the expected unserializable type
    assert "django.utils.functional.lazy.<locals>.__proxy__" in str(excinfo.value)
    logger.info("Successfully caught PydanticSerializationError when generating schema with raw lazy choices.")


# (Add the lazy_choice_model fixture to conftest.py or fixtures.py)
# Example fixture (place in appropriate fixtures file):
# from django.db import models
# from django.utils.translation import gettext_lazy as _
# import pytest

# @pytest.fixture(scope="session")
# def lazy_choice_model():
#     class LazyChoiceModel(models.Model):
#         class Meta:
#             app_label = "test_app" # Required for dynamic models

#         class LazyChoices(models.TextChoices):
#             CHOICE1 = "C1", _("Lazy Choice One")
#             CHOICE2 = "C2", _("Lazy Choice Two")

#         lazy_choice_field = models.CharField(
#             max_length=2,
#             choices=LazyChoices.choices,
#             help_text=_("Field with lazy choices"),
#             null=True,
#             blank=True,
#         )
#         # Add a simple field to ensure model is valid
#         name = models.CharField(max_length=50, default="Test")

#     return LazyChoiceModel


@pytest.mark.django_db
def test_field_requirement_fix():
    """
    Test that the field requirement fix correctly handles optional fields
    when field_info_kwargs is empty or invalid.
    """
    from django.db import models
    from pydantic import BaseModel, ValidationError

    # Create a test Django model with both required and optional fields
    class TestModel(models.Model):
        class Meta:
            app_label = "test_app"

        # Required fields
        required_field = models.CharField(max_length=100)
        required_int = models.IntegerField()

        # Optional fields (null=True)
        optional_field = models.CharField(max_length=100, null=True, blank=True)
        optional_date = models.DateField(null=True, blank=True)
        optional_decimal = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)

    # Set up the mapper and generate Pydantic model
    mapper = BidirectionalTypeMapper(RelationshipConversionAccessor())
    generated_model = generate_pydantic_class(TestModel, mapper, model_name="TestPydanticModel")

    # Verify the generated model has the correct field requirements
    assert issubclass(generated_model, BaseModel)

    # Test that required fields are actually required
    with pytest.raises(ValidationError):
        # Missing required fields should raise ValidationError
        generated_model()

    # Test that optional fields can be omitted
    try:
        instance = generated_model(
            required_field="test",
            required_int=42
        )
        assert instance.required_field == "test"
        assert instance.required_int == 42
        assert instance.optional_field is None
        assert instance.optional_date is None
        assert instance.optional_decimal is None
    except ValidationError as e:
        pytest.fail(f"Optional fields should not be required: {e}")

    # Test that optional fields can be provided
    try:
        instance = generated_model(
            required_field="test",
            required_int=42,
            optional_field="optional value",
            optional_date=date(2023, 1, 1),
            optional_decimal=Decimal("10.50")
        )
        assert instance.optional_field == "optional value"
        assert instance.optional_date == date(2023, 1, 1)
        assert instance.optional_decimal == Decimal("10.50")
    except ValidationError as e:
        pytest.fail(f"Optional fields should accept values: {e}")


@pytest.mark.django_db
def test_field_requirement_fix_with_invalid_kwargs():
    """
    Test that the field requirement fix correctly handles cases where
    field_info_kwargs is invalid and falls back to checking dj_field.null.
    """
    from django.db import models
    from pydantic import BaseModel, ValidationError

    # Create a test Django model with optional fields
    class TestModelWithOptional(models.Model):
        class Meta:
            app_label = "test_app"

        required_field = models.CharField(max_length=100)
        optional_field = models.CharField(max_length=100, null=True, blank=True)

    # Set up the mapper and generate Pydantic model
    mapper = BidirectionalTypeMapper(RelationshipConversionAccessor())
    generated_model = generate_pydantic_class(TestModelWithOptional, mapper, model_name="TestOptionalModel")

    # Verify the generated model has the correct field requirements
    assert issubclass(generated_model, BaseModel)

    # Test that optional field can be omitted
    try:
        instance = generated_model(required_field="test")
        assert instance.required_field == "test"
        assert instance.optional_field is None
    except ValidationError as e:
        pytest.fail(f"Optional field should not be required: {e}")

    # Test that optional field can be provided
    try:
        instance = generated_model(
            required_field="test",
            optional_field="optional value"
        )
        assert instance.optional_field == "optional value"
    except ValidationError as e:
        pytest.fail(f"Optional field should accept values: {e}")
