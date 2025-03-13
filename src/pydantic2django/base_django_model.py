import importlib
import uuid
from typing import Any, ClassVar, Generic, TypeVar, cast

from django.db import models
from pydantic import BaseModel

from .serialization import serialize_value

# Type variable for BaseModel subclasses
T = TypeVar("T", bound=BaseModel)


class Pydantic2DjangoBase(models.Model):
    """
    Abstract base class for storing Pydantic objects in the database.

    This class provides common functionality for both storage approaches:
    1. Storing the entire Pydantic object as JSON (Pydantic2DjangoStorePydanticObject)
    2. Mapping Pydantic fields to Django model fields (Pydantic2DjangoBaseClass)
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    object_type = models.CharField(
        max_length=255,
        help_text="Fully qualified name of the Pydantic model class",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Class-level cache for imported Pydantic classes
    _pydantic_class_cache: ClassVar[dict] = {}

    class Meta:
        abstract = True

    @classmethod
    def _get_pydantic_class_info(cls, pydantic_obj: Any) -> tuple[Any, str, str, str]:
        """
        Get information about the Pydantic class.

        Args:
            pydantic_obj: The Pydantic object

        Returns:
            Tuple of (pydantic_class, class_name, module_name, fully_qualified_name)
        """
        pydantic_class = pydantic_obj.__class__
        class_name = pydantic_class.__name__
        module_name = pydantic_class.__module__
        fully_qualified_name = f"{module_name}.{class_name}"
        return pydantic_class, class_name, module_name, fully_qualified_name

    @classmethod
    def _check_expected_type(cls, pydantic_obj: Any, class_name: str) -> None:
        """
        Check if the Pydantic object is of the expected type.

        Args:
            pydantic_obj: The Pydantic object to check
            class_name: The name of the Pydantic class

        Raises:
            TypeError: If the Pydantic object is not of the expected type
        """
        expected_type = getattr(cls, "_expected_pydantic_type", None)
        if expected_type is not None:
            if not isinstance(pydantic_obj, expected_type):
                expected_name = getattr(expected_type, "__name__", str(expected_type))
                raise TypeError(f"Expected Pydantic object of type {expected_name}, " f"but got {class_name}")

    @classmethod
    def _derive_name(cls, pydantic_obj: Any, name: str | None, class_name: str) -> str:
        """
        Derive a name for the Django model instance.

        Args:
            pydantic_obj: The Pydantic object
            name: Optional name provided by the caller
            class_name: The name of the Pydantic class

        Returns:
            The derived name
        """
        if name is not None:
            return name

        # Try to get the name from the Pydantic object if it has a name attribute
        try:
            obj_name = getattr(pydantic_obj, "name", None)
            if obj_name is not None:
                return obj_name
            return class_name
        except (AttributeError, TypeError):
            return class_name

    def _get_pydantic_class(self) -> type[BaseModel]:
        """
        Get the Pydantic class for this Django model instance.

        Returns:
            The Pydantic class

        Raises:
            ValueError: If the Pydantic class cannot be found
        """
        try:
            # Check if the class is already in the cache
            cache_key = self.object_type
            if cache_key in self.__class__._pydantic_class_cache:
                pydantic_class = self.__class__._pydantic_class_cache[cache_key]
            else:
                # Parse the module and class name from object_type
                module_path, class_name = self.object_type.rsplit(".", 1)
                # Import the appropriate class
                module = importlib.import_module(module_path)
                pydantic_class = getattr(module, class_name)
                # Store in the cache
                self.__class__._pydantic_class_cache[cache_key] = pydantic_class
            return pydantic_class
        except (ImportError, AttributeError, ValueError) as e:
            raise ValueError(f"Could not find Pydantic class for {self.object_type}: {str(e)}") from e

    def _verify_object_type_match(self, pydantic_obj: Any) -> str:
        """
        Verify that the Pydantic object type matches this Django model instance.

        Args:
            pydantic_obj: The Pydantic object to check

        Returns:
            The fully qualified name of the Pydantic class

        Raises:
            TypeError: If the Pydantic object type doesn't match
        """
        pydantic_class = pydantic_obj.__class__
        class_name = pydantic_class.__name__
        module_name = pydantic_class.__module__
        fully_qualified_name = f"{module_name}.{class_name}"

        # Check if the object types match (ignoring module for backward compatibility)
        if not self.object_type.endswith(class_name) and self.object_type != fully_qualified_name:
            raise TypeError(
                f"Expected Pydantic object of type matching {self.object_type}, " f"but got {fully_qualified_name}"
            )

        return fully_qualified_name


class Pydantic2DjangoStorePydanticObject(Pydantic2DjangoBase):
    """
    Class to store a Pydantic object in the database.

    Does not allow field access, all data is stored in the data field as a JSON object.
    """

    data = models.JSONField()

    class Meta:
        abstract = True

    @classmethod
    def from_pydantic(cls, pydantic_obj: Any, name: str | None = None) -> "Pydantic2DjangoStorePydanticObject":
        """
        Create a Django model instance from a Pydantic object.

        Args:
            pydantic_obj: The Pydantic object to store
            name: Optional name for the object (defaults to class name if available)

        Returns:
            A new instance of the appropriate Pydantic2DjangoBaseClass subclass

        Raises:
            TypeError: If the Pydantic object is not of the correct type for this Django model
        """
        # Get the Pydantic class and its fully qualified name
        (
            pydantic_class,
            class_name,
            module_name,
            fully_qualified_name,
        ) = cls._get_pydantic_class_info(pydantic_obj)

        # Check if this is a subclass with a specific expected type
        cls._check_expected_type(pydantic_obj, class_name)

        # Get data from the Pydantic object and serialize any nested objects
        data = pydantic_obj.model_dump()
        serialized_data = {key: serialize_value(value) for key, value in data.items()}

        # Use class_name as name if not provided and if object has a name attribute
        name = cls._derive_name(pydantic_obj, name, class_name)

        instance = cls(
            name=name,
            object_type=fully_qualified_name,
            data=serialized_data,
        )

        return instance

    def to_pydantic(self) -> Any:
        """
        Convert the stored data back to a Pydantic object.

        Returns:
            The reconstructed Pydantic object
        """
        pydantic_class = self._get_pydantic_class()

        # Get data with database field overrides
        data = self._get_data_with_db_overrides(pydantic_class)

        # Reconstruct the object
        return pydantic_class.model_validate(data)

    def _get_data_with_db_overrides(self, pydantic_class: Any) -> dict[str, Any]:
        """
        Get the JSON data with database field overrides.

        This method checks if any database fields match fields in the Pydantic model
        and if they do, it uses the database field value instead of the JSON value.

        Args:
            pydantic_class: The Pydantic class to check fields against

        Returns:
            The data dictionary with database field overrides
        """
        # Start with a copy of the stored JSON data
        data = self.data.copy()

        # Get all fields from the Pydantic model - we'll use the data keys as a fallback
        pydantic_field_names = set(data.keys())

        # Try to get field information from the Pydantic class
        try:
            # For Pydantic v2
            if hasattr(pydantic_class, "model_fields") and isinstance(pydantic_class.model_fields, dict):
                pydantic_field_names.update(pydantic_class.model_fields.keys())
            # For Pydantic v1
            elif hasattr(pydantic_class, "__fields__") and isinstance(pydantic_class.__fields__, dict):
                pydantic_field_names.update(pydantic_class.__fields__.keys())
        except Exception:
            # If we can't get the fields, just use what we have from the data
            pass

        # Get all fields from the Django model
        django_fields = {field.name: field for field in self._meta.fields}

        # Exclude these fields from consideration
        exclude_fields = {
            "id",
            "name",
            "object_type",
            "data",
            "created_at",
            "updated_at",
        }

        # Check each Django field to see if it matches a Pydantic field
        for field_name, _ in django_fields.items():
            if field_name in exclude_fields:
                continue

            # Check if this field exists in the Pydantic model or data
            if field_name in pydantic_field_names:
                # Get the value from the Django model
                value = getattr(self, field_name)

                # Only override if the value is not None (unless the field in data is also None)
                if value is not None or (field_name in data and data[field_name] is None):
                    data[field_name] = value

        return data

    def update_from_pydantic(self, pydantic_obj: Any) -> None:
        """
        Update this object with new data from a Pydantic object.

        Args:
            pydantic_obj: The Pydantic object with updated data
        """
        # Verify the object type matches
        fully_qualified_name = self._verify_object_type_match(pydantic_obj)

        # Update the object_type to the fully qualified name if it's not already
        if self.object_type != fully_qualified_name:
            self.object_type = fully_qualified_name

        self.data = pydantic_obj.model_dump()
        self.save()

    def sync_db_fields_from_data(self) -> None:
        """
        Synchronize database fields from the JSON data.

        This method updates the database fields with values from the JSON data
        if the field names match.
        """
        # Get all fields from the Django model
        django_fields = {field.name: field for field in self._meta.fields}

        # Exclude these fields from consideration
        exclude_fields = {
            "id",
            "name",
            "object_type",
            "data",
            "created_at",
            "updated_at",
        }

        # Check each Django field to see if it matches a field in the JSON data
        updated_fields = []
        for field_name, _ in django_fields.items():
            if field_name in exclude_fields:
                continue

            # Check if this field exists in the JSON data
            if field_name in self.data:
                # Get the value from the JSON data
                value = self.data[field_name]

                # Set the Django field value
                setattr(self, field_name, value)
                updated_fields.append(field_name)

        # Save the changes
        if updated_fields:
            self.save(update_fields=updated_fields)


class Pydantic2DjangoBaseClass(Pydantic2DjangoBase, Generic[T]):
    """
    Base class for storing Pydantic objects in the database.

    This model provides common functionality for serializing and deserializing
    Pydantic objects, with proper type hints and IDE support.
    """

    class Meta:
        abstract = True

    def __new__(cls, *args, **kwargs):
        """
        Override __new__ to ensure proper type checking.
        This is needed because Django's model metaclass doesn't preserve Generic type parameters.
        """
        return super().__new__(cls)

    def __getattr__(self, name: str) -> Any:
        """
        Forward method calls to the Pydantic model implementation.
        This enables proper type checking for methods defined in the Pydantic model.
        """
        # Get the Pydantic model class from object_type, using the cache if available
        try:
            pydantic_cls = self._get_pydantic_class()
        except ValueError as e:
            raise AttributeError(str(e)) from e

        # Check if the attribute exists in the Pydantic model
        if hasattr(pydantic_cls, name):
            # Get the attribute from the Pydantic model
            attr = getattr(pydantic_cls, name)

            # If it's a method, wrap it to convert between Django and Pydantic models
            if callable(attr) and not isinstance(attr, type):
                # Create a wrapper function that converts self to a Pydantic instance
                # and then calls the method on that instance
                def wrapped_method(*args, **kwargs):
                    # Convert Django model to Pydantic model
                    pydantic_instance = self.to_pydantic()
                    # Call the method on the Pydantic instance
                    result = getattr(pydantic_instance, name)(*args, **kwargs)
                    # Return the result
                    return result

                return wrapped_method
            else:
                # For non-method attributes, just return the attribute
                return attr

        # If the attribute doesn't exist in the Pydantic model, raise AttributeError
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}' "
            f"and '{pydantic_cls.__name__}' has no attribute '{name}'"
        )

    @classmethod
    def from_pydantic(cls, pydantic_obj: T, name: str | None = None) -> "Pydantic2DjangoBaseClass[T]":
        """
        Create a Django model instance from a Pydantic object.

        Args:
            pydantic_obj: The Pydantic object to convert
            name: Optional name for the object (defaults to class name if available)

        Returns:
            A new instance of the appropriate Pydantic2DjangoBaseClass subclass

        Raises:
            TypeError: If the Pydantic object is not of the correct type for this Django model
        """
        # Get the Pydantic class and its fully qualified name
        (
            pydantic_class,
            class_name,
            module_name,
            fully_qualified_name,
        ) = cls._get_pydantic_class_info(pydantic_obj)

        # Check if this is a subclass with a specific expected type
        cls._check_expected_type(pydantic_obj, class_name)

        # Use class_name as name if not provided and if object has a name attribute
        name = cls._derive_name(pydantic_obj, name, class_name)

        # Create a new instance with basic fields
        instance = cls(
            name=name,
            object_type=fully_qualified_name,
        )

        # Update fields from the Pydantic object
        instance.update_fields_from_pydantic(pydantic_obj)

        return instance

    def update_fields_from_pydantic(self, pydantic_obj: T) -> None:
        """
        Update this Django model's fields from a Pydantic object.

        Args:
            pydantic_obj: The Pydantic object with field values
        """
        # Get data from the Pydantic object
        data = pydantic_obj.model_dump()

        # Get all fields from the Django model
        django_fields = {field.name: field for field in self._meta.fields}

        # Exclude these fields from consideration
        exclude_fields = {
            "id",
            "name",
            "object_type",
            "created_at",
            "updated_at",
        }

        # Update each Django field if it matches a field in the Pydantic data
        for field_name, _ in django_fields.items():
            if field_name in exclude_fields:
                continue

            # Check if this field exists in the Pydantic data
            if field_name in data:
                # Get the value from the Pydantic data and serialize it if needed
                value = data[field_name]
                serialized_value = serialize_value(value)

                # Set the Django field value
                setattr(self, field_name, serialized_value)

    def to_pydantic(self, context: dict[str, Any] | None = None) -> T:
        """
        Convert this Django model to a Pydantic object.

        Args:
            context: Optional dictionary containing values for non-serializable fields.
                    Required if the model has any non-serializable fields.

        Returns:
            The corresponding Pydantic object

        Raises:
            ValueError: If context is required but not provided, or if provided context
                      is missing required fields.
        """
        pydantic_class = self._get_pydantic_class()

        # Get data from Django fields
        data = self._get_data_for_pydantic()

        # Check if we need context
        required_context = self._get_required_context_fields()
        if required_context:
            if not context:
                raise ValueError(
                    f"This model has non-serializable fields that require context: {', '.join(required_context)}. "
                    "Please provide the context dictionary when calling to_pydantic()."
                )

            # Verify all required fields are in the context
            missing_fields = [field for field in required_context if field not in context]
            if missing_fields:
                raise ValueError(f"Missing required context fields: {', '.join(missing_fields)}")

            # Add context values to data
            data.update({field: context[field] for field in required_context})

        # Reconstruct the object and cast to the correct type
        result = pydantic_class.model_validate(data)
        return cast(T, result)

    def _get_data_for_pydantic(self) -> dict[str, Any]:
        """
        Get the data from Django fields for creating a Pydantic object.

        Returns:
            A dictionary of field values
        """
        # Start with an empty dictionary
        data = {}

        # Get all fields from the Django model
        django_fields = {field.name: field for field in self._meta.fields}

        # Exclude these fields from consideration
        # TODO: There are cases where these fields are defined on the source pydantic object.
        # Need to handle that.
        exclude_fields = {
            "id",
            "name",
            "object_type",
            "created_at",
            "updated_at",
        }

        # TODO: Handle relationship fields
        # TODO: Handle nested models
        # Add each Django field value to the data dictionary
        for field_name, _ in django_fields.items():
            if field_name in exclude_fields:
                continue

            # Get the value from the Django model
            value = getattr(self, field_name)

            # Add to data dictionary
            data[field_name] = value

        return data

    def _get_required_context_fields(self) -> set[str]:
        """
        Get the set of field names that require context when converting to Pydantic.

        Returns:
            Set of field names that require context
        """
        required_fields = set()

        # Get all fields from the Django model
        django_fields = {field.name: field for field in self._meta.fields}

        # Exclude these fields from consideration
        exclude_fields = {
            "id",
            "name",
            "object_type",
            "created_at",
            "updated_at",
        }

        # Check each field
        for field_name, field in django_fields.items():
            if field_name in exclude_fields:
                continue

            # Check if this is a context field (non-serializable)
            # We use the is_relationship flag to indicate context fields
            # This was set in _resolve_field_type
            if isinstance(field, models.TextField) and getattr(field, "is_relationship", False):
                required_fields.add(field_name)

        return required_fields

    def update_from_pydantic(self, pydantic_obj: T) -> None:
        """
        Update this Django model with new data from a Pydantic object.

        Args:
            pydantic_obj: The Pydantic object with updated data
        """
        # Verify the object type matches
        fully_qualified_name = self._verify_object_type_match(pydantic_obj)

        # Update the object_type to the fully qualified name if it's not already
        if self.object_type != fully_qualified_name:
            self.object_type = fully_qualified_name

        self.update_fields_from_pydantic(pydantic_obj)
        self.save()

    def save_as_pydantic(self) -> T:
        """
        Convert to a Pydantic object, save the Django model, and return the Pydantic object.

        This is a convenience method for operations that need to save the Django model
        and then continue working with the Pydantic representation.

        Returns:
            The corresponding Pydantic object
        """
        self.save()
        return self.to_pydantic()
