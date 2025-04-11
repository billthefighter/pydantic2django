"""Base Django model class with Pydantic conversion capabilities."""
import dataclasses  # Added import
import importlib
import uuid
from typing import Any, ClassVar, Generic, Optional, TypeVar, cast

from django.db import models
from pydantic import BaseModel

# Corrected import path for ModelContext
from ..core.context import ModelContext

# Corrected import path for serialization
from ..core.serialization import serialize_value

# Type variable for BaseModel subclasses
PydanticT = TypeVar("PydanticT", bound=BaseModel)
# Type variable for Dataclass instances
DataclassT = TypeVar("DataclassT")


class CommonBaseModel(models.Model):
    """
    Abstract base class for storing serializable Python objects in the database.

    Provides common functionality for storing Pydantic models or dataclasses.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    # Rename object_type to class_path for clarity
    class_path = models.CharField(
        max_length=255,
        help_text="Fully qualified Python path of the stored object's class (e.g., my_module.MyClass)",
        db_index=True,  # Add index for potential lookups
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    # Removed object_type_field as it's redundant with class_path

    # Class-level cache for imported classes
    _class_cache: ClassVar[dict[str, type]] = {}

    class Meta:
        abstract = True
        # Add default ordering?
        # ordering = ['name']

    @classmethod
    def _get_class_info(cls, obj: Any) -> tuple[type, str, str, str]:
        """
        Get information about the object's class.

        Args:
            obj: The Pydantic model or dataclass instance.

        Returns:
            Tuple of (obj_class, class_name, module_name, fully_qualified_name)
        """
        obj_class = obj.__class__
        class_name = obj_class.__name__
        module_name = obj_class.__module__
        fully_qualified_name = f"{module_name}.{class_name}"
        return obj_class, class_name, module_name, fully_qualified_name

    @classmethod
    def _derive_name(cls, obj: Any, name: str | None, class_name: str) -> str:
        """
        Derive a name for the Django model instance.

        Args:
            obj: The Pydantic model or dataclass instance.
            name: Optional name provided by the caller.
            class_name: The name of the object's class.

        Returns:
            The derived name.
        """
        if name is not None:
            return name

        # Try to get the name from the object if it has a 'name' attribute
        obj_name = getattr(obj, "name", None)
        if isinstance(obj_name, str) and obj_name:
            return obj_name

        # Fallback to the class name
        return class_name

    def _get_class(self) -> type:
        """
        Get the stored object's class (Pydantic model or dataclass) from its path.

        Returns:
            The class type.

        Raises:
            ValueError: If the class cannot be found or imported.
        """
        if not self.class_path:
            raise ValueError("Cannot load class: 'class_path' field is empty.")

        try:
            # Check cache first
            if self.class_path in self.__class__._class_cache:
                return self.__class__._class_cache[self.class_path]

            # Parse module and class name
            module_path, class_name = self.class_path.rsplit(".", 1)
            # Import the module and get the class
            module = importlib.import_module(module_path)
            loaded_class = getattr(module, class_name)

            # Store in cache
            self.__class__._class_cache[self.class_path] = loaded_class
            return loaded_class
        except (ImportError, AttributeError, ValueError) as e:
            raise ValueError(f"Could not find or import class '{self.class_path}': {e}") from e

    def _verify_object_type_match(self, obj: Any) -> str:
        """
        Verify that the provided object's type matches the stored class path.

        Args:
            obj: The Pydantic model or dataclass instance to check.

        Returns:
            The fully qualified name of the object's class.

        Raises:
            TypeError: If the object's type doesn't match the stored class path.
        """
        obj_class = obj.__class__
        class_name = obj_class.__name__
        module_name = obj_class.__module__
        fully_qualified_name = f"{module_name}.{class_name}"

        # Direct comparison of the fully qualified path
        if self.class_path != fully_qualified_name:
            raise TypeError(
                f"Object type mismatch: Expected instance of '{self.class_path}', "
                f"but got instance of '{fully_qualified_name}'."
            )

        return fully_qualified_name


class Dataclass2DjangoBase(CommonBaseModel):
    """
    Abstract base class for storing Python Dataclass objects in the database.
    Inherits common fields and methods from CommonBaseModel.
    """

    # Add specific attributes or methods for dataclasses if needed later
    _expected_dataclass_type: ClassVar[Optional[type]] = None

    class Meta:
        abstract = True

    @classmethod
    def _check_expected_type(cls, dc_obj: Any, class_name: str) -> None:
        """
        Check if the object is a dataclass and matches the expected type (if set).

        Args:
            dc_obj: The object to check.
            class_name: The name of the object's class (for error messages).

        Raises:
            TypeError: If the object is not a dataclass or doesn't match the expected type.
        """
        if not dataclasses.is_dataclass(dc_obj):
            raise TypeError(f"Object provided is not a dataclass: type={type(dc_obj)}")

        expected_type = getattr(cls, "_expected_dataclass_type", None)
        if expected_type is not None:
            # Ensure expected_type is also a dataclass type for comparison
            if not dataclasses.is_dataclass(expected_type):
                # This indicates a configuration error in the subclass
                raise TypeError(
                    f"Internal configuration error: _expected_dataclass_type '{expected_type}' is not a dataclass."
                )
            if not isinstance(dc_obj, expected_type):
                expected_name = getattr(expected_type, "__name__", str(expected_type))
                raise TypeError(f"Expected dataclass object of type {expected_name}, but got {class_name}")


class Pydantic2DjangoBase(CommonBaseModel):
    """
    Abstract base class for storing Pydantic objects in the database.
    Inherits common fields and methods from CommonBaseModel.
    """

    _expected_pydantic_type: ClassVar[Optional[type[BaseModel]]] = None

    class Meta:
        abstract = True

    @classmethod
    def _check_expected_type(cls, pydantic_obj: Any, class_name: str) -> None:
        """
        Check if the Pydantic object is of the expected type.

        Args:
            pydantic_obj: The Pydantic object to check.
            class_name: The name of the Pydantic class (for error messages).

        Raises:
            TypeError: If the Pydantic object is not a Pydantic BaseModel or not of the expected type.
        """
        if not isinstance(pydantic_obj, BaseModel):
            raise TypeError(f"Object provided is not a Pydantic BaseModel: type={type(pydantic_obj)}")

        expected_type = getattr(cls, "_expected_pydantic_type", None)
        if expected_type is not None:
            # Ensure expected_type is a BaseModel subclass for comparison
            if not issubclass(expected_type, BaseModel):
                # This indicates a configuration error in the subclass
                raise TypeError(
                    f"Internal configuration error: _expected_pydantic_type '{expected_type}' is not a Pydantic BaseModel."
                )
            if not isinstance(pydantic_obj, expected_type):
                expected_name = getattr(expected_type, "__name__", str(expected_type))
                raise TypeError(f"Expected Pydantic object of type {expected_name}, but got {class_name}")

    # Methods previously here (_get_pydantic_class_info, _derive_name, _get_pydantic_class,
    # _verify_object_type_match) are now generalized in CommonBaseModel.


# --- Store Full Object as JSON ---


class Dataclass2DjangoStoreDataclassObject(Dataclass2DjangoBase):
    """
    Class to store a Python Dataclass object in the database as JSON.

    All data is stored in the 'data' field. Database fields matching dataclass
    fields can be optionally synced.
    """

    data = models.JSONField(help_text="JSON representation of the dataclass object.")

    class Meta:
        abstract = True
        verbose_name = "Stored Dataclass Object"
        verbose_name_plural = "Stored Dataclass Objects"

    @classmethod
    def from_dataclass(cls, dc_obj: Any, name: str | None = None) -> "Dataclass2DjangoStoreDataclassObject":
        """
        Create a Django model instance from a Dataclass object.

        Args:
            dc_obj: The Dataclass object to store.
            name: Optional name for the Django model instance.

        Returns:
            A new instance of this Django model.

        Raises:
            TypeError: If the provided object is not a dataclass or not of the expected type.
        """
        # Get class info and check type
        (
            dc_class,
            class_name,
            module_name,
            fully_qualified_name,
        ) = cls._get_class_info(dc_obj)
        cls._check_expected_type(dc_obj, class_name)  # Verifies it's a dataclass

        # Get data and serialize values
        try:
            data_dict = dataclasses.asdict(dc_obj)
        except TypeError as e:
            raise TypeError(f"Could not convert dataclass '{class_name}' to dict: {e}") from e

        serialized_data = {key: serialize_value(value) for key, value in data_dict.items()}

        # Derive name
        derived_name = cls._derive_name(dc_obj, name, class_name)

        instance = cls(
            name=derived_name,
            class_path=fully_qualified_name,
            data=serialized_data,
        )
        # Optionally sync fields immediately after creation if desired
        # instance.sync_db_fields_from_data(save=False)
        return instance

    def to_dataclass(self) -> Any:
        """
        Convert the stored JSON data back to a Dataclass object.

        Returns:
            The reconstructed Dataclass object.

        Raises:
            ValueError: If the class cannot be loaded or instantiation fails.
        """
        dataclass_type = self._get_class()
        if not dataclasses.is_dataclass(dataclass_type):
            raise ValueError(f"Stored class path '{self.class_path}' does not point to a dataclass.")

        # Use the stored JSON data
        stored_data = self.data

        # Basic reconstruction (does not handle complex types or context yet)
        try:
            # TODO: Add deserialization logic if serialize_value performs complex transformations
            # For now, assume stored data is directly usable.
            instance = dataclass_type(**stored_data)
        except TypeError as e:
            raise ValueError(
                f"Failed to instantiate dataclass '{dataclass_type.__name__}' from stored data. "
                f"Ensure stored data keys match dataclass fields. Error: {e}"
            ) from e
        except Exception as e:
            raise ValueError(f"An unexpected error occurred during dataclass reconstruction: {e}") from e

        return instance

    def update_from_dataclass(self, dc_obj: Any) -> None:
        """
        Update this Django model instance with new data from a Dataclass object.

        Args:
            dc_obj: The Dataclass object with updated data.

        Raises:
            TypeError: If the object type doesn't match or conversion fails.
        """
        # Verify the object type matches the stored path
        fully_qualified_name = self._verify_object_type_match(dc_obj)
        # Check if it's actually a dataclass (redundant if verify works, but safe)
        if not dataclasses.is_dataclass(dc_obj):
            raise TypeError("Provided object for update is not a dataclass.")

        # Update class_path if somehow inconsistent (shouldn't happen if verify passed)
        if self.class_path != fully_qualified_name:
            self.class_path = fully_qualified_name  # Correctness check

        # Get new data and serialize
        try:
            data_dict = dataclasses.asdict(dc_obj)
        except TypeError as e:
            raise TypeError(f"Could not convert dataclass '{dc_obj.__class__.__name__}' to dict for update: {e}") from e

        self.data = {key: serialize_value(value) for key, value in data_dict.items()}
        # Optionally sync fields before saving
        # self.sync_db_fields_from_data(save=False)
        self.save()

    def sync_db_fields_from_data(self, save: bool = True) -> None:
        """
        Synchronize database fields from the JSON 'data' field.

        Updates Django model fields (excluding common/meta fields) with values
        from the JSON data if the field names match.

        Args:
            save: If True (default), saves the instance after updating fields.
        """
        if not isinstance(self.data, dict):
            # Log or handle cases where data is not a dict
            return

        # Get model fields, excluding common ones and the data field itself
        model_field_names = {
            field.name
            for field in self._meta.fields
            if field.name not in ("id", "name", "class_path", "data", "created_at", "updated_at")
        }

        updated_fields = []
        for field_name in model_field_names:
            if field_name in self.data:
                current_value = getattr(self, field_name)
                new_value = self.data[field_name]
                # Basic check to avoid unnecessary updates
                # TODO: Consider type coercion or more robust comparison if needed
                if current_value != new_value:
                    setattr(self, field_name, new_value)
                    updated_fields.append(field_name)

        if updated_fields and save:
            self.save(update_fields=updated_fields)


class Pydantic2DjangoStorePydanticObject(Pydantic2DjangoBase):
    """
    Class to store a Pydantic object in the database as JSON.
    """

    data = models.JSONField(help_text="JSON representation of the Pydantic object.")

    class Meta:
        abstract = True
        verbose_name = "Stored Pydantic Object"
        verbose_name_plural = "Stored Pydantic Objects"

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
        ) = cls._get_class_info(pydantic_obj)

        # Check if this is a subclass with a specific expected type
        cls._check_expected_type(pydantic_obj, class_name)

        # Get data from the Pydantic object and serialize any nested objects
        # Use model_dump for Pydantic v2+
        try:
            data = pydantic_obj.model_dump()
        except AttributeError:  # Fallback for older Pydantic? Or raise error?
            raise TypeError("Failed to dump Pydantic model. Ensure you are using Pydantic v2+ with model_dump().")

        serialized_data = {key: serialize_value(value) for key, value in data.items()}

        # Use class_name as name if not provided and if object has a name attribute
        derived_name = cls._derive_name(pydantic_obj, name, class_name)

        instance = cls(
            name=derived_name,
            class_path=fully_qualified_name,  # Use renamed field
            data=serialized_data,
        )
        # Optionally sync fields
        # instance.sync_db_fields_from_data(save=False)
        return instance

    def to_pydantic(self, context: Optional[dict[str, Any]] = None) -> Any:
        """
        Convert the stored data back to a Pydantic object.

        Args:
            context: Optional dictionary containing context values for non-serializable fields

        Returns:
            The reconstructed Pydantic object

        Raises:
            ValueError: If required context is missing for non-serializable fields or class load/instantiation fails.
        """
        pydantic_class = self._get_class()  # Use common method
        if not issubclass(pydantic_class, BaseModel):
            raise ValueError(f"Stored class path '{self.class_path}' does not point to a Pydantic BaseModel.")

        # TODO: Integrate ModelContext logic properly if needed for this storage type
        # This likely belongs more in the field-mapping approach.
        # model_context = None
        # if not model_context:
        #     raise NotImplementedError("You should fix this")
        # # If we have context fields, validate the provided context
        # if model_context and model_context.required_context_keys:
        #     if not context:
        #         raise ValueError(
        #             f"This model has non-serializable fields that require context: "
        #             f"{', '.join(model_context.required_context_keys)}. "
        #             "Please provide the context dictionary when calling to_pydantic()."
        #         )
        #     model_context.validate_context(context)

        # Use stored data directly
        stored_data = self.data

        # If context is provided, overlay it (simple merge, context takes precedence)
        final_data = stored_data.copy()
        if context:
            final_data.update(context)

        # Reconstruct the object using model_validate for Pydantic v2+
        try:
            # TODO: Add deserialization logic if serialize_value performs complex transformations
            instance = pydantic_class.model_validate(final_data)
        except Exception as e:  # Catch Pydantic validation errors etc.
            raise ValueError(
                f"Failed to validate/instantiate Pydantic model '{pydantic_class.__name__}' from stored data: {e}"
            ) from e

        return instance

    def _get_data_with_db_overrides(self, pydantic_class: type[BaseModel]) -> dict:
        """
        Get model data with any database field overrides applied.
        (Primarily relevant for field-mapping, less so here, but kept for potential sync logic).
        """
        # This method seems less relevant when the primary source is `self.data`.
        # Keeping it simple: return the stored data.
        # If sync logic were more complex, this might need adjustment.
        return self.data if isinstance(self.data, dict) else {}

    def update_from_pydantic(self, pydantic_obj: Any) -> None:
        """
        Update this object with new data from a Pydantic object.

        Args:
            pydantic_obj: The Pydantic object with updated data
        """
        # Verify the object type matches
        fully_qualified_name = self._verify_object_type_match(pydantic_obj)
        # Check if it's a Pydantic model (redundant if verify works, but safe)
        if not isinstance(pydantic_obj, BaseModel):
            raise TypeError("Provided object for update is not a Pydantic BaseModel.")

        # Update the class_path if somehow inconsistent
        if self.class_path != fully_qualified_name:
            self.class_path = fully_qualified_name

        # Use model_dump for Pydantic v2+
        try:
            data = pydantic_obj.model_dump()
        except AttributeError:
            raise TypeError(
                "Failed to dump Pydantic model for update. Ensure you are using Pydantic v2+ with model_dump()."
            )

        self.data = {key: serialize_value(value) for key, value in data.items()}
        # Optionally sync fields
        # self.sync_db_fields_from_data(save=False)
        self.save()

    def sync_db_fields_from_data(self, save: bool = True) -> None:
        """
        Synchronize database fields from the JSON 'data' field.

        Updates Django model fields (excluding common/meta fields) with values
        from the JSON data if the field names match.

        Args:
            save: If True (default), saves the instance after updating fields.
        """
        if not isinstance(self.data, dict):
            return

        # Get model fields, excluding common ones and the data field itself
        model_field_names = {
            field.name
            for field in self._meta.fields
            if field.name not in ("id", "name", "class_path", "data", "created_at", "updated_at")
        }

        updated_fields = []
        for field_name in model_field_names:
            if field_name in self.data:
                current_value = getattr(self, field_name)
                new_value = self.data[field_name]
                # Basic check to avoid unnecessary updates
                if current_value != new_value:
                    setattr(self, field_name, new_value)
                    updated_fields.append(field_name)

        if updated_fields and save:
            self.save(update_fields=updated_fields)


# --- Map Object Fields to Database Fields ---


class Dataclass2DjangoBaseClass(Dataclass2DjangoBase, Generic[DataclassT]):
    """
    Base class for mapping Python Dataclass fields to Django model fields.

    Inherits from Dataclass2DjangoBase and provides methods to convert
    between the Dataclass instance and the Django model instance by matching field names.
    """

    class Meta:
        abstract = True
        verbose_name = "Mapped Dataclass"
        verbose_name_plural = "Mapped Dataclasses"

    # __getattr__ is less likely needed/useful for standard dataclasses compared to Pydantic models
    # which might have complex methods. Skip for now.

    @classmethod
    def from_dataclass(cls, dc_obj: DataclassT, name: str | None = None) -> "Dataclass2DjangoBaseClass[DataclassT]":
        """
        Create a Django model instance from a Dataclass object, mapping fields.

        Args:
            dc_obj: The Dataclass object to convert.
            name: Optional name for the Django model instance.

        Returns:
            A new instance of this Django model subclass.

        Raises:
            TypeError: If the object is not a dataclass or not of the expected type.
        """
        # Get class info and check type
        (
            dc_class,
            class_name,
            module_name,
            fully_qualified_name,
        ) = cls._get_class_info(dc_obj)
        cls._check_expected_type(dc_obj, class_name)  # Verifies it's a dataclass

        # Derive name
        derived_name = cls._derive_name(dc_obj, name, class_name)

        # Create instance with basic fields
        instance = cls(
            name=derived_name,
            class_path=fully_qualified_name,
        )

        # Update mapped fields
        instance.update_fields_from_dataclass(dc_obj)

        return instance

    def update_fields_from_dataclass(self, dc_obj: DataclassT) -> None:
        """
        Update this Django model's fields from a Dataclass object's fields.

        Args:
            dc_obj: The Dataclass object containing source values.

        Raises:
            TypeError: If conversion to dict fails.
        """
        if (
            not dataclasses.is_dataclass(dc_obj)
            or dc_obj.__class__.__module__ != self._get_class().__module__
            or dc_obj.__class__.__name__ != self._get_class().__name__
        ):
            # Check type consistency before proceeding
            raise TypeError(
                f"Provided object type {type(dc_obj)} does not match expected type {self.class_path} for update."
            )

        try:
            dc_data = dataclasses.asdict(dc_obj)
        except TypeError as e:
            raise TypeError(f"Could not convert dataclass '{dc_obj.__class__.__name__}' to dict for update: {e}") from e

        # Get Django model fields excluding common/meta ones
        model_field_names = {
            field.name
            for field in self._meta.fields
            if field.name not in ("id", "name", "class_path", "created_at", "updated_at")
        }

        for field_name in model_field_names:
            if field_name in dc_data:
                value = dc_data[field_name]
                # Apply serialization (important for complex types like datetime, UUID, etc.)
                serialized_value = serialize_value(value)
                setattr(self, field_name, serialized_value)
            # Else: Field exists on Django model but not on dataclass, leave it unchanged.

    def to_dataclass(self) -> DataclassT:
        """
        Convert this Django model instance back to a Dataclass object.

        Returns:
            The reconstructed Dataclass object.

        Raises:
            ValueError: If the class cannot be loaded or instantiation fails.
        """
        dataclass_type = self._get_class()
        if not dataclasses.is_dataclass(dataclass_type):
            raise ValueError(f"Stored class path '{self.class_path}' does not point to a dataclass.")

        # Get data from Django fields corresponding to dataclass fields
        data_for_dc = self._get_data_for_dataclass(dataclass_type)

        # Instantiate the dataclass
        try:
            # TODO: Add deserialization logic if needed
            instance = dataclass_type(**data_for_dc)
            # Cast to the generic type variable for type hinting
            return cast(DataclassT, instance)
        except TypeError as e:
            raise ValueError(
                f"Failed to instantiate dataclass '{dataclass_type.__name__}' from Django model fields. "
                f"Ensure required fields exist and types are compatible. Error: {e}"
            ) from e
        except Exception as e:
            raise ValueError(f"An unexpected error occurred during dataclass reconstruction: {e}") from e

    def _get_data_for_dataclass(self, dataclass_type: type) -> dict[str, Any]:
        """Get data from Django fields that correspond to the target dataclass fields."""
        data = {}
        try:
            dc_field_names = {f.name for f in dataclasses.fields(dataclass_type)}
        except TypeError:
            # Should not happen if is_dataclass check passed, but handle defensively
            raise ValueError(f"Could not get fields for non-dataclass type '{dataclass_type.__name__}'")

        # Add DB fields that are part of the dataclass
        for field in self._meta.fields:
            if field.name in dc_field_names:
                # TODO: Add potential deserialization based on target dataclass field type?
                data[field.name] = getattr(self, field.name)

        # Context handling is usually Pydantic-specific, skip for dataclasses unless needed
        return data

    def update_from_dataclass(self, dc_obj: DataclassT) -> None:
        """
        Update this Django model with new data from a Dataclass object and save.

        Args:
            dc_obj: The Dataclass object with updated data.
        """
        # Verify the object type matches first (includes check if it's a dataclass)
        fully_qualified_name = self._verify_object_type_match(dc_obj)

        # Update the class_path if somehow inconsistent
        if self.class_path != fully_qualified_name:
            self.class_path = fully_qualified_name

        self.update_fields_from_dataclass(dc_obj)
        self.save()

    def save_as_dataclass(self) -> DataclassT:
        """
        Save the Django model and return the corresponding Dataclass object.

        Returns:
            The corresponding Dataclass object.
        """
        self.save()
        return self.to_dataclass()


class Pydantic2DjangoBaseClass(Pydantic2DjangoBase, Generic[PydanticT]):
    """
    Base class for mapping Pydantic model fields to Django model fields.
    """

    class Meta:
        abstract = True
        verbose_name = "Mapped Pydantic Object"
        verbose_name_plural = "Mapped Pydantic Objects"

    def __new__(cls, *args, **kwargs):
        """
        Override __new__ to ensure proper type checking.
        Needed because Django's model metaclass doesn't preserve Generic type parameters well.
        """
        # The check itself might be complex. This placeholder ensures __new__ is considered.
        # Proper generic handling might require metaclass adjustments beyond this scope.
        return super().__new__(cls)

    def __getattr__(self, name: str) -> Any:
        """
        Forward method calls to the Pydantic model implementation.
        Enables type checking and execution of methods defined on the Pydantic model.
        """
        # Get the Pydantic model class
        try:
            pydantic_cls = self._get_class()  # Use common method
            if not issubclass(pydantic_cls, BaseModel):
                # This path shouldn't be hit if used correctly, but safeguard
                raise AttributeError(f"Stored class '{self.class_path}' is not a Pydantic BaseModel.")
        except ValueError as e:
            raise AttributeError(f"Cannot forward attribute '{name}': {e}") from e

        # Check if the attribute exists in the Pydantic model
        if hasattr(pydantic_cls, name):
            attr = getattr(pydantic_cls, name)

            # If it's a callable method (and not the type itself), wrap it
            if callable(attr) and not isinstance(attr, type):

                def wrapped_method(*args, **kwargs):
                    # Convert self (Django model) to Pydantic instance first
                    try:
                        pydantic_instance = self.to_pydantic()  # Assuming no context needed here
                    except ValueError as e:
                        # Handle potential errors during conversion (e.g., context missing)
                        raise RuntimeError(
                            f"Failed to convert Django model to Pydantic before calling '{name}': {e}"
                        ) from e

                    # Call the method on the Pydantic instance
                    result = getattr(pydantic_instance, name)(*args, **kwargs)
                    # TODO: Handle potential need to update self from result? Unlikely for most methods.
                    return result

                return wrapped_method
            else:
                # For non-method attributes (like class vars), return directly
                # This might need refinement depending on desired behavior for class vs instance attrs
                return attr

        # If attribute doesn't exist on Pydantic model, raise standard AttributeError
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}', "
            f"and Pydantic model '{pydantic_cls.__name__}' has no attribute '{name}'."
        )

    @classmethod
    def from_pydantic(cls, pydantic_obj: PydanticT, name: str | None = None) -> "Pydantic2DjangoBaseClass[PydanticT]":
        """
        Create a Django model instance from a Pydantic object, mapping fields.

        Args:
            pydantic_obj: The Pydantic object to convert.
            name: Optional name for the Django model instance.

        Returns:
            A new instance of this Django model subclass.

        Raises:
            TypeError: If the object is not a Pydantic model or not of the expected type.
        """
        # Get class info and check type
        (
            pydantic_class,
            class_name,
            module_name,
            fully_qualified_name,
        ) = cls._get_class_info(pydantic_obj)
        cls._check_expected_type(pydantic_obj, class_name)  # Verifies it's a Pydantic model

        # Derive name
        derived_name = cls._derive_name(pydantic_obj, name, class_name)

        # Create instance with basic fields
        instance = cls(
            name=derived_name,
            class_path=fully_qualified_name,  # Use renamed field
        )

        # Update mapped fields
        instance.update_fields_from_pydantic(pydantic_obj)

        return instance

    def update_fields_from_pydantic(self, pydantic_obj: PydanticT) -> None:
        """
        Update this Django model's fields from a Pydantic object's fields.

        Args:
            pydantic_obj: The Pydantic object containing source values.
        """
        if (
            not isinstance(pydantic_obj, BaseModel)
            or pydantic_obj.__class__.__module__ != self._get_class().__module__
            or pydantic_obj.__class__.__name__ != self._get_class().__name__
        ):
            # Check type consistency before proceeding
            raise TypeError(
                f"Provided object type {type(pydantic_obj)} does not match expected type {self.class_path} for update."
            )

        # Get data from the Pydantic object
        try:
            pydantic_data = pydantic_obj.model_dump()
        except AttributeError:
            raise TypeError(
                "Failed to dump Pydantic model for update. Ensure you are using Pydantic v2+ with model_dump()."
            )

        # Get Django model fields excluding common/meta ones
        model_field_names = {
            field.name
            for field in self._meta.fields
            if field.name not in ("id", "name", "class_path", "created_at", "updated_at")
        }

        # Update each Django field if it matches a field in the Pydantic data
        for field_name in model_field_names:
            if field_name in pydantic_data:
                value = pydantic_data[field_name]
                # Apply serialization (important for complex types)
                serialized_value = serialize_value(value)
                setattr(self, field_name, serialized_value)
            # Else: Field exists on Django model but not on Pydantic model, leave it unchanged.

    def to_pydantic(self, context: Optional[ModelContext] = None) -> PydanticT:
        """
        Convert this Django model instance back to a Pydantic object.

        Args:
            context: Optional ModelContext instance containing values for non-serializable fields.

        Returns:
            The corresponding Pydantic object.

        Raises:
            ValueError: If context is required but not provided, or if class load/instantiation fails.
        """
        pydantic_class = self._get_class()  # Use common method
        if not issubclass(pydantic_class, BaseModel):
            raise ValueError(f"Stored class path '{self.class_path}' does not point to a Pydantic BaseModel.")

        # Get data from Django fields corresponding to Pydantic fields
        data = self._get_data_for_pydantic(pydantic_class)

        # Handle context if required and provided
        required_context_keys = self._get_required_context_fields()  # Check if context is needed
        if required_context_keys:
            if not context:
                raise ValueError(
                    f"Conversion to Pydantic model '{pydantic_class.__name__}' requires context "
                    f"for fields: {', '.join(required_context_keys)}. Please provide a ModelContext instance."
                )
            # Validate and merge context data
            context_dict = context.to_conversion_dict()
            context.validate_context(context_dict)  # Validate required keys are present
            data.update(context_dict)  # Merge context, potentially overwriting DB values if keys overlap

        # Reconstruct the Pydantic object
        try:
            # TODO: Add potential deserialization logic here if needed before validation
            instance = pydantic_class.model_validate(data)
            # Cast to the generic type variable
            return cast(PydanticT, instance)
        except Exception as e:  # Catch Pydantic validation errors etc.
            raise ValueError(
                f"Failed to validate/instantiate Pydantic model '{pydantic_class.__name__}' from Django fields: {e}"
            ) from e

    def _get_data_for_pydantic(self, pydantic_class: type[BaseModel]) -> dict[str, Any]:
        """Get data from Django fields that correspond to the target Pydantic model fields."""
        data = {}
        try:
            pydantic_field_names = set(pydantic_class.model_fields.keys())
        except AttributeError:
            # Should not happen if issubclass(BaseModel) check passed
            raise ValueError(f"Could not get fields for non-Pydantic type '{pydantic_class.__name__}'")

        # Add DB fields that are part of the Pydantic model
        for field in self._meta.fields:
            if field.name in pydantic_field_names:
                # TODO: Add potential deserialization based on target Pydantic field type?
                data[field.name] = getattr(self, field.name)

        # Context values are merged in the calling `to_pydantic` method
        return data

    def _get_required_context_fields(self) -> set[str]:
        """
        Get the set of field names that require context when converting to Pydantic.
        (Placeholder implementation - needs refinement based on how context is defined).
        """
        # This requires a mechanism to identify which Django fields represent
        # non-serializable data that must come from context.
        # For now, assume no context is required by default for the base class.
        # Subclasses might override this or a more sophisticated mechanism could be added.
        # Example: Check for a custom field attribute like `is_context_field=True`
        required_fields = set()
        # pydantic_class = self._get_class()
        # pydantic_field_names = set(pydantic_class.model_fields.keys())
        # for field in self._meta.fields:
        #     if field.name in pydantic_field_names and getattr(field, 'is_context_field', False):
        #         required_fields.add(field.name)
        return required_fields  # Return empty set for now

    def update_from_pydantic(self, pydantic_obj: PydanticT) -> None:
        """
        Update this Django model with new data from a Pydantic object and save.

        Args:
            pydantic_obj: The Pydantic object with updated data.
        """
        # Verify the object type matches first (includes check if it's a BaseModel)
        fully_qualified_name = self._verify_object_type_match(pydantic_obj)

        # Update the class_path if somehow inconsistent
        if self.class_path != fully_qualified_name:
            self.class_path = fully_qualified_name

        self.update_fields_from_pydantic(pydantic_obj)
        self.save()

    def save_as_pydantic(self) -> PydanticT:
        """
        Save the Django model and return the corresponding Pydantic object.

        Returns:
            The corresponding Pydantic object.
        """
        self.save()
        # Pass None for context; assumes save_as doesn't need external context.
        # If context might be needed, this method signature/logic needs adjustment.
        return self.to_pydantic(context=None)  # Or handle context if necessary
