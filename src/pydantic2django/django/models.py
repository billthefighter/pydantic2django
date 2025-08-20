"""Base Django model class with Pydantic conversion capabilities."""

import dataclasses
import importlib
import inspect
import logging
import uuid
from typing import Any, ClassVar, Generic, TypeVar, cast

from django.db import models
from pydantic import BaseModel

# Corrected import path for ModelContext
from ..core.context import ModelContext

# Corrected import path for serialization
from ..core.serialization import serialize_value

logger = logging.getLogger(__name__)  # Ensure logger is defined globally for the file

# Type variable for BaseModel subclasses
PydanticT = TypeVar("PydanticT", bound=BaseModel)
# Type variable for Dataclass instances
DataclassT = TypeVar("DataclassT")

# --- TypedClass Base Functionality ---

TypedClassT = TypeVar("TypedClassT")  # Generic type for TypedClass instances


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


class TypedClass2DjangoBase(CommonBaseModel):
    """
    Abstract base class for storing generic Python class objects in the database.
    Inherits common fields and methods from CommonBaseModel.
    """

    _expected_typedclass_type: ClassVar[type | None] = None  # For type checking specific generic classes

    class Meta(CommonBaseModel.Meta):
        abstract = True

    @classmethod
    def _check_expected_type(cls, typed_obj: Any, class_name: str) -> None:
        """
        Check if the object matches the expected type (if set).
        For generic classes, this might be a simple type check or more involved
        if we introduce markers or specific base classes for discoverable typed classes.
        """
        # Basic check: is it an instance of the class itself?
        # More advanced checks could be added, e.g., if it inherits from a specific
        # user-defined base for conversion, or has a special attribute.

        expected_type = getattr(cls, "_expected_typedclass_type", None)
        if expected_type is not None:
            if not isinstance(typed_obj, expected_type):
                expected_name = getattr(expected_type, "__name__", str(expected_type))
                raise TypeError(f"Expected object of type {expected_name}, but got {class_name}")
        # Unlike Pydantic/Dataclass, there's no single inspectable "is_generic_class" function.
        # The discovery mechanism should handle identifying suitable classes.


# --- Store Full TypedClass Object as JSON ---


class TypedClass2DjangoStoreTypedClassObject(TypedClass2DjangoBase, Generic[TypedClassT]):
    """
    Class to store a generic Python class object in the database as JSON.
    All data is stored in the 'data' field.
    """

    data = models.JSONField(help_text="JSON representation of the typed class object.")

    class Meta(TypedClass2DjangoBase.Meta):
        abstract = True
        verbose_name = "Stored Typed Class Object"
        verbose_name_plural = "Stored Typed Class Objects"

    @classmethod
    def _extract_data_from_typedclass(cls, typed_obj: TypedClassT) -> dict[str, Any]:
        """
        Helper to extract data from a generic class instance for serialization.
        This is a key challenge and might need to be configurable.
        Initial approach: inspect __init__ params and class vars with type hints.
        More advanced: __dict__, __slots__, or a user-defined method.
        """
        data_dict = {}
        # Attempt 1: Inspect __init__ parameters if they correspond to attributes
        try:
            sig = inspect.signature(typed_obj.__class__.__init__)
            for param_name in sig.parameters:
                if param_name == "self":
                    continue
                if hasattr(typed_obj, param_name):
                    data_dict[param_name] = getattr(typed_obj, param_name)
        except (TypeError, ValueError):  # Non-inspectable __init__
            pass  # Fallback to __dict__ or other methods

        # Attempt 2: Use __dict__ if available and not already covered (simplistic)
        if hasattr(typed_obj, "__dict__") and not data_dict:  # Prioritize __init__ if it yielded something
            data_dict.update(typed_obj.__dict__)

        # Attempt 3: Iterate over class annotations if available (for class vars not in __init__ / __dict__)
        # This part requires careful handling to avoid unintended data or methods.
        # For now, a basic __dict__ or __init__ based extraction is safer.
        # Consider a __slots__ check as well if present.

        if not data_dict:
            logger.warning(
                f"Could not automatically extract data from {typed_obj.__class__.__name__}. "
                f"Consider implementing a 'to_dict()' method or using __slots__."
            )

        return {key: serialize_value(value) for key, value in data_dict.items() if not key.startswith("_")}

    @classmethod
    def from_typedclass(
        cls, typed_obj: TypedClassT, name: str | None = None
    ) -> "TypedClass2DjangoStoreTypedClassObject[TypedClassT]":
        """
        Create a Django model instance from a generic class object.
        """
        obj_class, class_name, module_name, fqn = cls._get_class_info(typed_obj)
        cls._check_expected_type(typed_obj, class_name)

        data_dict = cls._extract_data_from_typedclass(typed_obj)
        derived_name = cls._derive_name(typed_obj, name, class_name)

        instance = cls(
            name=derived_name,
            class_path=fqn,
            data=data_dict,
        )
        return instance

    def to_typedclass(self) -> TypedClassT:
        """
        Convert the stored JSON data back to a generic class object.
        This is highly challenging and relies on the class having a suitable __init__.
        """
        target_class = self._get_class()
        stored_data = self.data if isinstance(self.data, dict) else {}

        # Attempt to instantiate using stored_data.
        # This assumes keys in stored_data match __init__ parameters.
        # Deserialization of complex nested values from serialize_value needs to be handled.
        # For now, assume stored data is directly usable or simple types.
        # TODO: Implement robust deserialization matching serialize_value
        deserialized_data = {}
        for key, value in stored_data.items():
            # Placeholder: a proper deserialization function is needed here.
            # For example, if serialize_value converted datetimes to ISO strings,
            # this would need to parse them back.
            deserialized_data[key] = value

        try:
            # Filter data to only include parameters expected by __init__
            init_sig = inspect.signature(target_class.__init__)
            valid_params = {k: v for k, v in deserialized_data.items() if k in init_sig.parameters}

            instance = target_class(**valid_params)
            # For attributes not in __init__ but in stored_data (e.g. from __dict__),
            # attempt to setattr them if they are not private.
            for key, value in deserialized_data.items():
                if key not in valid_params and hasattr(instance, key) and not key.startswith("_"):
                    try:
                        setattr(instance, key, value)
                    except AttributeError:  # Read-only property, etc.
                        logger.debug(
                            f"Could not setattr {key} on {target_class.__name__} during to_typedclass reconstruction."
                        )
            return cast(TypedClassT, instance)

        except TypeError as e:
            raise ValueError(
                f"Failed to instantiate typed class '{target_class.__name__}' from stored data. "
                f"Ensure stored data keys match __init__ parameters or class attributes. Error: {e}"
            ) from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during typed class reconstruction: {e}", exc_info=True)
            raise ValueError(f"An unexpected error occurred during typed class reconstruction: {e}") from e

    def update_from_typedclass(self, typed_obj: TypedClassT) -> None:
        """
        Update this Django model instance with new data from a generic class object.
        """
        fqn = self._verify_object_type_match(typed_obj)
        if self.class_path != fqn:
            self.class_path = fqn

        self.data = self._extract_data_from_typedclass(typed_obj)
        self.save()


# --- Map TypedClass Object Fields to Database Fields ---


class TypedClass2DjangoBaseClass(TypedClass2DjangoBase, Generic[TypedClassT]):
    """
    Base class for mapping generic Python class fields to Django model fields.
    """

    class Meta(TypedClass2DjangoBase.Meta):
        abstract = True
        verbose_name = "Mapped Typed Class"
        verbose_name_plural = "Mapped Typed Classes"

    @classmethod
    def from_typedclass(
        cls, typed_obj: TypedClassT, name: str | None = None
    ) -> "TypedClass2DjangoBaseClass[TypedClassT]":
        """
        Create a Django model instance from a generic class object, mapping fields.
        """
        obj_class, class_name, module_name, fqn = cls._get_class_info(typed_obj)
        cls._check_expected_type(typed_obj, class_name)
        derived_name = cls._derive_name(typed_obj, name, class_name)

        instance = cls(
            name=derived_name,
            class_path=fqn,
        )
        instance.update_fields_from_typedclass(typed_obj)
        return instance

    def update_fields_from_typedclass(self, typed_obj: TypedClassT) -> None:
        """
        Update this Django model's fields from a generic class object's fields.
        """
        if (
            typed_obj.__class__.__module__ != self._get_class().__module__
            or typed_obj.__class__.__name__ != self._get_class().__name__
        ):
            raise TypeError(
                f"Provided object type {type(typed_obj)} does not match expected type {self.class_path} for update."
            )

        # Extract data. For mapped fields, we prefer __init__ params or direct attrs.
        # Using __dict__ might be too broad here if not all dict items are mapped fields.
        # Let's assume attributes corresponding to Django fields are directly accessible.

        model_field_names = {
            field.name
            for field in self._meta.fields
            if field.name not in ("id", "name", "class_path", "created_at", "updated_at")
        }

        for field_name in model_field_names:
            if hasattr(typed_obj, field_name):
                value = getattr(typed_obj, field_name)
                setattr(self, field_name, serialize_value(value))
            # Else: Field on Django model, not on typed_obj. Leave as is or handle.

    def to_typedclass(self) -> TypedClassT:
        """
        Convert this Django model instance back to a generic class object.
        """
        target_class = self._get_class()
        data_for_typedclass = self._get_data_for_typedclass(target_class)

        try:
            # This assumes target_class can be instantiated with these parameters.
            # Deserialization from serialize_value needs to be considered for complex types.
            # TODO: Implement robust deserialization
            deserialized_data = dict(data_for_typedclass.items())  # Placeholder

            init_sig = inspect.signature(target_class.__init__)
            valid_params = {k: v for k, v in deserialized_data.items() if k in init_sig.parameters}

            instance = target_class(**valid_params)

            # For attributes not in __init__ but set on Django model, try to setattr
            for key, value in deserialized_data.items():
                if key not in valid_params and hasattr(instance, key) and not key.startswith("_"):
                    try:
                        setattr(instance, key, value)  # value is already deserialized_data[key]
                    except AttributeError:
                        logger.debug(
                            f"Could not setattr {key} on {target_class.__name__} during to_typedclass reconstruction."
                        )

            return cast(TypedClassT, instance)
        except TypeError as e:
            raise ValueError(
                f"Failed to instantiate typed class '{target_class.__name__}' from Django fields. Error: {e}"
            ) from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during typed class reconstruction: {e}", exc_info=True)
            raise ValueError(f"An unexpected error occurred during typed class reconstruction: {e}") from e

    def _get_data_for_typedclass(self, target_class: type) -> dict[str, Any]:
        """Get data from Django fields that correspond to the target class's likely attributes."""
        data = {}
        # Heuristic: get attributes from __init__ signature and class annotations
        # This is a simplified version. A robust solution might need to align with TypedClassFieldFactory's discovery.

        # Potential attributes: from __init__
        potential_attrs = set()
        try:
            init_sig = inspect.signature(target_class.__init__)
            potential_attrs.update(p for p in init_sig.parameters if p != "self")
        except (TypeError, ValueError):
            pass

        # Potential attributes: from class annotations (less reliable for instance data if not in __init__)
        # try:
        #     annotations = inspect.get_annotations(target_class)
        #     potential_attrs.update(annotations.keys())
        # except Exception:
        #     pass

        for field in self._meta.fields:
            if field.name in potential_attrs or hasattr(target_class, field.name):  # Check if it's a likely attribute
                # TODO: Add deserialization logic based on target_class's type hint for field.name
                data[field.name] = getattr(self, field.name)
        return data

    def update_from_typedclass(self, typed_obj: TypedClassT) -> None:
        """
        Update this Django model with new data from a generic class object and save.
        """
        fqn = self._verify_object_type_match(typed_obj)
        if self.class_path != fqn:
            self.class_path = fqn

        self.update_fields_from_typedclass(typed_obj)
        self.save()

    def save_as_typedclass(self) -> TypedClassT:
        """
        Save the Django model and return the corresponding generic class object.
        """
        self.save()
        return self.to_typedclass()


class Dataclass2DjangoBase(CommonBaseModel):
    """
    Abstract base class for storing Python Dataclass objects in the database.
    Inherits common fields and methods from CommonBaseModel.
    """

    # Add specific attributes or methods for dataclasses if needed later
    _expected_dataclass_type: ClassVar[type | None] = None

    class Meta(CommonBaseModel.Meta):
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

    _expected_pydantic_type: ClassVar[type[BaseModel] | None] = None

    class Meta(CommonBaseModel.Meta):
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

    class Meta(Dataclass2DjangoBase.Meta):
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
            logger.error(f"An unexpected error occurred during dataclass reconstruction: {e}", exc_info=True)
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

    class Meta(Pydantic2DjangoBase.Meta):
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
        try:
            obj_class, class_name, module_name, fqn = cls._get_class_info(pydantic_obj)
            cls._check_expected_type(pydantic_obj, class_name)

            data_dict = pydantic_obj.model_dump(mode="json")
            derived_name = cls._derive_name(pydantic_obj, name, class_name)

            instance = cls(name=derived_name, class_path=fqn, data=data_dict)
            instance.sync_db_fields_from_data(save=False)
            return instance

        except TypeError:
            raise  # Re-raise TypeError as it's a specific, meaningful exception here
        except Exception as e:
            # Catch any other unexpected error during Pydantic object creation
            logger.error(f"An unexpected error occurred creating Django model for '{class_name}': {e}", exc_info=True)
            raise ValueError(f"Unexpected error creating Django model for '{class_name}'") from e

    def to_pydantic(self, context: dict[str, Any] | None = None) -> Any:
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
        except TypeError:
            raise  # Re-raise TypeError as it's a specific, meaningful exception here
        except Exception as e:
            # Catch any other unexpected error during Pydantic object creation
            logger.error(
                f"An unexpected error occurred creating Pydantic model for '{pydantic_class.__name__}': {e}",
                exc_info=True,
            )
            raise ValueError(f"Unexpected error creating Pydantic model for '{pydantic_class.__name__}'") from e

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
        except AttributeError as err:
            raise TypeError(
                "Failed to dump Pydantic model for update. Ensure you are using Pydantic v2+ with model_dump()."
            ) from err

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

    class Meta(Dataclass2DjangoBase.Meta):
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
            logger.error(f"An unexpected error occurred during dataclass reconstruction: {e}", exc_info=True)
            raise ValueError(f"An unexpected error occurred during dataclass reconstruction: {e}") from e

    def _get_data_for_dataclass(self, dataclass_type: type) -> dict[str, Any]:
        """Get data from Django fields that correspond to the target dataclass fields."""
        data = {}
        try:
            dc_field_names = {f.name for f in dataclasses.fields(dataclass_type)}
        except TypeError as err:
            # Should not happen if is_dataclass check passed, but handle defensively
            raise ValueError(f"Could not get fields for non-dataclass type '{dataclass_type.__name__}'") from err

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

    class Meta(Pydantic2DjangoBase.Meta):
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
        except AttributeError as err:
            raise TypeError(
                "Failed to dump Pydantic model for update. Ensure you are using Pydantic v2+ with model_dump()."
            ) from err

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

    def to_pydantic(self, context: ModelContext | None = None) -> PydanticT:
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
        except TypeError:
            raise  # Re-raise TypeError as it's a specific, meaningful exception here
        except Exception as e:
            # Catch any other unexpected error during Pydantic object creation
            logger.error(
                f"An unexpected error occurred creating Pydantic model for '{pydantic_class.__name__}': {e}",
                exc_info=True,
            )
            raise ValueError(f"Unexpected error creating Pydantic model for '{pydantic_class.__name__}'") from e

    def _get_data_for_pydantic(self, pydantic_class: type[BaseModel]) -> dict[str, Any]:
        """Get data from Django fields that correspond to the target Pydantic model fields."""
        data = {}
        try:
            pydantic_field_names = set(pydantic_class.model_fields.keys())
        except AttributeError as err:
            # Should not happen if issubclass(BaseModel) check passed
            raise ValueError(f"Could not get fields for non-Pydantic type '{pydantic_class.__name__}'") from err

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
        return self.to_pydantic(context=None)


# XML Schema Base Functionality


class Xml2DjangoBaseClass(models.Model):
    """
    Base class for Django models generated from XML Schema definitions.
    Provides XML-specific conversion methods and metadata handling.
    """

    # Schema linking fields
    conversion_session = models.UUIDField(null=True, blank=True, help_text="Links to the conversion session")
    schema_source = models.CharField(max_length=255, null=True, blank=True, help_text="Source XSD file")

    # XML Schema metadata (can be overridden in generated models)
    _xml_namespace: ClassVar[str | None] = None
    _xml_schema_location: ClassVar[str | None] = None
    _xml_type_name: ClassVar[str | None] = None

    class Meta:
        abstract = True

    @classmethod
    def from_xml_dict(cls, xml_data: dict[str, Any], **kwargs) -> "Xml2DjangoBaseClass":
        """
        Create Django model instance from XML data dictionary.

        Args:
            xml_data: Dictionary representation of XML data
            **kwargs: Additional field values to override

        Returns:
            Django model instance
        """
        # Convert XML data (field names might need conversion from XML naming)
        data = cls._convert_xml_data(xml_data)

        # Override with any provided kwargs
        data.update(kwargs)

        # Create Django instance
        return cls(**data)

    @classmethod
    def _convert_xml_data(cls, xml_data: dict[str, Any]) -> dict[str, Any]:
        """
        Convert XML data dictionary to Django model field format.
        Can be overridden to handle XML-to-Django field name mapping.
        """
        # Basic implementation - can be enhanced to handle:
        # - camelCase to snake_case conversion
        # - XML attribute vs element handling
        # - Namespace handling
        converted = {}
        for key, value in xml_data.items():
            # Skip XML metadata
            if key.startswith("_") or key.startswith("@"):
                continue

            # Convert camelCase to snake_case for Django field names
            django_field_name = cls._xml_name_to_django_field(key)
            converted[django_field_name] = value

        return converted

    @classmethod
    def _xml_name_to_django_field(cls, xml_name: str) -> str:
        """Convert XML element/attribute name to Django field name."""
        # Simple camelCase to snake_case conversion
        import re

        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", xml_name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    def to_xml_dict(self, include_metadata: bool = False) -> dict[str, Any]:
        """
        Convert Django model instance to XML-compatible dictionary.

        Args:
            include_metadata: Whether to include XML namespace and schema information

        Returns:
            Dictionary suitable for XML serialization
        """
        data = {}

        # Get field values
        for field in self._meta.fields:
            value = getattr(self, field.name)
            if value is not None:
                # Convert Django field name back to XML name
                xml_name = self._django_field_to_xml_name(field.name)
                data[xml_name] = self._serialize_field_value(value, field)

        # Include metadata if requested
        if include_metadata:
            if self._xml_namespace:
                data["_namespace"] = self._xml_namespace
            if self._xml_schema_location:
                data["_schema_location"] = self._xml_schema_location
            if self._xml_type_name:
                data["_type_name"] = self._xml_type_name

        return data

    def _django_field_to_xml_name(self, field_name: str) -> str:
        """Convert Django field name back to XML element/attribute name."""
        # Simple snake_case to camelCase conversion
        components = field_name.split("_")
        return components[0] + "".join(word.capitalize() for word in components[1:])

    def _serialize_field_value(self, value: Any, field: models.Field) -> Any:
        """Serialize field value for XML output."""
        # Handle different field types
        if isinstance(field, (models.DateTimeField, models.DateField, models.TimeField)):
            return value.isoformat() if value else None
        elif isinstance(field, models.UUIDField):
            return str(value) if value else None
        elif isinstance(field, (models.ForeignKey, models.OneToOneField)):
            # For relationships, return the related object's primary key
            return value.pk if value else None
        elif isinstance(field, models.ManyToManyField):
            # For M2M, return list of primary keys
            return [obj.pk for obj in value.all()] if value else []
        else:
            return value

    def to_xml_string(self, root_element_name: str | None = None, include_declaration: bool = True) -> str:
        """
        Convert Django model instance to XML string.

        Args:
            root_element_name: Name for the root XML element (defaults to model class name)
            include_declaration: Whether to include XML declaration

        Returns:
            XML string representation
        """
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise ImportError("XML support requires xml.etree.ElementTree")

        root_name = root_element_name or self._xml_type_name or self.__class__.__name__
        root = ET.Element(root_name)

        # Add namespace if available
        if self._xml_namespace:
            root.set("xmlns", self._xml_namespace)

        # Convert to dict and build XML
        data = self.to_xml_dict()
        for key, value in data.items():
            if key.startswith("_"):  # Skip metadata
                continue
            if value is not None:
                if isinstance(value, list):
                    # Handle lists (from maxOccurs="unbounded")
                    for item in value:
                        elem = ET.SubElement(root, key)
                        elem.text = str(item)
                else:
                    elem = ET.SubElement(root, key)
                    elem.text = str(value)

        # Convert to string
        xml_str = ET.tostring(root, encoding="unicode")

        if include_declaration:
            xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str

        return xml_str

    @classmethod
    def from_xml_string(cls, xml_string: str) -> "Xml2DjangoBaseClass":
        """
        Create Django model instance from XML string.

        Args:
            xml_string: XML string to parse

        Returns:
            Django model instance
        """
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise ImportError("XML support requires xml.etree.ElementTree")

        root = ET.fromstring(xml_string)
        data = {}

        # Extract elements
        for elem in root:
            if elem.tag in data:
                # Handle multiple elements with same name (lists)
                if not isinstance(data[elem.tag], list):
                    data[elem.tag] = [data[elem.tag]]
                data[elem.tag].append(elem.text)
            else:
                data[elem.tag] = elem.text

        # Extract attributes (prefixed with @)
        for attr_name, attr_value in root.attrib.items():
            if not attr_name.startswith("{"):  # Skip namespace declarations
                data[f"@{attr_name}"] = attr_value

        return cls.from_xml_dict(data)


# --- Optional TimescaleDB integration base ---
# Provides a combined abstract base that includes TimescaleDB support when available.
try:
    # django-timescaledb package import path
    from timescale.db.models.models import (
        TimescaleModel,  # type: ignore[import-not-found]
    )
except Exception:  # pragma: no cover - optional dependency
    # Fallback no-op mixin when TimescaleDB is not installed
    class TimescaleModel:  # type: ignore[no-redef]
        pass


class XmlTimescaleBase(Xml2DjangoBaseClass, TimescaleModel):
    """Abstract base class combining XML helpers with TimescaleDB features.

    Inherits from `Xml2DjangoBaseClass` and `timescale.db.models.models.TimescaleModel`.
    When the TimescaleDB package is not installed, `TimescaleModel` is a no-op mixin.
    """

    class Meta:
        abstract = True
