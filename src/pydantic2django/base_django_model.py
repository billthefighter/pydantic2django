import importlib
import uuid
from typing import Any

from django.db import models


class Pydantic2DjangoBaseClass(models.Model):
    """
    Base class for storing LLMaestro objects in the database.

    This model provides common functionality for serializing and deserializing
    Pydantic objects from the LLMaestro library.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    object_type = models.CharField(max_length=100)
    data = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

    @classmethod
    def from_pydantic(cls, pydantic_obj: Any, name: str | None = None) -> "Pydantic2DjangoBaseClass":
        """
        Create a Django model instance from a Pydantic object.

        Args:
            pydantic_obj: The Pydantic object to store
            name: Optional name for the object (defaults to object_type if available)

        Returns:
            A new instance of the appropriate Pydantic2DjangoBaseClass subclass
        """
        object_type = pydantic_obj.__class__.__name__
        data = pydantic_obj.model_dump()

        # Use object_type as name if not provided and if object has a name attribute
        if name is None and hasattr(pydantic_obj, "name"):
            name = pydantic_obj.name
        elif name is None:
            name = object_type

        return cls(
            name=name,
            object_type=object_type,
            data=data,
        )

    def to_pydantic(self) -> Any:
        """
        Convert the stored data back to a Pydantic object.

        Returns:
            The reconstructed Pydantic object
        """
        # Import the appropriate class
        module_path = self._get_module_path()
        module = importlib.import_module(module_path)
        pydantic_class = getattr(module, self.object_type)

        # Get data with database field overrides
        data = self._get_data_with_db_overrides(pydantic_class)

        # Reconstruct the object
        return pydantic_class.model_validate(data)

    def _get_module_path(self) -> str:
        """
        Get the module path for the Pydantic class.

        This method should be overridden by subclasses to provide the correct module path.

        Returns:
            The module path as a string
        """
        raise NotImplementedError(
            "Subclasses must implement _get_module_path to provide the module path for the Pydantic class"
        )

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
