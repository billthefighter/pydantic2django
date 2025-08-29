import logging
from typing import Any

from django.db import models

# Import necessary functions from other utils modules
from ...core.utils.strings import sanitize_string
from .naming import get_related_model_name, sanitize_related_name

logger = logging.getLogger(__name__)


class RawCode:
    """A wrapper to inject raw code strings into the generated output."""

    def __init__(self, code: str):
        self.code = code

    def __repr__(self):
        return self.code


class FieldSerializer:
    """
    Handles extraction and processing of field attributes into string form
    from Django model fields for code generation.
    """

    @staticmethod
    def serialize_field_attributes(field: models.Field) -> list[str]:
        """
        Serialize common Django model field attributes to a list of parameter strings.

        Args:
            field: The Django model field instance.

        Returns:
            List of parameter strings (e.g., "null=True", "max_length=255").
        """
        params = []
        field_name = getattr(field, "name", "?")  # For logging

        # Common field parameters
        try:
            if hasattr(field, "verbose_name") and field.verbose_name:
                # Check if verbose_name is different from the auto-generated one
                auto_verbose_name = field.name.replace("_", " ").capitalize()
                if str(field.verbose_name) != auto_verbose_name:
                    params.append(f"verbose_name='{sanitize_string(field.verbose_name)}'")

            if hasattr(field, "help_text") and field.help_text:
                params.append(f"help_text='{sanitize_string(field.help_text)}'")

            # Explicitly include null/blank only if they differ from the default for the field type
            # (Most fields default to null=False, blank=False)
            if hasattr(field, "null") and field.null:
                params.append(f"null={field.null}")

            if hasattr(field, "blank") and field.blank:
                # Only add blank=True if null=True is also set, common Django pattern
                if hasattr(field, "null") and field.null:
                    params.append(f"blank={field.blank}")
                else:
                    logger.debug(
                        f"Field '{field_name}' has blank=True but null=False. Omitting blank=True from serialization."
                    )

            # Handle choices
            if hasattr(field, "choices") and field.choices:
                # TODO: Need a robust way to serialize choices, might be complex tuples/enums
                try:
                    choices_repr = repr(field.choices)  # Basic repr, might need improvement
                    params.append(f"choices={choices_repr}")
                except Exception as e:
                    logger.warning(f"Could not serialize choices for field '{field_name}': {e}")

            # Handle default value
            if hasattr(field, "default") and field.default is not models.NOT_PROVIDED:
                # Skip default for AutoFields as it's implicit
                if not isinstance(field, (models.AutoField, models.BigAutoField)):
                    try:
                        default_repr = repr(field.default)  # Use repr for safety
                        # Avoid adding default=None if null=True is already implied
                        if not (field.null and field.default is None):
                            params.append(f"default={default_repr}")
                    except Exception as e:
                        logger.warning(f"Could not serialize default value for field '{field_name}': {e}")

            # Field-specific parameters
            if isinstance(field, (models.CharField, models.SlugField, models.FilePathField)):
                # Only add max_length if it's not the default (which varies)
                # We need a way to know the default for each field type if we want to omit defaults.
                # For now, always include it if present.
                if hasattr(field, "max_length") and field.max_length is not None:
                    params.append(f"max_length={field.max_length}")

            if isinstance(field, models.DecimalField):
                # Defaults are max_digits=None, decimal_places=None
                if hasattr(field, "max_digits") and field.max_digits is not None:
                    params.append(f"max_digits={field.max_digits}")
                if hasattr(field, "decimal_places") and field.decimal_places is not None:
                    params.append(f"decimal_places={field.decimal_places}")

        except Exception as e:
            logger.error(f"Error serializing attributes for field '{field_name}': {e}", exc_info=True)

        return params

    @staticmethod
    def serialize_field(field: models.Field) -> str:
        """
        Serialize a Django model field to its string representation for code generation.

        Args:
            field: The Django model field instance.

        Returns:
            String representation (e.g., "models.CharField(max_length=255, null=True)").
        """
        field_name = getattr(field, "name", "?")
        logger.debug(f"Serializing field: {field_name} (Type: {type(field).__name__})")
        try:
            field_type_name = type(field).__name__
            params = FieldSerializer.serialize_field_attributes(field)

            # Handle relationship fields specifically
            if isinstance(field, (models.ForeignKey, models.OneToOneField, models.ManyToManyField)):
                related_model_name = get_related_model_name(field)
                if not related_model_name:
                    # Attempt to get from field.related_model as another fallback
                    related_model_obj = getattr(field, "related_model", None)
                    if related_model_obj:
                        # Avoid accessing __name__ directly on Literal['self']
                        if isinstance(related_model_obj, str) and related_model_obj == "self":
                            related_model_name = "self"
                        else:
                            related_model_name = getattr(related_model_obj, "__name__", str(related_model_obj))

                if related_model_name:
                    # Quote the model name (might be 'app.Model' or 'Model')
                    params.append(f"to='{related_model_name}'")
                else:
                    # This should ideally not happen if discovery worked correctly
                    logger.error(
                        f"CRITICAL: Could not determine related model for relationship field '{field_name}'. Defaulting to 'self'."
                    )
                    params.append("to='self'")

                # on_delete for ForeignKey/OneToOneField
                if isinstance(field, (models.ForeignKey, models.OneToOneField)):
                    try:
                        # Use remote_field preferentially
                        remote_field = getattr(field, "remote_field", None)
                        if remote_field and hasattr(remote_field, "on_delete"):
                            on_delete_func = remote_field.on_delete
                            on_delete_name = getattr(on_delete_func, "__name__", None)
                            if on_delete_name:
                                params.append(f"on_delete=models.{on_delete_name}")
                            else:
                                logger.warning(
                                    f"Could not determine on_delete name for field '{field_name}'. Defaulting to CASCADE."
                                )
                                params.append("on_delete=models.CASCADE")  # Sensible default
                        else:
                            logger.warning(
                                f"Could not find remote_field or on_delete for field '{field_name}'. Defaulting to CASCADE."
                            )
                            params.append("on_delete=models.CASCADE")
                    except AttributeError as e:
                        logger.warning(
                            f"AttributeError determining on_delete for '{field_name}': {e}. Defaulting to CASCADE."
                        )
                        params.append("on_delete=models.CASCADE")

                # related_name (common to all relationships)
                # Use remote_field preferentially
                remote_field_obj = getattr(field, "remote_field", None)
                related_name = getattr(remote_field_obj, "related_name", None)
                if related_name:
                    # Sanitize related_name before adding
                    params.append(f"related_name='{sanitize_related_name(related_name, field_name=field_name)}'")

                # ManyToMany specific attributes
                if isinstance(field, models.ManyToManyField):
                    remote_m2m_field = getattr(field, "remote_field", None)
                    through_model = getattr(remote_m2m_field, "through", None) if remote_m2m_field else None
                    # Check auto_created on the through_model itself if it exists
                    auto_created_meta = False
                    if through_model and not isinstance(through_model, str) and hasattr(through_model, "_meta"):
                        auto_created_meta = getattr(through_model._meta, "auto_created", False)

                    if through_model and not isinstance(through_model, str) and not auto_created_meta:
                        through_name = getattr(through_model, "__name__", str(through_model))
                        params.append(f"through={through_name}")
                    elif isinstance(through_model, str):
                        params.append(f"through='{through_model}'")  # Use string directly

            # Construct final definition string
            param_str = ", ".join(params)
            final_def = f"models.{field_type_name}({param_str})"
            logger.debug(f"Serialized definition for '{field_name}': {final_def}")
            return final_def

        except Exception as e:
            logger.error(f"Failed to serialize field '{field_name}' (Type: {type(field).__name__}): {e}", exc_info=True)
            # Fallback to a simple TextField to avoid crashing generation
            return f"models.TextField(help_text='Serialization failed for field {field_name}: {e}')"


def generate_field_definition_string(
    field_class: type[models.Field],
    field_kwargs: dict[str, Any],
    app_label: str,  # Needed for resolving 'self' in relationships
) -> str:
    """
    Generates the string representation of a Django field definition from its class and kwargs.

    Args:
        field_class: The Django field class (e.g., models.CharField).
        field_kwargs: A dictionary of keyword arguments for the field.
        app_label: The app label of the model the field belongs to.

    Returns:
        The field definition string (e.g., "models.CharField(max_length=255)").
    """
    param_parts = []
    # Need to import sanitize_string
    from ...core.utils.strings import sanitize_string

    sorted_kwargs = sorted(field_kwargs.items())

    for key, value in sorted_kwargs:
        # Special handling for values that should be raw code
        if isinstance(value, RawCode):
            param_parts.append(f"{key}={value.code}")
            continue

        # Special handling for relationship 'to' and 'through' fields
        if key == "to" or key == "through":
            model_name_str = None
            if isinstance(value, str):
                # Use string directly (e.g., 'self', 'app_label.ModelName')
                model_name_str = value
            elif isinstance(value, type) and issubclass(value, models.Model):
                # Get name from model type
                try:
                    # Prefer app_label.ModelName format if possible
                    meta = getattr(value, "_meta", None)
                    if meta:
                        app_label_val = getattr(meta, "app_label", None)
                        object_name_val = getattr(meta, "object_name", None)  # Use object_name for class name
                        if app_label_val and object_name_val:
                            model_name_str = f"{app_label_val}.{object_name_val}"
                        elif object_name_val:
                            model_name_str = object_name_val  # Fallback to just name
                        else:
                            model_name_str = value.__name__  # Final fallback
                    else:
                        model_name_str = value.__name__  # Fallback if no _meta
                except AttributeError:
                    model_name_str = value.__name__  # Fallback on error
            else:
                # Fallback for unexpected types
                logger.warning(f"Unexpected type for relationship '{key}' argument: {type(value)}. Using repr().")
                model_name_str = repr(value)

            if model_name_str:
                # Ensure self is quoted correctly
                if model_name_str == "self":
                    param_parts.append(f"{key}='self'")
                else:
                    param_parts.append(f"{key}='{model_name_str}'")
            else:
                logger.error(
                    f"Could not determine model name for relationship '{key}' argument: {value}. Omitting from definition."
                )

        # Special handling for on_delete
        elif key == "on_delete":
            # Allow RawCode override for on_delete
            if isinstance(value, RawCode):
                param_parts.append(f"on_delete={value.code}")
            elif callable(value) and hasattr(value, "__name__"):
                param_parts.append(f"on_delete=models.{value.__name__}")
            else:
                # Default to CASCADE if unresolvable
                param_parts.append("on_delete=models.CASCADE")
                logger.warning(f"Could not serialize on_delete value: {value}. Defaulting to CASCADE.")
        # General kwarg serialization
        else:
            try:
                # Use sanitize_string for string values to handle quotes/escapes
                if isinstance(value, str):
                    repr_value = f"'{sanitize_string(value)}'"
                # Use repr for others (bool, int, float, complex types like choices)
                else:
                    repr_value = repr(value)
                param_parts.append(f"{key}={repr_value}")
            except Exception as e:
                logger.warning(f"Could not serialize kwarg '{key}={value}' for field class {field_class.__name__}: {e}")
                param_parts.append(f"{key}=None  # Serialization failed")

    param_str = ", ".join(param_parts)
    return f"models.{field_class.__name__}({param_str})"
