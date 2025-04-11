import re
from typing import Optional

from django.db import models


def sanitize_related_name(name: str, model_name: str = "", field_name: str = "") -> str:
    """
    Sanitize a related name for Django models.

    Ensures the name is a valid Python identifier, replacing invalid characters
    and providing fallbacks if the name is empty.

    Args:
        name: The name to sanitize
        model_name: Optional model name for context in fallback generation
        field_name: Optional field name for context in fallback generation

    Returns:
        A sanitized related name suitable for Django's related_name attribute.
    """
    # Replace spaces and special characters with underscores
    sanitized = re.sub(r"[^\w]", "_", str(name))  # Ensure input is string

    # Ensure it starts with a letter or underscore
    if sanitized and not sanitized[0].isalpha() and sanitized[0] != "_":
        sanitized = f"_{sanitized}"

    # If it's empty after sanitization, generate a fallback name
    if not sanitized:
        if model_name and field_name:
            sanitized = f"{model_name.lower()}_{field_name.lower()}_set"
        elif model_name:
            sanitized = f"{model_name.lower()}_set"
        elif field_name:
            sanitized = f"{field_name.lower()}_set"
        else:
            sanitized = "related_items"  # Generic fallback

    # Ensure it's not a Python keyword (add underscore if it is)
    # Requires importing keyword module
    # import keyword
    # if keyword.iskeyword(sanitized):
    #     sanitized += "_"

    # Ensure it doesn't end with '+' (invalid related_name suffix)
    if sanitized.endswith("+"):
        sanitized = sanitized.rstrip("+") + "_"

    # Django convention: lower case
    return sanitized.lower()


def get_related_model_name(field: models.Field) -> Optional[str]:
    """
    Safely get the related model name from a Django relationship field.

    Handles potential errors and different ways the related model might be specified.

    Args:
        field: The Django relationship field (ForeignKey, ManyToManyField, OneToOneField).

    Returns:
        The name of the related model (e.g., 'AppName.ModelName' or 'ModelName')
        or None if it cannot be determined.
    """
    related_model = None
    try:
        if hasattr(field, "remote_field") and field.remote_field:
            related_model = field.remote_field.model
        elif hasattr(field, "related_model"):  # Older Django versions might use this
            related_model = field.related_model

        if related_model:
            if isinstance(related_model, str):
                # If it's already a string 'app.Model' or 'Model', use it
                return related_model
            elif hasattr(related_model, "_meta"):
                # Standard case: access _meta.label or construct from app_label and model_name
                if hasattr(related_model._meta, "label"):
                    return related_model._meta.label  # 'app_label.ModelName'
                else:
                    app_label = getattr(related_model._meta, "app_label", None)
                    model_name = getattr(related_model._meta, "model_name", None)
                    if app_label and model_name:
                        return f"{app_label}.{model_name}"
                    elif model_name:  # Fallback if only model name is available
                        return model_name
            elif hasattr(related_model, "__name__"):
                # Fallback for simpler cases or potential custom models without _meta
                return related_model.__name__

    except Exception:
        # logger.warning(f"Could not determine related model name for field {getattr(field, 'name', '?')}: {e}")
        pass

    # logger.warning(f"Failed to determine related model name for field {getattr(field, 'name', '?')}")
    return None
