import re


def clean_field_for_template(field_type: str) -> str:
    """
    Cleans a field type string for use within Django templates.
    Removes problematic characters or constructs.

    Args:
        field_type: The field type string.

    Returns:
        The cleaned field type string suitable for templates.
    """
    # Example cleaning: remove Optional[] wrapper, handle complex generics simply
    cleaned = re.sub(r"Optional\[(.*?)\]", r"\1", field_type)
    # Add more specific cleaning rules as needed
    cleaned = cleaned.replace("list[", "List[")  # Example normalization
    # ... further cleaning ...
    return cleaned


def clean_field_type_for_template(field: dict) -> str:
    """
    Cleans the 'field_type' within a field dictionary for template usage.

    Args:
        field: The field dictionary.

    Returns:
        The cleaned field type string.
    """
    field_type = field.get("field_type", "Any")
    return clean_field_for_template(field_type)
