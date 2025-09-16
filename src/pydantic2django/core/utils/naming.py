import re


def normalize_namespace_separators(name: str) -> str:
    """
    Replace common namespace and punctuation separators with underscores.

    Examples:
    - "xlink:type" -> "xlink_type"
    - "ns.element-name" -> "ns_element_name"
    - "foo bar" -> "foo_bar"
    """
    return re.sub(r"[:\.\-\s]+", "_", str(name))


def to_snake_case(name: str) -> str:
    """
    Convert a possibly CamelCase or mixedCase string into snake_case.

    - Inserts underscores between lower->Upper boundaries
    - Handles multi-capital sequences reasonably (ABCTest -> abc_test)
    """
    # First handle boundaries like "ABCd" -> "ABC_d"
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", str(name))
    # Then handle lower/number to upper boundary: "fooBAR" -> "foo_BAR"
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)


def sanitize_field_identifier(name: str) -> str:
    """
    Produce a valid Python/Django field identifier from an arbitrary source name.

    Rules:
    - Replace namespace/punctuation separators (":", ".", "-", spaces) with "_"
    - Convert CamelCase to snake_case
    - Remove invalid characters (keep only [A-Za-z0-9_])
    - Ensure starts with a letter or underscore; if not, prefix with "_"
    - Lowercase and collapse multiple underscores

    Note: Does not suffix Python keywords/builtins; callers decide policy.
    """
    # Normalize separators
    replaced = normalize_namespace_separators(name)
    # Convert to snake
    snake = to_snake_case(replaced)
    # Keep valid identifier chars only
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", snake)
    # Ensure starts with letter or underscore
    if not cleaned or not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = f"_{cleaned}" if cleaned else "_"
    # Lowercase and collapse underscores
    lowered = cleaned.lower()
    return re.sub(r"__+", "_", lowered)


def enum_class_name_from_field(field_name: str) -> str:
    """
    Derive a PascalCase enum class name from a field name.

    - Replace separators with underscores, then title-case and remove underscores
    - Example: "type" -> "Type", "xlink:type" -> "XlinkType"
    """
    base = normalize_namespace_separators(field_name)
    return base.replace("_", " ").title().replace(" ", "")
