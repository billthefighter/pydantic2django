# Pydantic2Django Bidirectional Mapper Details

This document provides details on how the `BidirectionalTypeMapper` handles specific field type conversions between Django models and Pydantic models.

## Django `choices` Field Mapping

When mapping a Django field (like `CharField` or `IntegerField`) that has the `choices` attribute set, the `BidirectionalTypeMapper` employs a hybrid approach for the resulting Pydantic field:

1.  **Pydantic Type:** The Python type hint for the Pydantic field is set to `typing.Literal[...]`, where the literal values are the *raw database values* defined in the Django `choices` (e.g., `Literal['S', 'M', 'L']` or `Literal[1, 2, 3]`). If the Django field has `null=True`, the type becomes `Optional[Literal[...]]`.
    *   **Benefit:** This provides strong typing and allows Pydantic to perform validation, ensuring that only the allowed raw values are assigned to the field.

2.  **Metadata:** The original Django `choices` tuple, containing the `(raw_value, human_readable_label)` pairs (e.g., `[('S', 'Small'), ('M', 'Medium'), ('L', 'Large')]`), is preserved within the Pydantic `FieldInfo` associated with the field. Specifically, it's stored under the `json_schema_extra` key:
    ```python
    FieldInfo(..., json_schema_extra={'choices': [('S', 'Small'), ('M', 'Medium')]})
    ```
    *   **Benefit:** This keeps the human-readable labels associated with the field, making them available for other purposes like generating API documentation (e.g., OpenAPI schemas), building UI components (like dropdowns), or custom logic, without sacrificing the validation provided by the `Literal` type.

**Trade-off:** This approach prioritizes data validation using Pydantic's `Literal` type based on the raw stored values. The human-readable labels are available as metadata but are not part of the core Pydantic type validation itself. The Django `get_FOO_display()` method is not directly used during the conversion process, as the focus is on mapping the underlying data values and types.

## Other Field Mappings

*(This section can be expanded later with details about other interesting or complex mappings, such as relationships, JSON fields, etc.)*
