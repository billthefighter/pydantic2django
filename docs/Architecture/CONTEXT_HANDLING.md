# Context Handling in Pydantic2Django

## Overview

Pydantic2Django handles non-serializable fields through a context system. When a Pydantic model contains fields that cannot be directly serialized to the database (like custom classes, functions, or complex objects), these fields are marked as "context fields" and require special handling during conversion back to Pydantic objects.

## How It Works

### 1. Field Detection

During model generation, fields are analyzed for serializability. A field is considered non-serializable if it:
- Is a custom class without JSON serialization methods
- Contains callable objects
- Contains complex objects that can't be stored in the database
- Is explicitly marked as non-serializable

### 2. Storage

Non-serializable fields are stored in the database as:
```python
models.TextField(is_relationship=True)
```

This special flag indicates that the field requires context when converting back to a Pydantic object.

### 3. Type Safety

For each Django model with context fields, a corresponding `TypedDict` is generated:

```python
class MyModelContext(TypedDict):
    """Required context fields for converting MyModel back to MyPydanticModel."""
    field1: Any
    field2: Any
```

### 4. Usage Example

```python
from my_app.models import MyModel, MyModelContext
from my_app.pydantic_models import MyPydanticModel

# Creating from Pydantic
pydantic_obj = MyPydanticModel(
    field1=ComplexObject(),
    field2=lambda x: x + 1
)
django_obj = MyModel.from_pydantic(pydantic_obj)

# Converting back to Pydantic
context: MyModelContext = {
    "field1": ComplexObject(),  # Must provide the non-serializable objects
    "field2": lambda x: x + 1
}
pydantic_obj = django_obj.to_pydantic(context=context)
```

## Error Handling

If you attempt to convert a Django model back to Pydantic without providing the required context, you'll get a clear error message:

```python
ValueError: This model has non-serializable fields that require context: field1, field2.
Please provide the context dictionary when calling to_pydantic().
```

If you provide context but miss some required fields:

```python
ValueError: Missing required context fields: field2
```

## Best Practices

1. **Type Hints**: Always use the generated Context type for better IDE support:
   ```python
   context: MyModelContext = {...}
   ```

2. **Documentation**: The generated Django models include docstrings listing all required context fields.

3. **Validation**: Consider implementing validation for context values before passing them to `to_pydantic()`.

4. **State Management**: Keep context values organized and easily accessible in your application.

## Technical Details

### Field Type Resolution

The `is_serializable_type()` function in `field_utils.py` determines if a type is serializable by checking:
- Basic Python types (str, int, float, bool, etc.)
- Collections (list, dict, set) with serializable elements
- Pydantic models
- Enums
- Classes with JSON serialization methods

### Context Field Detection

Fields are marked as context fields when:
1. The type is not serializable
2. The field is explicitly marked with `is_relationship=True`
3. The field requires custom serialization logic

### Generated Code

For each model with context fields, the following is generated:
1. A TypedDict for context validation
2. A modified `to_pydantic()` method requiring context
3. Documentation explaining the context requirements
4. Proper imports and exports in the models file

## Example Implementation

```python
# Generated Django Model
from typing import TypedDict

class UserContext(TypedDict):
    avatar_processor: Any
    permissions_handler: Any

class User(Pydantic2DjangoBaseClass[UserPydantic]):
    name = models.CharField(max_length=255)
    avatar_processor = models.TextField(is_relationship=True)
    permissions_handler = models.TextField(is_relationship=True)

    def to_pydantic(self, context: UserContext) -> UserPydantic:
        return super().to_pydantic(context=context)

# Usage
user = User.objects.get(id=1)
context: UserContext = {
    "avatar_processor": AvatarProcessor(),
    "permissions_handler": PermissionsHandler()
}
pydantic_user = user.to_pydantic(context=context)
```
