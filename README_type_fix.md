# Type Handling Fix for Context Class Generation

## Problem
When generating context classes for Django models, the field types were being output with fully qualified names (FQNs), which caused syntax errors in the generated code. The issue occurred in the `__post_init__` method of generated context classes:

```python
def __post_init__(self):
    """Initialize context fields after instance creation."""
    self.add_field(
        field_name="rate_limit",
        field_type=llmaestro.config.base.RateLimitConfig,  # FQN causing issues
        is_optional=True,
        is_list=False,
        additional_metadata={}
    )
```

## Solution
The fix modifies the `generate_context_class` method in `static_django_model_generator.py` to properly handle field types:

1. Extract just the class name from field types instead of using FQNs
2. Improve type handling, especially for generic types with type arguments
3. Use proper imports at the top of the file instead of embedding FQNs in the code

The key change is in how field types are processed before being passed to the template:

```python
# Process the field type to get just the class name for the template
if hasattr(field_type, "__name__"):
    # If it's a class, use its name
    type_name = field_type.__name__
else:
    # Otherwise get formatted type string from the TypeHandler
    type_name = model_context.get_formatted_field_type(field_context.field_name)

# Handle special typing constructs (Optional, List, etc.)
if hasattr(field_type, "__origin__"):
    origin = get_origin(field_type)
    if origin is not None:
        origin_name = origin.__name__
        args = get_args(field_type)
        if args:
            arg_names = []
            for arg in args:
                if hasattr(arg, "__name__"):
                    arg_names.append(arg.__name__)
                else:
                    arg_names.append(str(arg).split(".")[-1])
            type_name = f"{origin_name}[{', '.join(arg_names)}]"
```

## Benefits
- Cleaner, more readable generated code
- No more syntax errors from FQNs
- Properly manages imports at the top of the file
- Better handling of generic types and type parameters
