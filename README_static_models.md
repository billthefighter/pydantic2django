# Static Django Model Generator

The `StaticDjangoModelGenerator` is a tool that generates static Django model definitions from Pydantic models. This is useful when you want to create a Django app with models that match your Pydantic models, but you don't want to rely on dynamic model generation at runtime.

## Features

- Discovers Pydantic models from specified packages
- Generates Django model definitions with proper field types and relationships
- Preserves field attributes like verbose_name, help_text, etc.
- Handles relationships (ForeignKey, ManyToManyField)
- Generates a complete models.py file ready to use in your Django app

## Installation

The `StaticDjangoModelGenerator` is part of the `pydantic2django` package. You can install it using pip:

```bash
pip install pydantic2django
```

## Usage

### Command-line Interface

You can use the `StaticDjangoModelGenerator` from the command line:

```bash
python -m pydantic2django.static_django_model_generator --packages your_package.models --output your_app/models.py --app-label your_app --verbose
```

Arguments:
- `--packages`, `-p`: Packages to scan for Pydantic models (required)
- `--output`, `-o`: Output file path (default: generated_models.py)
- `--app-label`, `-a`: Django app label (default: django_app)
- `--verbose`, `-v`: Print verbose output

### Python API

You can also use the `StaticDjangoModelGenerator` in your Python code:

```python
from pydantic2django.static_django_model_generator import StaticDjangoModelGenerator

# Create the generator
generator = StaticDjangoModelGenerator(
    output_path="your_app/models.py",
    packages=["your_package.models"],
    app_label="your_app",
    verbose=True,
)

# Generate the models
generator.generate()
```

### Filtering Models

You can filter which models to include using a filter function:

```python
from pydantic2django.discovery import include_models, exclude_models, has_field

# Only include specific models
generator = StaticDjangoModelGenerator(
    output_path="your_app/models.py",
    packages=["your_package.models"],
    app_label="your_app",
    filter_function=include_models(["User", "Product"]),
    verbose=True,
)

# Exclude specific models
generator = StaticDjangoModelGenerator(
    output_path="your_app/models.py",
    packages=["your_package.models"],
    app_label="your_app",
    filter_function=exclude_models(["InternalConfig", "PrivateData"]),
    verbose=True,
)

# Only include models with a specific field
generator = StaticDjangoModelGenerator(
    output_path="your_app/models.py",
    packages=["your_package.models"],
    app_label="your_app",
    filter_function=has_field("email"),
    verbose=True,
)
```

## Example

Here's a complete example of how to use the `StaticDjangoModelGenerator`:

```python
from pydantic2django.static_django_model_generator import StaticDjangoModelGenerator

def main():
    # Define the output path
    output_path = "your_app/models.py"

    # Define the packages to scan for Pydantic models
    packages = ["your_package.models"]

    # Define the Django app label
    app_label = "your_app"

    # Create the generator
    generator = StaticDjangoModelGenerator(
        output_path=output_path,
        packages=packages,
        app_label=app_label,
        verbose=True,
    )

    # Generate the models
    generator.generate()

    print(f"Generated static Django models at {output_path}")

if __name__ == "__main__":
    main()
```

## Generated Models

The generated models will inherit from `Pydantic2DjangoBaseClass`, which provides methods for converting between Django models and Pydantic models. Here's an example of a generated model:

```python
class User(Pydantic2DjangoBaseClass):
    email = models.EmailField(verbose_name='Email Address', max_length=255, unique=True)
    first_name = models.CharField(verbose_name='First Name', max_length=100)
    last_name = models.CharField(verbose_name='Last Name', max_length=100)
    is_active = models.BooleanField(verbose_name='Active', default=True)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "your_app_user"
        app_label = "your_app"
        verbose_name = "User"
        verbose_name_plural = "Users"
        abstract = False

    class PydanticConfig:
        module_path = "your_package.models"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("object_type", "User")
        super().__init__(*args, **kwargs)
```

## The Base Class: Pydantic2DjangoBaseClass

The `Pydantic2DjangoBaseClass` is a powerful base class that provides seamless integration between Django models and Pydantic models. It includes:

1. **Type-safe conversion**: Convert between Django and Pydantic models with proper type hints
2. **Method forwarding**: Access Pydantic model methods directly from Django model instances
3. **Common fields**: Includes common fields like `id`, `name`, `created_at`, and `updated_at`
4. **Serialization/deserialization**: Convert to and from Pydantic models with validation

### Using the Base Class Directly

You can also use the `Pydantic2DjangoBaseClass` directly in your Django models:

```python
from pydantic2django.base_django_model import Pydantic2DjangoBaseClass
from django.db import models
from your_package.models import UserPydantic  # Your Pydantic model

class User(Pydantic2DjangoBaseClass[UserPydantic]):
    email = models.EmailField(max_length=255, unique=True)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)

    class Meta(Pydantic2DjangoBaseClass.Meta):
        db_table = "your_app_user"
        app_label = "your_app"

    class PydanticConfig:
        module_path = "your_package.models"
```

With this setup, you get full type checking and IDE support for your Django models based on their Pydantic counterparts.

## Alternative: JSON Storage

For cases where you want to store the entire Pydantic object as JSON rather than mapping individual fields, you can use the `Pydantic2DjangoStorePydanticObject` base class:

```python
from pydantic2django.base_django_model import Pydantic2DjangoStorePydanticObject
from django.db import models
from your_package.models import UserPydantic  # Your Pydantic model

class UserStore(Pydantic2DjangoStorePydanticObject):
    # You can add additional fields here if needed

    class Meta:
        db_table = "your_app_user_store"
        app_label = "your_app"
```

This approach is useful when:
- Your Pydantic models change frequently
- You have complex nested structures
- You want to preserve all data without mapping each field

## Benefits of Static Models

Using static models instead of dynamic models has several benefits:

1. **Performance**: Static models are faster because they don't need to be generated at runtime.
2. **IDE Support**: Static models provide better IDE support with code completion and type checking.
3. **Migrations**: Django migrations work better with static models.
4. **Debugging**: It's easier to debug static models because you can see the actual code.
5. **Customization**: You can customize the generated models by editing the generated file.

## Limitations

- The generator can only handle field types that are supported by both Pydantic and Django.
- Custom field types may not be properly converted.
- Some advanced Pydantic features like validators and computed fields are not converted to Django.
