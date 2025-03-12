import os
import sys
from typing import Any, Optional

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure Django settings before importing Django models
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
        ],
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        },
    )
    django.setup()

from pydantic import BaseModel, Field

from src.pydantic2django.mock_discovery import register_model
from src.pydantic2django.static_django_model_generator import StaticDjangoModelGenerator


# Define some example Pydantic models
class Address(BaseModel):
    """A simple address model."""

    street: str
    city: str
    state: str
    zip_code: str
    country: str = "USA"


class User(BaseModel):
    """A user model with various field types."""

    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool = True
    age: Optional[int] = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    addresses: list[Address] = Field(default_factory=list)


class Product(BaseModel):
    """A product model."""

    id: int
    name: str
    description: Optional[str] = None
    price: float
    in_stock: bool = True
    categories: list[str] = Field(default_factory=list)


# Define a filter function to only include certain models
def filter_models(model: type[BaseModel]) -> bool:
    """Filter function to only include User and Product models."""
    return model.__name__ in ["User", "Product"]


def main():
    """Main function to demonstrate the StaticDjangoModelGenerator."""
    # Create the output directory if it doesn't exist
    os.makedirs("generated", exist_ok=True)

    # Register our models with the mock discovery module
    register_model("User", User)
    register_model("Product", Product)

    # Create the generator
    generator = StaticDjangoModelGenerator(
        output_path="generated/models.py",
        packages=["examples.generate_models_example"],  # Use this module
        app_label="example_app",
        filter_function=filter_models,
        verbose=True,
    )

    # Generate the models file
    output_path = generator.generate()

    if output_path:
        print(f"\nGenerated models file at: {output_path}")
        print("\nYou can now use these models in your Django application.")
        print("Example usage:")
        print("```python")
        print("from generated.models import DjangoUser, DjangoProduct")
        print("")
        print("# Create a new user")
        print("user = DjangoUser(name='johndoe', object_type='User')")
        print("user.data = {")
        print("    'username': 'johndoe',")
        print("    'email': 'john@example.com',")
        print("    'is_active': True")
        print("}")
        print("user.save()")
        print("")
        print("# Convert back to Pydantic model")
        print("pydantic_user = user.to_pydantic()")
        print("print(pydantic_user)")
        print("```")
    else:
        print("No models were generated. Check your filter function.")


if __name__ == "__main__":
    main()
