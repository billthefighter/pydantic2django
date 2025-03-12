import json
import os
import sys

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure Django settings
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "example_app",
        ],
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": "db.sqlite3",
            }
        },
    )
    django.setup()

# Import the generated models
# Note: You need to run generate_models_example.py first to create these models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "generated")))

try:
    from models import DjangoProduct, DjangoUser
except ImportError:
    print("Error: Generated models not found.")
    print("Please run generate_models_example.py first to create the models.")
    sys.exit(1)

# Import the original Pydantic models for comparison
from generate_models_example import Product, User


def main():
    """Demonstrate how to use the generated Django models."""
    print("Creating Django database tables...")
    from django.db import connection

    with connection.cursor() as cursor:
        # Create the tables for our models
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS example_app_djangouser (
            id CHAR(32) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            object_type VARCHAR(100) NOT NULL,
            data JSON NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL
        )
        """
        )

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS example_app_djangoproduct (
            id CHAR(32) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            object_type VARCHAR(100) NOT NULL,
            data JSON NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL
        )
        """
        )

    # Create a Pydantic User object
    pydantic_user = User(
        id=1,
        username="johndoe",
        email="john@example.com",
        full_name="John Doe",
        is_active=True,
        age=30,
        tags=["customer", "premium"],
        metadata={"last_login": "2023-01-01", "signup_source": "web"},
    )

    # Convert to Django model and save
    django_user = DjangoUser.from_pydantic(pydantic_user, name="John Doe")
    django_user.save()
    print(f"Saved Django user with ID: {django_user.id}")

    # Create a Pydantic Product object
    pydantic_product = Product(
        id=101,
        name="Awesome Product",
        description="This is an awesome product",
        price=99.99,
        in_stock=True,
        categories=["electronics", "gadgets"],
    )

    # Convert to Django model and save
    django_product = DjangoProduct.from_pydantic(pydantic_product)
    django_product.save()
    print(f"Saved Django product with ID: {django_product.id}")

    # Retrieve the objects from the database
    retrieved_user = DjangoUser.objects.get(name="John Doe")
    print("\nRetrieved user from database:")
    print(f"  ID: {retrieved_user.id}")
    print(f"  Name: {retrieved_user.name}")
    print(f"  Object Type: {retrieved_user.object_type}")
    print(f"  Created At: {retrieved_user.created_at}")
    print(f"  Data: {json.dumps(retrieved_user.data, indent=2)}")

    # Convert back to Pydantic model
    pydantic_user_from_db = retrieved_user.to_pydantic()
    print("\nConverted back to Pydantic model:")
    print(f"  Username: {pydantic_user_from_db.username}")
    print(f"  Email: {pydantic_user_from_db.email}")
    print(f"  Full Name: {pydantic_user_from_db.full_name}")
    print(f"  Tags: {pydantic_user_from_db.tags}")

    # Demonstrate updating a model
    pydantic_user_from_db.tags.append("vip")
    pydantic_user_from_db.metadata["last_login"] = "2023-02-01"

    # Update the Django model with the modified Pydantic model
    retrieved_user.update_from_pydantic(pydantic_user_from_db)

    # Verify the update
    updated_user = DjangoUser.objects.get(id=retrieved_user.id)
    print("\nUpdated user data:")
    print(f"  Tags: {updated_user.data['tags']}")
    print(f"  Last Login: {updated_user.data['metadata']['last_login']}")

    print("\nDemonstration complete!")


if __name__ == "__main__":
    main()
