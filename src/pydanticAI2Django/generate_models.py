import logging
from enum import Enum
from typing import TypeVar

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("model_conversion")

# Configure more detailed logging for the import handler
import_handler_logger = logging.getLogger("pydantic2django.import_handler")
import_handler_logger.setLevel(logging.WARNING)

# Type variable for model classes
T = TypeVar("T")
PromptType = TypeVar("PromptType", bound=Enum)

# Configure Django settings before importing any Django-related modules
import django  # noqa: E402
from django.conf import settings  # noqa: E402

# Configure Django settings if not already configured
if not settings.configured:
    settings.configure(
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
        ],
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        },
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
    )
    django.setup()


from pydantic2django.dataclass.discovery import DataclassDiscovery
from pydantic2django.dataclass.generator import DataclassDjangoModelGenerator

logger = logging.getLogger(__name__)


def filter_messages_module(cls: type) -> bool:
    """Filter function to only include dataclasses from the pydantic_ai.messages module."""
    return getattr(cls, "__module__", None) == "pydantic_ai.messages"


def generate_models():
    """
    Generate Django models from PydanticAI
    """
    discovery = DataclassDiscovery()
    # discovery.discover_models(["pydantic_ai"], "pydantic_ai") # Removed redundant call

    generator = DataclassDjangoModelGenerator(
        output_path="generated_models.py",
        app_label="pai2django",
        filter_function=filter_messages_module,  # Use the filter function here
        packages=["pydantic_ai"],  # Still discover within the package
        discovery_instance=discovery,
        verbose=True,  # Add verbose logging
    )
    generator.generate()


if __name__ == "__main__":
    generate_models()
