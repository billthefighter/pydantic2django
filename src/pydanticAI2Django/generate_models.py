import logging
from enum import Enum
from typing import TypeVar

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("model_conversion")

# Configure more detailed logging for the import handler
import_handler_logger = logging.getLogger("pydantic2django.import_handler")
import_handler_logger.setLevel(logging.INFO)

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


logger = logging.getLogger(__name__)


def generate_models():
    """
    Generate Django models from PydanticAI
    """
    discovery = ModelDiscovery()
    discovery.discover_models(["pydantic_ai"], "pydantic_ai")

    generator = StaticDjangoModelGenerator(
        output_path="generated_models.py",
        app_label="pydantic_ai",
        filter_function=None,  # Remove filter to ensure all registered models are included
        discovery_module=discovery,
        verbose=True,  # Add verbose logging
    )
    generator.generate()


if __name__ == "__main__":
    generate_models()
