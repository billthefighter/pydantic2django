"""
Generated Django models from Pydantic models.
Generated at: 2025-04-09 10:48:57
"""


"""
Imports for generated models and context classes.
"""
# Standard library imports
from typing import TypeVar

# Django and Pydantic imports
# Pydantic2Django imports
from pydantic2django.base_django_model import Pydantic2DjangoBaseClass

# Additional type imports from typing module

# Original Pydantic model imports

# Context class field type imports

# Type variable for model classes
T = TypeVar("T")

# Generated Django models
"""
Django model for ConsoleOptions.
"""


class DjangoConsoleOptions(Pydantic2DjangoBaseClass):
    """
    Django model for ConsoleOptions.
    """

    size = pai2django.DjangoConsoleOptions.size
    legacy_windows = pai2django.DjangoConsoleOptions.legacy_windows
    min_width = pai2django.DjangoConsoleOptions.min_width
    max_width = pai2django.DjangoConsoleOptions.max_width
    is_terminal = pai2django.DjangoConsoleOptions.is_terminal
    encoding = pai2django.DjangoConsoleOptions.encoding
    max_height = pai2django.DjangoConsoleOptions.max_height
    no_wrap = pai2django.DjangoConsoleOptions.no_wrap
    highlight = pai2django.DjangoConsoleOptions.highlight
    markup = pai2django.DjangoConsoleOptions.markup
    height = pai2django.DjangoConsoleOptions.height

    class Meta(Pydantic2DjangoBaseClass.Meta):
        app_label = "django_llm"
        abstract = False


"""
Django model for SDKError.
"""


class DjangoSDKError(Pydantic2DjangoBaseClass):
    """
    Django model for SDKError.
    """

    message = pai2django.DjangoSDKError.message
    status_code = pai2django.DjangoSDKError.status_code
    body = pai2django.DjangoSDKError.body

    class Meta(Pydantic2DjangoBaseClass.Meta):
        app_label = "django_llm"
        abstract = False


# List of all generated models
__all__ = [
    "DjangoConsoleOptions",
    "DjangoSDKError",
]
