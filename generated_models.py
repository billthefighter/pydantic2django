"""
Generated Django models from Pydantic models.
Generated at: 2025-04-09 20:59:51
"""


"""
Imports for generated models and context classes.
"""
# Standard library imports
from typing import TypeVar

# Django and Pydantic imports
from django.db import models

# Pydantic2Django imports
from pydantic2django.base_django_model import Pydantic2DjangoBaseClass

# Additional type imports from typing module

# Original Pydantic model imports

# Context class field type imports

# Type variable for model classes
T = TypeVar("T")

# Generated Django models
"""
Django model for PartDeltaEvent.
"""


class DjangoPartDeltaEvent(Pydantic2DjangoBaseClass):
    """
    Django model for PartDeltaEvent.
    """

    index = models.BigAutoField(blank=False, null=False)
    delta = models.JSONField(blank=False, null=False)
    event_kind = models.ImageField(blank=False, default="part_delta", null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for PartStartEvent.
"""


class DjangoPartStartEvent(Pydantic2DjangoBaseClass):
    """
    Django model for PartStartEvent.
    """

    index = models.BigAutoField(blank=False, null=False)
    part = models.JSONField(blank=False, null=False)
    event_kind = models.ImageField(blank=False, default="part_start", null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for TextPart.
"""


class DjangoTextPart(Pydantic2DjangoBaseClass):
    """
    Django model for TextPart.
    """

    content = models.ImageField(blank=False, null=False)
    part_kind = models.ImageField(blank=False, default="text", null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for TextPartDelta.
"""


class DjangoTextPartDelta(Pydantic2DjangoBaseClass):
    """
    Django model for TextPartDelta.
    """

    content_delta = models.ImageField(blank=False, null=False)
    part_delta_kind = models.ImageField(blank=False, default="text", null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for ToolCallPart.
"""


class DjangoToolCallPart(Pydantic2DjangoBaseClass):
    """
    Django model for ToolCallPart.
    """

    tool_name = models.ImageField(blank=False, null=False)
    args = models.JSONField(blank=False, null=False)
    tool_call_id = models.ImageField(blank=False, null=False)
    part_kind = models.ImageField(blank=False, default="tool-call", null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for ToolCallPartDelta.
"""


class DjangoToolCallPartDelta(Pydantic2DjangoBaseClass):
    """
    Django model for ToolCallPartDelta.
    """

    tool_name_delta = models.JSONField(blank=False, default=None, null=False)
    args_delta = models.JSONField(blank=False, default=None, null=False)
    tool_call_id = models.JSONField(blank=False, default=None, null=False)
    part_delta_kind = models.ImageField(blank=False, default="tool_call", null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for AudioUrl.
"""


class DjangoAudioUrl(Pydantic2DjangoBaseClass):
    """
    Django model for AudioUrl.
    """

    url = models.ImageField(blank=False, null=False)
    kind = models.ImageField(blank=False, default="audio-url", null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for BinaryContent.
"""


class DjangoBinaryContent(Pydantic2DjangoBaseClass):
    """
    Django model for BinaryContent.
    """

    data = models.BinaryField(blank=False, null=False)
    media_type = models.JSONField(blank=False, null=False)
    kind = models.ImageField(blank=False, default="binary", null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for DocumentUrl.
"""


class DjangoDocumentUrl(Pydantic2DjangoBaseClass):
    """
    Django model for DocumentUrl.
    """

    url = models.ImageField(blank=False, null=False)
    kind = models.ImageField(blank=False, default="document-url", null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for FinalResultEvent.
"""


class DjangoFinalResultEvent(Pydantic2DjangoBaseClass):
    """
    Django model for FinalResultEvent.
    """

    tool_name = models.JSONField(blank=False, null=False)
    tool_call_id = models.JSONField(blank=False, null=False)
    event_kind = models.ImageField(blank=False, default="final_result", null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for FunctionToolCallEvent.
"""


class DjangoFunctionToolCallEvent(Pydantic2DjangoBaseClass):
    """
    Django model for FunctionToolCallEvent.
    """

    part = models.JSONField(blank=False, null=False)
    call_id = models.ImageField(blank=False, null=False)
    event_kind = models.ImageField(blank=False, default="function_tool_call", null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for FunctionToolResultEvent.
"""


class DjangoFunctionToolResultEvent(Pydantic2DjangoBaseClass):
    """
    Django model for FunctionToolResultEvent.
    """

    result = models.JSONField(blank=False, null=False)
    tool_call_id = models.ImageField(blank=False, null=False)
    event_kind = models.ImageField(blank=False, default="function_tool_result", null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for ImageUrl.
"""


class DjangoImageUrl(Pydantic2DjangoBaseClass):
    """
    Django model for ImageUrl.
    """

    url = models.ImageField(blank=False, null=False)
    kind = models.ImageField(blank=False, default="image-url", null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for ModelRequest.
"""


class DjangoModelRequest(Pydantic2DjangoBaseClass):
    """
    Django model for ModelRequest.
    """

    parts = models.JSONField(blank=False, null=False)
    kind = models.ImageField(blank=False, default="request", null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for ModelResponse.
"""


class DjangoModelResponse(Pydantic2DjangoBaseClass):
    """
    Django model for ModelResponse.
    """

    parts = models.JSONField(blank=False, null=False)
    model_name = models.JSONField(blank=False, default=None, null=False)
    timestamp = models.DateTimeField(blank=False, null=False)
    kind = models.ImageField(blank=False, default="response", null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for RetryPromptPart.
"""


class DjangoRetryPromptPart(Pydantic2DjangoBaseClass):
    """
    Django model for RetryPromptPart.
    """

    content = models.JSONField(blank=False, null=False)
    tool_name = models.JSONField(blank=False, default=None, null=False)
    tool_call_id = models.ImageField(blank=False, null=False)
    timestamp = models.DateTimeField(blank=False, null=False)
    part_kind = models.ImageField(blank=False, default="retry-prompt", null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for SystemPromptPart.
"""


class DjangoSystemPromptPart(Pydantic2DjangoBaseClass):
    """
    Django model for SystemPromptPart.
    """

    content = models.ImageField(blank=False, null=False)
    timestamp = models.DateTimeField(blank=False, null=False)
    dynamic_ref = models.JSONField(blank=False, default=None, null=False)
    part_kind = models.ImageField(blank=False, default="system-prompt", null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for ToolReturnPart.
"""


class DjangoToolReturnPart(Pydantic2DjangoBaseClass):
    """
    Django model for ToolReturnPart.
    """

    tool_name = models.ImageField(blank=False, null=False)
    content = models.JSONField(blank=False, null=False)
    tool_call_id = models.ImageField(blank=False, null=False)
    timestamp = models.DateTimeField(blank=False, null=False)
    part_kind = models.ImageField(blank=False, default="tool-return", null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for UserPromptPart.
"""


class DjangoUserPromptPart(Pydantic2DjangoBaseClass):
    """
    Django model for UserPromptPart.
    """

    content = models.JSONField(blank=False, null=False)
    timestamp = models.DateTimeField(blank=False, null=False)
    part_kind = models.ImageField(blank=False, default="user-prompt", null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


# List of all generated models
__all__ = [
    "DjangoPartDeltaEvent",
    "DjangoPartStartEvent",
    "DjangoTextPart",
    "DjangoTextPartDelta",
    "DjangoToolCallPart",
    "DjangoToolCallPartDelta",
    "DjangoAudioUrl",
    "DjangoBinaryContent",
    "DjangoDocumentUrl",
    "DjangoFinalResultEvent",
    "DjangoFunctionToolCallEvent",
    "DjangoFunctionToolResultEvent",
    "DjangoImageUrl",
    "DjangoModelRequest",
    "DjangoModelResponse",
    "DjangoRetryPromptPart",
    "DjangoSystemPromptPart",
    "DjangoToolReturnPart",
    "DjangoUserPromptPart",
]
