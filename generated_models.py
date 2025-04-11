"""
Generated Django models from Pydantic models.
Generated at: 2025-04-11 11:09:44
"""


"""
Imports for generated models and context classes.
"""
# Standard library imports
from typing import TypeVar

# Django and Pydantic imports
from django.db import models

# Pydantic2Django imports

# Additional type imports from typing module

# Original Pydantic model imports

# Context class field type imports

# Type variable for model classes
T = TypeVar("T")

# Generated Django models
"""
Django model for PartDeltaEvent.
"""


class DjangoPartDeltaEvent(Dataclass2DjangoBaseClass):
    """
    Django model for PartDeltaEvent.
    """

    index = models.IntegerField(blank=False, null=False)
    delta = models.JSONField(blank=False, null=False)
    event_kind = models.CharField(
        blank=False, choices=[("part_delta", "part_delta")], default="part_delta", max_length=10, null=False
    )

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for PartStartEvent.
"""


class DjangoPartStartEvent(Dataclass2DjangoBaseClass):
    """
    Django model for PartStartEvent.
    """

    index = models.IntegerField(blank=False, null=False)
    part = models.JSONField(blank=False, null=False)
    event_kind = models.CharField(
        blank=False, choices=[("part_start", "part_start")], default="part_start", max_length=10, null=False
    )

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for TextPart.
"""


class DjangoTextPart(Dataclass2DjangoBaseClass):
    """
    Django model for TextPart.
    """

    content = models.TextField(blank=False, null=False)
    part_kind = models.CharField(blank=False, choices=[("text", "text")], default="text", max_length=4, null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for TextPartDelta.
"""


class DjangoTextPartDelta(Dataclass2DjangoBaseClass):
    """
    Django model for TextPartDelta.
    """

    content_delta = models.TextField(blank=False, null=False)
    part_delta_kind = models.CharField(
        blank=False, choices=[("text", "text")], default="text", max_length=4, null=False
    )

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for ToolCallPart.
"""


class DjangoToolCallPart(Dataclass2DjangoBaseClass):
    """
    Django model for ToolCallPart.
    """

    tool_name = models.TextField(blank=False, null=False)
    args = models.JSONField(blank=False, null=False)
    tool_call_id = models.TextField(blank=False, null=False)
    part_kind = models.CharField(
        blank=False, choices=[("tool-call", "tool-call")], default="tool-call", max_length=9, null=False
    )

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for ToolCallPartDelta.
"""


class DjangoToolCallPartDelta(Dataclass2DjangoBaseClass):
    """
    Django model for ToolCallPartDelta.
    """

    tool_name_delta = models.TextField(blank=False, default=None, null=False)
    args_delta = models.JSONField(blank=False, default=None, null=False)
    tool_call_id = models.TextField(blank=False, default=None, null=False)
    part_delta_kind = models.CharField(
        blank=False, choices=[("tool_call", "tool_call")], default="tool_call", max_length=9, null=False
    )

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for AudioUrl.
"""


class DjangoAudioUrl(Dataclass2DjangoBaseClass):
    """
    Django model for AudioUrl.
    """

    url = models.TextField(blank=False, null=False)
    kind = models.CharField(
        blank=False, choices=[("audio-url", "audio-url")], default="audio-url", max_length=9, null=False
    )

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for BinaryContent.
"""


class DjangoBinaryContent(Dataclass2DjangoBaseClass):
    """
    Django model for BinaryContent.
    """

    data = models.BinaryField(blank=False, null=False)
    media_type = models.TextField(blank=False, null=False)
    kind = models.CharField(blank=False, choices=[("binary", "binary")], default="binary", max_length=6, null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for DocumentUrl.
"""


class DjangoDocumentUrl(Dataclass2DjangoBaseClass):
    """
    Django model for DocumentUrl.
    """

    url = models.TextField(blank=False, null=False)
    kind = models.CharField(
        blank=False, choices=[("document-url", "document-url")], default="document-url", max_length=12, null=False
    )

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for FinalResultEvent.
"""


class DjangoFinalResultEvent(Dataclass2DjangoBaseClass):
    """
    Django model for FinalResultEvent.
    """

    tool_name = models.TextField(blank=False, null=False)
    tool_call_id = models.TextField(blank=False, null=False)
    event_kind = models.CharField(
        blank=False, choices=[("final_result", "final_result")], default="final_result", max_length=12, null=False
    )

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for FunctionToolCallEvent.
"""


class DjangoFunctionToolCallEvent(Dataclass2DjangoBaseClass):
    """
    Django model for FunctionToolCallEvent.
    """

    part = models.JSONField(blank=False, null=False)
    call_id = models.TextField(blank=False, null=False)
    event_kind = models.CharField(
        blank=False,
        choices=[("function_tool_call", "function_tool_call")],
        default="function_tool_call",
        max_length=18,
        null=False,
    )

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for FunctionToolResultEvent.
"""


class DjangoFunctionToolResultEvent(Dataclass2DjangoBaseClass):
    """
    Django model for FunctionToolResultEvent.
    """

    result = models.JSONField(blank=False, null=False)
    tool_call_id = models.TextField(blank=False, null=False)
    event_kind = models.CharField(
        blank=False,
        choices=[("function_tool_result", "function_tool_result")],
        default="function_tool_result",
        max_length=20,
        null=False,
    )

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for ImageUrl.
"""


class DjangoImageUrl(Dataclass2DjangoBaseClass):
    """
    Django model for ImageUrl.
    """

    url = models.TextField(blank=False, null=False)
    kind = models.CharField(
        blank=False, choices=[("image-url", "image-url")], default="image-url", max_length=9, null=False
    )

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for ModelRequest.
"""


class DjangoModelRequest(Dataclass2DjangoBaseClass):
    """
    Django model for ModelRequest.
    """

    parts = models.JSONField(blank=False, null=False)
    kind = models.CharField(blank=False, choices=[("request", "request")], default="request", max_length=7, null=False)

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for ModelResponse.
"""


class DjangoModelResponse(Dataclass2DjangoBaseClass):
    """
    Django model for ModelResponse.
    """

    parts = models.JSONField(blank=False, null=False)
    model_name = models.TextField(blank=False, default=None, null=False)
    timestamp = models.DateTimeField(blank=False, null=False)
    kind = models.CharField(
        blank=False, choices=[("response", "response")], default="response", max_length=8, null=False
    )

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for RetryPromptPart.
"""


class DjangoRetryPromptPart(Dataclass2DjangoBaseClass):
    """
    Django model for RetryPromptPart.
    """

    content = models.JSONField(blank=False, null=False)
    tool_name = models.TextField(blank=False, default=None, null=False)
    tool_call_id = models.TextField(blank=False, null=False)
    timestamp = models.DateTimeField(blank=False, null=False)
    part_kind = models.CharField(
        blank=False, choices=[("retry-prompt", "retry-prompt")], default="retry-prompt", max_length=12, null=False
    )

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for SystemPromptPart.
"""


class DjangoSystemPromptPart(Dataclass2DjangoBaseClass):
    """
    Django model for SystemPromptPart.
    """

    content = models.TextField(blank=False, null=False)
    timestamp = models.DateTimeField(blank=False, null=False)
    dynamic_ref = models.TextField(blank=False, default=None, null=False)
    part_kind = models.CharField(
        blank=False, choices=[("system-prompt", "system-prompt")], default="system-prompt", max_length=13, null=False
    )

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for ToolReturnPart.
"""


class DjangoToolReturnPart(Dataclass2DjangoBaseClass):
    """
    Django model for ToolReturnPart.
    """

    tool_name = models.TextField(blank=False, null=False)
    content = models.JSONField(blank=False, null=False)
    tool_call_id = models.TextField(blank=False, null=False)
    timestamp = models.DateTimeField(blank=False, null=False)
    part_kind = models.CharField(
        blank=False, choices=[("tool-return", "tool-return")], default="tool-return", max_length=11, null=False
    )

    class Meta:
        app_label = "pai2django"
        abstract = False


"""
Django model for UserPromptPart.
"""


class DjangoUserPromptPart(Dataclass2DjangoBaseClass):
    """
    Django model for UserPromptPart.
    """

    content = models.JSONField(blank=False, null=False)
    timestamp = models.DateTimeField(blank=False, null=False)
    part_kind = models.CharField(
        blank=False, choices=[("user-prompt", "user-prompt")], default="user-prompt", max_length=11, null=False
    )

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
