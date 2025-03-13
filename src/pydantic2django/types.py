"""
Type definitions for pydantic2django.
"""
from typing import TypeVar, Union

from django.db import models
from pydantic import BaseModel

# Type variable for BaseModel subclasses
T = TypeVar("T", bound=BaseModel)

# Type alias for Django model fields
DjangoField = Union[models.Field, type[models.Field]]
