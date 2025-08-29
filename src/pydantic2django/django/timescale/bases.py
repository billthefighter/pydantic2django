"""Common abstract base classes for TimescaleDB-enabled models.

These combine existing base classes with the TimescaleModel mixin when available.
"""

try:
    # Reuse TimescaleModel import/fallback defined in pydantic2django.django.models
    from pydantic2django.django.models import TimescaleModel  # type: ignore
except Exception:  # pragma: no cover - defensive fallback

    class TimescaleModel:  # type: ignore[no-redef]
        pass


# Import the existing bases for each source type
from pydantic2django.django.models import (
    Dataclass2DjangoBaseClass,
    Pydantic2DjangoBaseClass,
    Xml2DjangoBaseClass,
)


class TimescaleBaseMixin(TimescaleModel):
    class Meta:
        abstract = True


class XmlTimescaleBase(Xml2DjangoBaseClass, TimescaleBaseMixin):
    class Meta:
        abstract = True


class PydanticTimescaleBase(Pydantic2DjangoBaseClass, TimescaleBaseMixin):
    class Meta:
        abstract = True


class DataclassTimescaleBase(Dataclass2DjangoBaseClass, TimescaleBaseMixin):
    class Meta:
        abstract = True
