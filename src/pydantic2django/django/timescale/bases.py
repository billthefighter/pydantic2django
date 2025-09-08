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
# Django imports for constraint management

from pydantic2django.django.models import (
    Dataclass2DjangoBaseClass,
    Pydantic2DjangoBaseClass,
    Xml2DjangoBaseClass,
)


class TimescaleBaseMixin(TimescaleModel):
    class Meta:
        abstract = True

    def __init_subclass__(cls, **kwargs):  # type: ignore[override]
        """Avoid emitting redundant constraints; rely on the PK semantics for ``id``.

        Because ``django-timescaledb`` expects to drop the default primary-key constraint
        (``<table>_pkey``) during hypertable creation, adding an additional named unique
        constraint on ``id`` in migrations can cause conflicts. We therefore do not add
        any extra unique constraint on ``id`` here.
        """
        super().__init_subclass__(**kwargs)
        # Intentionally no-op: we no longer attach a UniqueConstraint('id')


def _ensure_unique_id_on_prepared(sender, **kwargs) -> None:
    """No-op: we no longer add a UniqueConstraint on id for Timescale models."""
    return


# Do not attach signal; we no longer add constraints dynamically


class XmlTimescaleBase(Xml2DjangoBaseClass, TimescaleBaseMixin):
    class Meta:  # type: ignore[misc]
        abstract = True


class PydanticTimescaleBase(Pydantic2DjangoBaseClass, TimescaleBaseMixin):
    class Meta:  # type: ignore[misc]
        abstract = True


class DataclassTimescaleBase(Dataclass2DjangoBaseClass, TimescaleBaseMixin):
    class Meta:  # type: ignore[misc]
        abstract = True
