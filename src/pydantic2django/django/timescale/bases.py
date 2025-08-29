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
from django.db import models as _dj_models
from django.db.models.signals import class_prepared as _class_prepared

from pydantic2django.django.models import (
    Dataclass2DjangoBaseClass,
    Pydantic2DjangoBaseClass,
    Xml2DjangoBaseClass,
)


class TimescaleBaseMixin(TimescaleModel):
    class Meta:
        abstract = True

    def __init_subclass__(cls, **kwargs):  # type: ignore[override]
        """Ensure Timescale hypertables keep a unique constraint on the surrogate ``id``.

        Context:
        - ``django-timescaledb`` drops the primary key during hypertable creation.
        - Downstream ForeignKeys require a unique or primary constraint on the referenced column.

        Strategy:
        - For every concrete subclass, append a ``UniqueConstraint`` on ``id`` named
          ``<db_table>_id_unique`` if one is not already present.
        - This constraint is created by Django as a separate migration operation (AddConstraint)
          after the hypertable conversion, restoring uniqueness guarantees on ``id``.
        """
        super().__init_subclass__(**kwargs)
        try:
            # Only apply to concrete models that inherit from this mixin
            meta = getattr(cls, "_meta", None)
            if not meta or getattr(meta, "abstract", True):
                return

            # Avoid duplicate constraints
            existing_id_unique = False
            for constraint in getattr(meta, "constraints", []) or []:
                if isinstance(constraint, _dj_models.UniqueConstraint):
                    # Constraint can specify fields by name list
                    fields = getattr(constraint, "fields", None) or []
                    if list(fields) == ["id"]:
                        existing_id_unique = True
                        break

            if not existing_id_unique:
                constraint_name = f"{meta.db_table}_id_unique"
                # Reassign constraints list to ensure mutability across Django versions
                current_constraints = list(getattr(meta, "constraints", []) or [])
                current_constraints.append(_dj_models.UniqueConstraint(fields=["id"], name=constraint_name))
                meta.constraints = current_constraints
        except Exception:
            # Be defensive: never break class creation in user projects/tests
            pass


def _ensure_unique_id_on_prepared(sender, **kwargs) -> None:
    """Attach a UniqueConstraint(id) to every concrete Timescale model post class prep."""
    try:
        if not issubclass(sender, TimescaleBaseMixin):  # type: ignore[arg-type]
            return
        meta = getattr(sender, "_meta", None)
        if not meta or getattr(meta, "abstract", True):
            return

        # Skip if already present
        for constraint in getattr(meta, "constraints", []) or []:
            if isinstance(constraint, _dj_models.UniqueConstraint) and list(getattr(constraint, "fields", [])) == [
                "id"
            ]:
                return

        constraint_name = f"{meta.db_table}_id_unique"
        meta.constraints = list(getattr(meta, "constraints", []) or []) + [
            _dj_models.UniqueConstraint(fields=["id"], name=constraint_name)
        ]
    except Exception:
        # Never crash signal handler
        return


# Connect once
_class_prepared.connect(_ensure_unique_id_on_prepared, dispatch_uid="p2d_timescale_unique_id")


class XmlTimescaleBase(Xml2DjangoBaseClass, TimescaleBaseMixin):
    class Meta:
        abstract = True


class PydanticTimescaleBase(Pydantic2DjangoBaseClass, TimescaleBaseMixin):
    class Meta:
        abstract = True


class DataclassTimescaleBase(Dataclass2DjangoBaseClass, TimescaleBaseMixin):
    class Meta:
        abstract = True
