import dataclasses
import importlib
import logging
from dataclasses import dataclass, field
from typing import Optional, TypeVar, cast

from django.db import models
from pydantic import BaseModel

from .context import ModelContext

logger = logging.getLogger(__name__)


@dataclass
class RelationshipMapper:
    """
    Bidirectional mapper between source models (Pydantic/Dataclass) and Django models.
    """

    # Allow storing either source type
    pydantic_model: Optional[type[BaseModel]] = None
    dataclass_model: Optional[type] = None
    django_model: Optional[type[models.Model]] = None
    context: Optional[ModelContext] = None  # Keep context if needed later

    @property
    def source_model(self) -> Optional[type]:
        """Return the source model (either Pydantic or Dataclass)."""
        return self.pydantic_model or self.dataclass_model


P = TypeVar("P", bound=BaseModel)
D = TypeVar("D", bound=models.Model)


@dataclass
class RelationshipConversionAccessor:
    available_relationships: list[RelationshipMapper] = field(default_factory=list)
    # dependencies: Optional[dict[str, set[str]]] = field(default=None) # Keep if used

    @classmethod
    def from_dict(cls, relationship_mapping_dict: dict) -> "RelationshipConversionAccessor":
        """
        Convert a dictionary of strings representing model qualified names to a RelationshipConversionAccessor

        The dictionary should be of the form:
        {
            "pydantic_model_qualified_name": "django_model_qualified_name",
            ...
        }
        """
        available_relationships = []
        for pydantic_mqn, django_mqn in relationship_mapping_dict.items():
            try:
                # Split the module path and class name
                pydantic_module_path, pydantic_class_name = pydantic_mqn.rsplit(".", 1)
                django_module_path, django_class_name = django_mqn.rsplit(".", 1)

                # Import the modules
                pydantic_module = importlib.import_module(pydantic_module_path)
                django_module = importlib.import_module(django_module_path)

                # Get the actual class objects
                pydantic_model = getattr(pydantic_module, pydantic_class_name)
                django_model = getattr(django_module, django_class_name)

                available_relationships.append(RelationshipMapper(pydantic_model, django_model, context=None))
            except Exception as e:
                logger.warning(f"Error importing model {pydantic_mqn} or {django_mqn}: {e}")
                continue
        return cls(available_relationships)

    def to_dict(self) -> dict:
        """
        Convert the relationships to a dictionary of strings representing
        model qualified names for bidirectional conversion.

        Can be stored in a JSON field, and used to reconstruct the relationships.
        """
        relationship_mapping_dict = {}
        for relationship in self.available_relationships:
            # Skip relationships where either model is None
            if relationship.pydantic_model is None or relationship.django_model is None:
                continue

            pydantic_mqn = self._get_pydantic_model_qualified_name(relationship.pydantic_model)
            django_mqn = self._get_django_model_qualified_name(relationship.django_model)
            relationship_mapping_dict[pydantic_mqn] = django_mqn

        return relationship_mapping_dict

    def _get_pydantic_model_qualified_name(self, model: type[BaseModel] | None) -> str:
        """Get the fully qualified name of a Pydantic model as module.class_name"""
        if model is None:
            return ""
        return f"{model.__module__}.{model.__name__}"

    def _get_django_model_qualified_name(self, model: type[models.Model] | None) -> str:
        """Get the fully qualified name of a Django model as app_label.model_name"""
        if model is None:
            return ""
        return f"{model._meta.app_label}.{model.__name__}"

    @property
    def available_source_models(self) -> list[type]:
        """Get a list of all source models (Pydantic or Dataclass)."""
        models = []
        for r in self.available_relationships:
            if r.pydantic_model:
                models.append(r.pydantic_model)
            if r.dataclass_model:
                models.append(r.dataclass_model)
        return models

    @property
    def available_django_models(self) -> list[type[models.Model]]:
        """Get a list of all Django models in the relationship accessor"""
        return [r.django_model for r in self.available_relationships if r.django_model is not None]

    def add_pydantic_model(self, model: type[BaseModel]) -> None:
        """Add a Pydantic model to the relationship accessor"""
        # Check if the model is already in available_pydantic_models by comparing class names
        model_name = model.__name__
        existing_models = [m.__name__ for m in self.available_source_models]

        if model_name not in existing_models:
            self.available_relationships.append(RelationshipMapper(model, None, context=None))

    def add_dataclass_model(self, model: type) -> None:
        """Add a Dataclass model to the relationship accessor"""
        # Check if the model is already mapped
        if any(r.dataclass_model == model for r in self.available_relationships):
            return  # Already exists
        # Check if a Pydantic model with the same name is already mapped (potential conflict)
        if any(r.pydantic_model and r.pydantic_model.__name__ == model.__name__ for r in self.available_relationships):
            logger.warning(f"Adding dataclass {model.__name__}, but a Pydantic model with the same name exists.")

        self.available_relationships.append(RelationshipMapper(dataclass_model=model))

    def add_django_model(self, model: type[models.Model]) -> None:
        """Add a Django model to the relationship accessor"""
        # Check if the model is already in available_django_models by comparing class names
        model_name = model.__name__
        existing_models = [m.__name__ for m in self.available_django_models]

        if model_name not in existing_models:
            self.available_relationships.append(RelationshipMapper(None, None, model, context=None))

    def get_django_model_for_pydantic(self, pydantic_model: type[BaseModel]) -> Optional[type[models.Model]]:
        """
        Find the corresponding Django model for a given Pydantic model

        Returns None if no matching Django model is found
        """
        for relationship in self.available_relationships:
            if relationship.pydantic_model == pydantic_model and relationship.django_model is not None:
                return relationship.django_model
        return None

    def get_pydantic_model_for_django(self, django_model: type[models.Model]) -> Optional[type[BaseModel]]:
        """
        Find the corresponding Pydantic model for a given Django model

        Returns None if no matching Pydantic model is found
        """
        for relationship in self.available_relationships:
            if relationship.django_model == django_model and relationship.pydantic_model is not None:
                return relationship.pydantic_model
        return None

    def get_django_model_for_dataclass(self, dataclass_model: type) -> Optional[type[models.Model]]:
        """Find the corresponding Django model for a given Dataclass model."""
        logger.debug(f"Searching for Django model matching dataclass: {dataclass_model.__name__}")
        for relationship in self.available_relationships:
            # Check if this mapper holds the target dataclass and has a linked Django model
            if relationship.dataclass_model == dataclass_model and relationship.django_model is not None:
                logger.debug(f"  Found match: {relationship.django_model.__name__}")
                return relationship.django_model
        logger.debug(f"  No match found for dataclass {dataclass_model.__name__}")
        return None

    def map_relationship(self, source_model: type, django_model: type[models.Model]) -> None:
        """
        Create or update a mapping between a source model (Pydantic/Dataclass) and a Django model.
        """
        source_type = (
            "pydantic"
            if isinstance(source_model, type) and issubclass(source_model, BaseModel)
            else "dataclass"
            if dataclasses.is_dataclass(source_model)
            else "unknown"
        )

        if source_type == "unknown":
            logger.warning(f"Cannot map relationship for unknown source type: {source_model}")
            return

        # Check if either model already exists in a relationship
        for relationship in self.available_relationships:
            if source_type == "pydantic" and relationship.pydantic_model == source_model:
                relationship.django_model = django_model
                # Ensure dataclass_model is None if we map pydantic
                relationship.dataclass_model = None
                logger.debug(f"Updated mapping: Pydantic {source_model.__name__} -> Django {django_model.__name__}")
                return
            if source_type == "dataclass" and relationship.dataclass_model == source_model:
                relationship.django_model = django_model
                # Ensure pydantic_model is None
                relationship.pydantic_model = None
                logger.debug(f"Updated mapping: Dataclass {source_model.__name__} -> Django {django_model.__name__}")
                return
            if relationship.django_model == django_model:
                # Map the source model based on its type
                if source_type == "pydantic":
                    relationship.pydantic_model = cast(type[BaseModel], source_model)
                    relationship.dataclass_model = None
                    logger.debug(
                        f"Updated mapping: Pydantic {source_model.__name__} -> Django {django_model.__name__} (found via Django model)"
                    )
                elif source_type == "dataclass":
                    relationship.dataclass_model = cast(type, source_model)
                    relationship.pydantic_model = None
                    logger.debug(
                        f"Updated mapping: Dataclass {source_model.__name__} -> Django {django_model.__name__} (found via Django model)"
                    )
                return

        # If no existing relationship found, create a new one
        logger.debug(
            f"Creating new mapping: {source_type.capitalize()} {source_model.__name__} -> Django {django_model.__name__}"
        )
        if source_type == "pydantic":
            self.available_relationships.append(
                RelationshipMapper(pydantic_model=cast(type[BaseModel], source_model), django_model=django_model)
            )
        elif source_type == "dataclass":
            self.available_relationships.append(
                RelationshipMapper(dataclass_model=cast(type, source_model), django_model=django_model)
            )

    def is_source_model_known(self, model: type) -> bool:
        """Check if a specific source model (Pydantic or Dataclass) is known."""
        is_pydantic = isinstance(model, type) and issubclass(model, BaseModel)
        is_dataclass = dataclasses.is_dataclass(model)

        for r in self.available_relationships:
            if is_pydantic and r.pydantic_model == model:
                return True
            if is_dataclass and r.dataclass_model == model:
                return True
        return False

    # Add a method to lookup source type by name
    def get_source_model_by_name(self, model_name: str) -> Optional[type]:
        """Find a known source model (Pydantic or Dataclass) by its class name."""
        for r in self.available_relationships:
            if r.pydantic_model and r.pydantic_model.__name__ == model_name:
                return r.pydantic_model
            if r.dataclass_model and r.dataclass_model.__name__ == model_name:
                return r.dataclass_model
        return None
