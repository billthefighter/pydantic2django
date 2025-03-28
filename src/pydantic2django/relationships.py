import importlib
import logging
from dataclasses import dataclass, field
from typing import Optional, TypeVar

from django.db import models
from pydantic import BaseModel

from pydantic2django.context_storage import ModelContext

logger = logging.getLogger(__name__)


@dataclass
class RelationshipMapper:
    """
    Bidirectional mapper between objects
    """

    pydantic_model: type[BaseModel] | None
    django_model: type[models.Model] | None
    context: ModelContext | None


P = TypeVar("P", bound=BaseModel)
D = TypeVar("D", bound=models.Model)


@dataclass
class RelationshipConversionAccessor:
    available_relationships: list[RelationshipMapper] = field(default_factory=list)
    dependencies: Optional[dict[str, set[str]]] = field(default=None)

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
    def available_pydantic_models(self) -> list[type[BaseModel]]:
        """Get a list of all Pydantic models in the relationship accessor"""
        return [r.pydantic_model for r in self.available_relationships if r.pydantic_model is not None]

    @property
    def available_django_models(self) -> list[type[models.Model]]:
        """Get a list of all Django models in the relationship accessor"""
        return [r.django_model for r in self.available_relationships if r.django_model is not None]

    def add_pydantic_model(self, model: type[BaseModel]) -> None:
        """Add a Pydantic model to the relationship accessor"""
        # Check if the model is already in available_pydantic_models by comparing class names
        model_name = model.__name__
        existing_models = [m.__name__ for m in self.available_pydantic_models]

        if model_name not in existing_models:
            self.available_relationships.append(RelationshipMapper(model, None, context=None))

    def add_django_model(self, model: type[models.Model]) -> None:
        """Add a Django model to the relationship accessor"""
        # Check if the model is already in available_django_models by comparing class names
        model_name = model.__name__
        existing_models = [m.__name__ for m in self.available_django_models]

        if model_name not in existing_models:
            self.available_relationships.append(RelationshipMapper(None, model, context=None))

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

    def map_relationship(self, pydantic_model: type[BaseModel], django_model: type[models.Model]) -> None:
        """
        Create or update a mapping between a Pydantic model and a Django model
        """
        # First check if either model already exists in a relationship
        for relationship in self.available_relationships:
            if relationship.pydantic_model == pydantic_model:
                relationship.django_model = django_model
                return
            if relationship.django_model == django_model:
                relationship.pydantic_model = pydantic_model
                return

        # If neither model exists in a relationship, create a new one
        self.available_relationships.append(RelationshipMapper(pydantic_model, django_model, None))

    def has_pydantic_model(self, model: type[BaseModel]) -> bool:
        """Check if a specific Pydantic model is in the relationship accessor"""
        return any(r.pydantic_model == model for r in self.available_relationships)
