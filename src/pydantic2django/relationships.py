import importlib
import logging
from dataclasses import dataclass

from django.db import models
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class RelationshipMapper:
    """
    Bidirectional mapper between objects
    """

    pydantic_model: type[BaseModel]
    django_model: type[models.Model]


@dataclass
class RelationshipConversionAccessor:
    available_relationships: list[RelationshipMapper]

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

                available_relationships.append(RelationshipMapper(pydantic_model, django_model))
            except Exception as e:
                logger.warning(f"Error importing model {pydantic_mqn} or {django_mqn}: {e}")
                continue
        return cls(available_relationships)

    def to_dict(self, obj: BaseModel) -> dict:
        """
        Convert a Pydantic model to a dictionary of strings representing
        model qualified names for bidirection conversion.

        Can be stored in a JSON field, and used to convert back to a Pydantic model.
        """
        relationship_mapping_dict = {}
        for relationship in self.available_relationships:
            pydantic_mqn = self._get_pydantic_model_qualified_name(relationship.pydantic_model)
            django_mqn = self._get_django_model_qualified_name(relationship.django_model)
            relationship_mapping_dict[pydantic_mqn] = django_mqn

        return relationship_mapping_dict

    def _get_pydantic_model_qualified_name(self, model: type[BaseModel]) -> str:
        """Get the fully qualified name of a Pydantic model as module.class_name"""
        return f"{model.__module__}.{model.__name__}"

    def _get_django_model_qualified_name(self, model: type[models.Model]) -> str:
        """Get the fully qualified name of a Django model as app_label.model_name"""
        return f"{model._meta.app_label}.{model.__name__}"

    @property
    def available_pydantic_models(self) -> list[type[BaseModel]]:
        """Get a list of all Pydantic models in the relationship accessor"""
        return [relationship.pydantic_model for relationship in self.available_relationships]

    @property
    def available_django_models(self) -> list[type[models.Model]]:
        """Get a list of all Django models in the relationship accessor"""
        return [relationship.django_model for relationship in self.available_relationships]
