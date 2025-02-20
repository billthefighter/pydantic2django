"""
Admin interface generation for pydantic2django.

This module provides functionality for automatically generating Django admin interfaces
for dynamically created models.
"""
import logging

from django.apps import apps
from django.contrib import admin
from django.db import models

logger = logging.getLogger(__name__)


class DynamicModelAdmin(admin.ModelAdmin):
    """Dynamic admin interface for discovered models."""

    list_display = ("id",)  # Start with basic fields
    search_fields = ("id",)

    def get_list_display(self, request):
        """Dynamically determine which fields to display."""
        model = self.model
        # Start with id field
        fields = ["id"]
        # Add other fields that might be interesting
        for field in model._meta.fields:
            if field.name != "id":
                fields.append(field.name)
        return fields


def register_model_admin(model: type[models.Model], model_name: str) -> None:
    """Register a single model with the Django admin interface.

    Args:
        model: The Django model to register
        model_name: The name of the model for logging purposes
    """
    try:
        admin.site.register(model, DynamicModelAdmin)
        logger.info(f"Registered {model_name} with admin")
    except admin.sites.AlreadyRegistered:
        logger.debug(f"Admin interface for {model_name} already registered")


def register_model_admins(app_label: str) -> None:
    """Register admin interfaces for all models in the app.

    This should be called after Django is fully initialized and migrations are complete.

    Args:
        app_label: The Django app label the models are registered under
    """
    logger.info(f"Registering admin interfaces for {app_label}...")

    for model in apps.get_app_config(app_label).get_models():
        try:
            admin.site.register(model, DynamicModelAdmin)
            logger.info(f"Registered admin interface for {model.__name__}")
        except admin.sites.AlreadyRegistered:
            logger.info(f"Admin interface for {model.__name__} already registered")
