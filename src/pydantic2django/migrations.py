"""
Migration state comparison functionality for Pydantic to Django models.
"""
from typing import Any

from django.db import connections, models
from django.db.migrations.autodetector import MigrationAutodetector
from django.db.migrations.executor import MigrationExecutor
from django.db.migrations.questioner import NonInteractiveMigrationQuestioner
from django.db.migrations.state import ModelState, ProjectState


def get_migration_operations(model: type[models.Model]) -> list[str]:
    """
    Compare a Django model against the current migration state and return needed operations.

    Args:
        model: The Django model to check against migrations

    Returns:
        A list of needed migration operations as strings, empty if model matches migrations
    """
    connection = connections["default"]
    executor = MigrationExecutor(connection)

    # Get the current migration state
    try:
        project_state = executor.loader.project_state((model._meta.app_label, None))
    except Exception:
        # If there are no migrations, use an empty state
        project_state = ProjectState()

    # Create a new state with our model
    final_state = ProjectState()
    model_state = ModelState.from_model(model)
    final_state.add_model(model_state)

    # Compare states
    autodetector = MigrationAutodetector(project_state, final_state, NonInteractiveMigrationQuestioner())

    changes = autodetector.changes(graph=executor.loader.graph)

    # Extract operations
    operations = []
    if model._meta.app_label in changes:
        for migration in changes[model._meta.app_label]:
            for operation in migration.operations:
                operations.append(str(operation))

    return operations


def check_model_migrations(model: type[models.Model]) -> tuple[bool, list[str]]:
    """
    Check if a Django model matches its current migration state.

    Args:
        model: The Django model to check

    Returns:
        A tuple of (is_synchronized: bool, needed_operations: list[str])
    """
    operations = get_migration_operations(model)
    return len(operations) == 0, operations


def get_table_description(model: type[models.Model]) -> list[dict[str, Any]]:
    """
    Get the current database table description for a model.

    Args:
        model: The Django model to inspect

    Returns:
        A list of column descriptions from the database
    """
    connection = connections["default"]
    with connection.cursor() as cursor:
        return connection.introspection.get_table_description(cursor, model._meta.db_table)
