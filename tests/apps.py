from django.apps import AppConfig


class TestsConfig(AppConfig):
    name = "tests"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self):
        # Import models here to ensure Django discovers them
        # when the app is ready.
        from .fixtures import fixtures  # noqa: F401 (unused import)

        # Although the import is unused directly here, importing the module
        # ensures the models defined within it are registered with Django.
