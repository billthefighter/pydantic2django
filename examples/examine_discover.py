import inspect
import logging
import os
import sys
from collections import defaultdict

# --- Django Setup (MUST come before pydantic2django imports) ---
import django
from django.conf import settings

# Configure Django settings if not already configured
if not settings.configured:
    logger = logging.getLogger("DjangoSetup")  # Temp logger for setup
    logger.info("Configuring Django settings...")
    settings.configure(
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            # "django_llm", # Don't need the actual app installed for discovery test
        ],
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        },
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
    )
    django.setup()
    logger.info("Django setup complete.")
# -------------------------------------------------------------

# --- Setup Project Path ---
# Assuming this script is in the project root (parent of LLMaestro and src)
project_root = os.path.dirname(os.path.abspath(__file__))
llmaestro_src_path = os.path.join(project_root, "LLMaestro", "src")

if llmaestro_src_path not in sys.path:
    sys.path.insert(0, llmaestro_src_path)
    print(f"Added {llmaestro_src_path} to sys.path")

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.DEBUG,  # Capture all levels
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # filename="discovery_test.log", # Optional: Log to a file
    # filemode="w",                # Optional: Overwrite log file each time
)
logger = logging.getLogger("DiscoveryTestScript")
# Make pydantic2django logs more verbose if needed
logging.getLogger("pydantic2django.discovery").setLevel(logging.DEBUG)
logging.getLogger("pydantic2django.factory").setLevel(logging.DEBUG)  # Also useful

# --- Imports (After Path Setup) ---
try:
    from pydantic import BaseModel

    from llmaestro.core.persistence import PersistentModel  # Import the base class for the filter
    from pydantic2django.discovery import ModelDiscovery
except ImportError as e:
    logger.error(f"Failed to import necessary modules: {e}")
    logger.error("Ensure LLMaestro/src is in the Python path and dependencies are installed.")
    sys.exit(1)


# --- Filter Function ---
def is_persistent_model(obj):
    """
    Filter function that returns True if a class is a subclass of PersistentModel.
    """
    # Check if it's a class, is a subclass of PersistentModel, and not PersistentModel itself
    is_class = inspect.isclass(obj)
    is_subclass = False
    if is_class:
        try:
            # Handle potential errors if obj is not a class with __mro__
            is_subclass = issubclass(obj, PersistentModel)
        except TypeError:
            is_subclass = False  # Not a class or incompatible type

    return is_class and is_subclass and obj != PersistentModel


# --- Main Test Logic ---
def run_discovery_test():
    """Runs the model discovery test."""
    logger.info("--- Starting Discovery Test ---")

    discovery = ModelDiscovery()
    app_label = "test_llmaestro_app"  # Use a distinct app label for testing

    # --- Step 1: Discover Models ---
    logger.info(f"--- Calling discover_models(packages=['llmaestro'], app_label='{app_label}') ---")
    try:
        # Use a filter that only selects subclasses of PersistentModel
        # Adjust filter signature to match implementation (takes only model_class)
        def discovery_filter(model_class: type[BaseModel]) -> bool:
            return is_persistent_model(model_class)

        # Pass the adjusted filter function
        discovery.discover_models(package_names=["llmaestro"], app_label=app_label, filter_function=discovery_filter)
    except Exception:
        logger.exception("Error during model discovery")
        return  # Stop if discovery fails

    logger.info(f"Total models discovered initially: {len(discovery.discovered_models)}")
    logger.info(f"Models kept after filtering: {len(discovery.filtered_models)}")
    logger.debug("Filtered models list:")
    for name, cls in discovery.filtered_models.items():
        logger.debug(f"  - {name}: {cls.__module__}.{cls.__name__}")

    # Log where models were discovered from (might show duplicates)
    model_origins = defaultdict(list)
    for name, cls in discovery.discovered_models.items():
        model_origins[f"{cls.__module__}.{cls.__name__}"].append(f"Found as '{name}'")

    logger.info("--- Potential Model Origins (Check for duplicates by FQN) ---")
    for fqn, locations in model_origins.items():
        if len(locations) > 1:
            logger.warning(f"Duplicate Discovery? Model {fqn} found multiple times: {locations}")
        else:
            logger.debug(f"Model {fqn}: {locations}")

    # --- Step 2: Analyze Dependencies ---
    logger.info("--- Calling analyze_dependencies() ---")
    try:
        discovery.analyze_dependencies(app_label=app_label)
    except Exception:
        logger.exception("Error during dependency analysis")
        return

    logger.info("Calculated Dependencies (based on filtered models):")
    if not discovery.dependencies:
        logger.warning("No dependencies were calculated.")
    for model_name, deps in discovery.dependencies.items():
        # Only show dependencies for models that passed the filter
        if model_name in discovery.filtered_models:
            logger.debug(f"  - {model_name}: {deps if deps else '{}'}")

    # --- Step 3: Validate Dependencies ---
    logger.info("--- Calling validate_dependencies() ---")
    missing = discovery.validate_dependencies()
    if missing:
        logger.error(f"Validation Error: Missing dependencies found: {missing}")
    else:
        logger.info("Dependency validation passed (all referenced models found within filtered set).")

    # --- Step 4: Get Registration Order ---
    logger.info("--- Calling get_registration_order() ---")
    try:
        order = discovery.get_registration_order()  # This also calls validate_dependencies implicitly now
        logger.info("Calculated Registration Order:")
        for i, model_name in enumerate(order):
            logger.info(f"  {i+1}. {model_name}")
            # Additionally log the calculated dependencies for each model in the final order
            deps = discovery.dependencies.get(model_name, set())
            logger.debug(f"     Dependencies: {deps if deps else '{}'}")

    except Exception:
        logger.exception("Error getting registration order")

    logger.info("--- Discovery Test Finished ---")


if __name__ == "__main__":
    run_discovery_test()
