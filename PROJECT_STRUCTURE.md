# Project Structure

This document outlines the directory structure and key files of the `pydantic2django` project.

## Root Directory

-   [`.git/`](./.git/): Git version control directory.
-   [`.github/`](./.github/): GitHub specific files (workflows, issue templates, etc.) (Assumed, common practice)
-   [`.venv/`](./.venv/): Python virtual environment (typically excluded by `.gitignore`).
-   [`docs/`](./docs/): Project documentation (if present).
-   [`examples/`](./examples/): Contains example usage code and demonstrations.
-   [`scripts/`](./scripts/): Utility scripts for development or maintenance.
-   [`src/`](./src/): Contains the main source code for the library.
    -   [`pydantic2django/`](./src/pydantic2django/): The core package directory.
-   [`tests/`](./tests/): Contains all unit and integration tests.
-   [`.gitignore`](./.gitignore): Specifies intentionally untracked files that Git should ignore.
-   [`.pre-commit-config.yaml`](./.pre-commit-config.yaml): Configuration for pre-commit hooks.
-   [`LICENSE`](./LICENSE): Project license file (Assumed, add if present).
-   [`mypy.ini`](./mypy.ini): Configuration file for the MyPy type checker.
-   [`uv.lock`](./uv.lock): Defines exact, locked dependency versions managed by uv (generated via `uv lock` or `uv sync`).
-   [`pyproject.toml`](./pyproject.toml): Project metadata, dependencies, and tool configurations (like Ruff, Pytest) using PEP 621 with Hatchling; uv reads this to manage environments and installs.
-   [`README.md`](./README.md): The main introductory file for the project.
-   [`setup.py`](./setup.py): Legacy setup script (may be present for compatibility).
-   Other configuration files (`.ruff_cache/`, `.mypy_cache/`, `.pytest_cache/`, `.coverage`, `test_db.sqlite3`) are typically development artifacts.

## `src/pydantic2django/` Directory

This is the main package directory containing the library's core logic.

-   [`core/`](./src/pydantic2django/core/): Core components and abstractions shared across different parts of the library.
    -   [`relationships.py`](./src/pydantic2django/core/relationships.py): Manages mappings between source models (Pydantic/Dataclass) and Django models.
    -   [`context.py`](./src/pydantic2django/core/context.py): Handles context management during model processing or conversion.
    -   [`base.py`](./src/pydantic2django/core/base.py): Base classes or foundational elements.
-   [`dataclass/`](./src/pydantic2django/dataclass/): Modules specifically related to handling Python dataclasses.
    -   (Files related to dataclass processing, conversion, etc.)
-   [`django/`](./src/pydantic2django/django/): Components specific to Django integration.
    -   [`models.py`](./src/pydantic2django/django/models.py): Base Django models or mixins provided by the library.
    -   [`fields.py`](./src/pydantic2django/django/fields.py): Custom Django model fields.
    -   (Other Django-specific utilities)
-   [`pydantic/`](./src/pydantic2django/pydantic/): Modules specifically related to handling Pydantic models.
    -   (Files related to Pydantic model processing, conversion, etc.)
-   [`static_generators/`](./src/pydantic2django/static_generators/): Logic for statically generating Django model files from source models.
    -   [`model_generator.py`](./src/pydantic2django/static_generators/model_generator.py): The main class responsible for generating Django model code.
-   [`utils/`](./src/pydantic2django/utils/): General utility functions used throughout the library.
    -   [`discovery.py`](./src/pydantic2django/utils/discovery.py): Functions for discovering models in packages.
    -   (Other helper functions)
-   [`__init__.py`](./src/pydantic2django/__init__.py): Makes the directory a Python package and potentially exposes key components.

*Note: Specific file names within subdirectories like `dataclass/`, `django/`, `pydantic/`, and `utils/` might vary. The links point to the directories.*
