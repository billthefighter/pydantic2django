# TypedClass to Django Model Conversion

## Overview

This module provides an **experimental** mechanism to convert generic Python classes (i.e., classes that are not Pydantic models or standard Python dataclasses) into Django models. The primary goal is to enable the persistence of instances of these arbitrary classes in a Django database.

## Motivation

While `pydantic2django` excels at converting Pydantic models and dataclasses, there are scenarios where one might want to store the state of other types of Python objects. This could be for:

-   Configuration objects.
-   Instances of classes from third-party libraries where modifying their structure (e.g., to be a Pydantic model) is not feasible.
-   Storing simplified representations of more complex runtime objects, such as those from `pydantic-ai` (e.g., `Provider` or `Model` instances), focusing on their configurable attributes rather than their full runtime state.

## Mechanism

The conversion process follows a similar pattern to the `dataclass` and `pydantic` converters:

1.  **Discovery (`typedclass.discovery.TypedClassDiscovery`)**:
    *   Scans specified Python packages for classes.
    *   Filters out Pydantic models, dataclasses, and abstract base classes (ABCs).
    *   Identifies remaining classes as potential candidates for conversion.
    *   Analyzes dependencies between these classes by inspecting type hints in `__init__` method signatures and class-level attribute annotations.

2.  **Factory (`typedclass.factory.TypedClassModelFactory` & `typedclass.factory.TypedClassFieldFactory`)**:
    *   For each discovered class, it extracts attribute information primarily from its `__init__` parameters (name, type hint, default value) and any additional class-level typed attributes not covered by `__init__`.
    *   The `TypedClassFieldFactory` attempts to map the Python types of these attributes to appropriate Django model fields (e.g., `str` to `models.CharField`, `int` to `models.IntegerField`).

3.  **Generator (`typedclass.generator.TypedClassDjangoModelGenerator`)**:
    *   Orchestrates the discovery and model generation process.
    *   Uses Jinja2 templates to render the final `models.py` file content, including the generated Django model classes.

## Modes of Operation (Planned)

-   **`typed_only` (Default - Initial Implementation Focus)**: The system will only attempt to convert attributes that have recognizable simple Python types (like `int`, `str`, `bool`, `datetime.datetime`, `list` of simple types) or are typed as another discovered class (which would become a relational field like `ForeignKey`). Attributes with complex, unrecognized, or missing type hints will be skipped with a warning or defaulted to a non-ideal type like `TextField`.
-   **`reckless` (Future Enhancement)**: This mode would be more aggressive:
    *   For attributes with complex or unrecognized types, it might default to storing them in a `models.JSONField` (attempting a best-effort serialization) or a `models.TextField` (storing `repr()`).
    *   It could optionally generate placeholder serialization/deserialization helper methods or properties within the Django model (e.g., `serialize_<attribute_name>()` and `deserialize_<attribute_name>()`) that the user would need to implement.

## Intended Limitations & Warnings

This feature is highly experimental and comes with significant limitations:

1.  **Brittleness**: Converting arbitrary Python classes is inherently more complex and less predictable than converting structured Pydantic models or dataclasses. Expect this module to be somewhat brittle, especially with unusual class structures or types.
2.  **Reliance on Type Hints**: The conversion heavily relies on accurate and available type hints in `__init__` methods and for class-level attributes. Missing, incorrect, or overly generic (`Any`) type hints will likely lead to suboptimal or incorrect Django model fields.
3.  **Complex/Dynamic Attributes**:
    *   Attributes that are dynamically assigned (e.g., inside methods after `__init__`) and are not declared via `__init__` parameters or class-level annotations will **not** be discovered or converted.
    *   Trying to directly persist highly complex, stateful, or non-data-centric objects (e.g., `asyncio` event loops, open file handles, active network connections, or the `AsyncOpenAI` client instance itself from `pydantic-ai`) is generally ill-advised and often not meaningful for database storage. The goal should be to store the *configuration* or a serializable *data representation* of such objects, not their live, ephemeral state.
4.  **Serialization of Complex Objects**: Even in a future "reckless mode" using `JSONField`:
    *   Not all Python objects are JSON-serializable by default. Custom serialization logic (which this tool might only provide placeholders for) will often be required.
    *   Restoring a fully functional Python instance from a JSON representation can be non-trivial and require significant custom code, especially if the object has complex internal state or dependencies.
5.  **Circular Dependencies**: While the discovery mechanism attempts to order model generation to handle dependencies, serializing deeply nested or circularly dependent object structures (especially into JSON) can be problematic and might lead to recursion errors or incomplete data.
6.  **Methods and Behavior**: This tool focuses *only* on data attributes for persistence. Class methods, properties with complex logic (beyond simple attribute access), and other callable behaviors are **not** translated or stored as part of the Django model's structure. The generated Django models will be data containers.
7.  **Not a Full ORM for Generic Classes**: This utility is a schema *conversion* tool, not a comprehensive Object-Relational Mapper (ORM) for arbitrary Python classes. It generates Django model definitions; it does not provide mechanisms for automatically fetching, updating, or managing instances of your original generic classes through Django's ORM in the same way Django models work.

## Key Areas to Test (TypedClass Specific)

Beyond the standard tests for model generation and field mapping common to other converters, the `typedclass` module requires specific testing for:

1.  **Class Discovery (`TypedClassDiscovery`)**:
    *   Correct identification of generic classes vs. Pydantic models, dataclasses, and ABCs.
    *   Accurate dependency analysis based on `__init__` type hints (simple types, other discovered typed classes, `Optional`, basic generics like `list[OtherTypedClass]`).
    *   Correct dependency analysis from class-level attribute type hints.
    *   Graceful handling of classes with non-standard `__init__` methods (e.g., those using `*args`, `**kwargs` extensively without clear type hints for parameters intended for storage, or `__init__` from C extensions).
2.  **Field Extraction (`TypedClassModelFactory._get_model_fields_info`)**:
    *   Prioritization of `__init__` parameters over class-level annotations if names collide.
    *   Correct extraction of attribute name, type hint, and default value from `__init__`.
    *   Correct extraction of class-level attributes (with type hints) not present in `__init__`.
    *   Behavior with attributes lacking type hints (should be skipped or handled by a defined strategy, e.g., `Any` -> `TextField` in a future reckless mode).
3.  **Type Translation (`TypedClassTypeTranslator` & `TypedClassFieldFactory`)**:
    *   Mapping of common Python types (`int`, `str`, `float`, `bool`, `datetime.date`, `datetime.datetime`, `uuid.UUID`) to appropriate Django fields.
    *   Handling of `Optional[X]` to `null=True, blank=True`.
    *   Initial handling of simple generics like `list[str]` (likely to `JSONField` or `TextField` initially, unless a more specific strategy is developed).
    *   Correct generation of relational fields (`ForeignKey`) if an attribute is typed as another discovered `TypedClassType`.
    *   Behavior when a type cannot be mapped (default to `TextField` or skip with warning, per chosen mode).
4.  **Django Model Base Functionality (`TypedClass2DjangoBaseClass`, `TypedClass2DjangoStoreTypedClassObject`)**:
    *   `from_typedclass()`: Correctly populating the Django model (either mapped fields or the `data` JSON field) from an instance of the source generic class.
    *   `to_typedclass()`: Correctly reconstructing an instance of the source generic class from the Django model. This is particularly challenging for `TypedClass2DjangoStoreTypedClassObject` and will rely heavily on how `_extract_data_from_typedclass` works and the nature of the class's `__init__`.
    *   Serialization/deserialization in `_extract_data_from_typedclass` (for store-as-JSON) and the reverse process in `to_typedclass` must be somewhat symmetrical for basic types.
    *   Robustness when source classes have attributes that are complex, non-standard, or not easily serializable (especially for `TypedClass2DjangoStoreTypedClassObject`).
5.  **Targeted `pydantic-ai` classes** (e.g., `OpenAIProvider`, `OpenAIModel`):
    *   Identify which attributes are primarily for configuration and are suitable for persistence.
    *   How attributes like `client: AsyncOpenAI` (which should *not* be directly persisted) are handled (e.g., skipped by type, or require manual exclusion if running in a future reckless mode).
    *   Test conversion of fields like `model_name: str`, `api_key: str | None`, `base_url: str | None`.

## Use Cases & Best Practices

-   **Best for**: Classes that primarily act as data containers or configuration holders, even if they are not Pydantic models or dataclasses.
-   **Focus on Serializable Data**: Prioritize converting classes or attributes whose state can be meaningfully represented by standard database types or JSON.
-   **Manual Adjustments**: Expect that the generated `models.py` file may often require manual review, adjustment, and refinement, especially for complex types or relationships.
-   **Custom Serialization**: Be prepared to write custom serialization/deserialization logic if using a "reckless mode" for complex types stored in `JSONField` or `TextField`.

This feature aims to provide a helpful starting point for a challenging problem, but careful consideration of these limitations is crucial for successful use.
