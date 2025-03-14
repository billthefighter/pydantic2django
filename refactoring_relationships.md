# Refactoring Plan: Centralizing Relationship Handling in pydantic2django

## Overview

This document outlines the plan to refactor relationship handling in pydantic2django to follow the Single Responsibility Principle by centralizing all relationship logic in the `RelationshipFieldHandler` class.

## Current Issues

1. **Scattered Relationship Logic**
   - `FieldTypeResolver` contains relationship detection
   - `TypeMapper` has relationship mapping logic
   - `RelationshipFieldHandler` has relationship handling
   - `static_django_model_generator.py` contains relationship generation logic

2. **Inconsistent Implementations**
   - Different components may handle relationships differently
   - No single source of truth for relationship behavior
   - Risk of inconsistent behavior across the codebase

3. **Circular Dependencies**
   - `RelationshipFieldHandler` depends on `FieldTypeResolver`
   - `FieldTypeResolver` contains its own relationship logic
   - Complex dependency graph makes changes risky

## Refactoring Goals

1. Make `RelationshipFieldHandler` the single source of truth for:
   - Relationship detection
   - Relationship field creation
   - Relationship type mapping
   - Relationship metadata handling
   - Relationship serialization

2. Eliminate duplicate logic in other components
3. Simplify the dependency graph
4. Improve testability and maintainability

## Implementation Plan

### Phase 1: Enhance RelationshipFieldHandler

1. Add new methods to `RelationshipFieldHandler`:
```python
class RelationshipFieldHandler:
    @staticmethod
    def detect_field_type(field_type: Any) -> tuple[Optional[type[models.Field]], dict[str, Any]]:
        """Single entry point for relationship type detection"""

    @staticmethod
    def create_field(field_name: str, field_info: FieldInfo, field_type: Any,
                    app_label: str, model_name: Optional[str] = None) -> Optional[models.Field]:
        """Single entry point for creating relationship fields"""

    @staticmethod
    def get_relationship_metadata(field_type: Any) -> dict[str, Any]:
        """Extract relationship-specific metadata"""

    @staticmethod
    def serialize_relationship(field: models.Field, value: Any) -> Any:
        """Handle relationship field serialization"""
```

2. Move existing relationship logic from other classes into these methods
3. Add comprehensive tests for the new methods

### Phase 2: Update Dependent Components

1. Update `FieldTypeResolver`:
```python
class FieldTypeResolver:
    @staticmethod
    def resolve_field_type(field_type: Any) -> tuple[type[models.Field], dict[str, Any]]:
        # Delegate relationship handling
        field_class, kwargs = RelationshipFieldHandler.detect_field_type(field_type)
        if field_class:
            return field_class, kwargs

        # Continue with normal field resolution
```

2. Update `TypeMapper`:
```python
class TypeMapper:
    @classmethod
    def get_mapping_for_type(cls, python_type: Any) -> Optional[TypeMappingDefinition]:
        # Delegate relationship handling
        field_class, kwargs = RelationshipFieldHandler.detect_field_type(python_type)
        if field_class:
            return TypeMappingDefinition(
                python_type=python_type,
                django_field=field_class,
                **kwargs
            )
```

3. Update `static_django_model_generator.py` to use the centralized relationship handling

### Phase 3: Clean Up and Documentation

1. Remove deprecated relationship handling code from:
   - `FieldTypeResolver`
   - `TypeMapper`
   - Other components

2. Update documentation:
   - Add clear relationship handling guidelines
   - Document the single responsibility principle
   - Update API documentation

3. Add migration guides for any breaking changes

### Phase 4: Testing and Validation

1. Add new test cases:
   - Relationship detection
   - Field creation
   - Metadata handling
   - Serialization
   - Edge cases

2. Update existing tests to use the new centralized handling

3. Add integration tests for:
   - Complex relationship scenarios
   - Circular relationships
   - Inheritance relationships

## Breaking Changes

1. **API Changes**:
   - Deprecated relationship methods in other classes
   - New centralized methods in `RelationshipFieldHandler`
   - Updated method signatures for consistency

2. **Behavior Changes**:
   - Standardized relationship handling
   - Consistent metadata extraction
   - Unified serialization approach

## Migration Guide

### For Library Users

1. Replace direct usage of `FieldTypeResolver` relationship methods:
```python
# Old
field_type = FieldTypeResolver.is_relationship_field(field_type)

# New
field_type = RelationshipFieldHandler.detect_field_type(field_type)[0]
```

2. Update relationship field creation:
```python
# Old
field = create_relationship_field(...)

# New
field = RelationshipFieldHandler.create_field(...)
```
