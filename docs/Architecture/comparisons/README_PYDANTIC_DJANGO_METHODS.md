# Pydantic to Django Method Integration Approaches

This document compares two different approaches for integrating Pydantic model methods with Django models in the pydantic2django library.

## Overview of Approaches

The library provides two distinct mechanisms for making Pydantic model methods available on Django models:

1. **Static Method Copying** (`methods.py`)
2. **Dynamic Method Resolution** (`__getattr__` in `Pydantic2DjangoBaseClass`)

## Core Approach Differences

### `methods.py` Approach
- **Static Method Copying**: Copies methods from Pydantic models to Django models at class definition time
- **Direct Integration**: Methods become actual attributes of the Django model class
- **One-time Process**: Methods are copied once when the Django model is created
- **Transformation-based**: Transforms and wraps methods to handle return type conversions

### `__getattr__` Approach in `Pydantic2DjangoBaseClass`
- **Dynamic Method Resolution**: Intercepts attribute access at runtime
- **Delegation Pattern**: Forwards method calls to the Pydantic model
- **On-demand Conversion**: Converts Django model to Pydantic instance for each method call
- **Proxy-based**: Acts as a proxy to the Pydantic model's methods

## Technical Implementation Differences

### Method Access Timing
- `methods.py`: Methods are available at class definition time
- `__getattr__`: Methods are resolved dynamically at runtime

### Performance Characteristics
- `methods.py`: Better performance as methods are directly available
- `__getattr__`: Incurs overhead for each method call (conversion + delegation)

### Memory Usage
- `methods.py`: Higher memory usage as methods are duplicated
- `__getattr__`: Lower memory usage as methods are not duplicated

### Method Return Handling
- `methods.py`: Sophisticated handling of return types with conversion based on type annotations
- `__getattr__`: Simpler approach that returns results directly from Pydantic method calls

### Method Type Preservation
- `methods.py`: Preserves method types (regular, class, static) with proper wrapping
- `__getattr__`: Only handles instance methods, not class or static methods

## Advantages and Disadvantages

### `methods.py` Approach

#### Advantages
- Better IDE support (methods appear in autocompletion)
- Better performance (no runtime conversion for each call)
- Handles class methods and static methods
- Sophisticated return type conversion based on annotations
- Works with inheritance (methods are part of the class)

#### Disadvantages
- More complex implementation
- Duplicates code (methods exist in both Pydantic and Django models)
- Requires explicit copying of methods
- May get out of sync if Pydantic model changes

### `__getattr__` Approach

#### Advantages
- Simpler implementation
- No code duplication (methods only exist in Pydantic model)
- Automatically handles new methods added to Pydantic model
- Consistent conversion between Django and Pydantic models

#### Disadvantages
- Poorer IDE support (methods don't appear in autocompletion)
- Performance overhead for each method call
- Doesn't handle class methods or static methods
- Less control over return type conversion
- May not work well with inheritance

## How `factory.py` and `discovery.py` Fit In

Both `factory.py` and `discovery.py` are foundational components that work with either method integration approach:

### `factory.py`
- Creates the Django model classes from Pydantic models
- Sets up the `object_type` reference that enables the `__getattr__` approach
- Provides the structure that both method integration approaches build upon
- **Does not directly implement either approach**, but creates the necessary infrastructure

### `discovery.py`
- Provides high-level functionality for discovering and registering models
- Uses `factory.py` to create Django models
- Manages model dependencies and registration order
- **Works with both approaches** and doesn't explicitly choose one over the other

In practice, the library's architecture allows for flexibility:

1. `factory.py` creates Django model classes with proper type hints
2. The `__getattr__` approach in `Pydantic2DjangoBaseClass` provides automatic method delegation
3. `methods.py` can be used optionally to copy specific methods for better performance and IDE support
4. `discovery.py` provides high-level discovery and registration regardless of which method approach is used

## Use Case Differences

### Development Experience
- `methods.py` provides better IDE support and type checking
- `__getattr__` is more transparent but less discoverable

### Maintenance
- `methods.py` requires updating Django models when Pydantic models change
- `__getattr__` automatically adapts to changes in Pydantic models

### Performance Requirements
- `methods.py` is better for performance-critical code
- `__getattr__` is simpler but has runtime overhead

## How They Complement Each Other

These approaches can be complementary:

1. `factory.py` creates Django model classes
2. `methods.py` can copy frequently used methods directly to the Django model
3. `__getattr__` can serve as a fallback for less common methods

This combination would provide:
- Direct access to common methods with good performance
- Fallback access to all other methods
- Best of both worlds for IDE support and flexibility

## Practical Implications

### For Library Users
- `methods.py` approach requires explicit copying of methods
- `__getattr__` approach works automatically for all methods

### For Library Maintainers
- `methods.py` provides more control over method behavior
- `__getattr__` is easier to maintain but has less control

### For Performance
- `methods.py` is better for frequently called methods
- `__getattr__` is acceptable for occasionally used methods

## Conclusion

The two approaches represent different design philosophies:

- `methods.py` follows a **static, explicit** approach that prioritizes performance and IDE support
- `__getattr__` follows a **dynamic, implicit** approach that prioritizes simplicity and maintenance

The choice between them depends on specific requirements around performance, maintainability, and developer experience. In practice, a hybrid approach might offer the best balance, using `methods.py` for critical methods and `__getattr__` as a fallback.
