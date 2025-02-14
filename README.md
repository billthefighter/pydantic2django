# Pydantic2Django

Convert Pydantic models to Django models dynamically.

[![PyPI version](https://badge.fury.io/py/pydantic2django.svg)](https://badge.fury.io/py/pydantic2django)
[![Python Versions](https://img.shields.io/pypi/pyversions/pydantic2django.svg)](https://pypi.org/project/pydantic2django/)
[![Django Versions](https://img.shields.io/badge/django-3.2%2B-blue.svg)](https://www.djangoproject.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Pydantic2Django is a library that allows you to use Pydantic models as Django models without recreating them. It automatically syncs between Pydantic and Django models, minimizing manual configuration.

## Features

- Automatic conversion of Pydantic models to Django models
- Field type mapping between Pydantic and Django
- Migration state validation
- IDE completion support
- Preservation of Pydantic model methods

## Installation

```bash
pip install pydantic2django
# or with poetry
poetry add pydantic2django
```

## Quick Start

```python
from pydantic import BaseModel
from pydantic2django import make_django_model

class UserModel(BaseModel):
    name: str
    age: int
    email: str

# Convert to Django model
DjangoUserModel = make_django_model(UserModel)
```

## Migration Detection

The library includes functionality to detect when your Pydantic models are out of sync with your Django database migrations. This helps ensure that your database schema stays in sync with your Pydantic models.

When you create a Django model using `make_django_model()`, it will automatically check if the model matches the current migration state:

```python
from pydantic import BaseModel
from pydantic2django import make_django_model

class User(BaseModel):
    name: str
    age: int

# The second return value contains any needed migration operations
django_model, operations = make_django_model(User)

if operations:
    print("Migration needed! Run makemigrations and migrate.")
    print("Required operations:", operations)
```

You can disable migration checking by passing `check_migrations=False`:

```python
# Skip migration checking
django_model, operations = make_django_model(User, check_migrations=False)
assert operations is None  # No migration check performed
```

This helps catch potential issues early by warning you when your Pydantic models have changed in a way that requires database migrations.

## Documentation

[Documentation Link - Coming Soon]

## Development

This project uses Poetry for dependency management and Ruff for linting.

```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Run linting
poetry run ruff check .
```

## Potential Improvements

### Model Discovery and Introspection
- Automatic discovery of Pydantic models in specified directories
- Bulk conversion of multiple models while preserving relationships
- Support for model inheritance hierarchies
- Migration state validation against existing Django migrations

### Field and Relationship Support
- Generic foreign key support
- Proxy model support
- Custom intermediate model validation for M2M relationships
- Support for more complex field types (JSONField, ArrayField, etc.)
- Enhanced validation of relationship cycles

### Integration Features
- IDE completion improvements for dynamically generated models
- Type hint preservation for better code analysis
- Integration with Django's migration framework
- Automatic migration generation based on Pydantic model changes

### Performance Optimizations
- Caching of generated Django models
- Lazy relationship resolution
- Optimized bulk operations

### Testing and Validation
- Comprehensive test suite across Python and Django versions
- Migration state verification
- Database schema synchronization checks
- Performance benchmarking suite

### Documentation and Examples
- Comprehensive API documentation
- Best practices guide
- Common patterns and use cases
- Migration guides for different scenarios

### CI/CD Improvements
- Automated testing across multiple Python and Django versions
- Coverage reporting and badges
- Automated PyPI deployment
- Documentation generation and deployment

## License

This project is licensed under the MIT License - see the LICENSE file for details.
