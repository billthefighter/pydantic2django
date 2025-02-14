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

## License

This project is licensed under the MIT License - see the LICENSE file for details. 