"""
Tests for core functionality.
"""
import pytest
from django.db import models
from pydantic import BaseModel

from pydantic2django import make_django_model

def test_make_django_model_basic():
    """Test basic model conversion."""
    class UserModel(BaseModel):
        name: str
        age: int
        email: str

    DjangoUserModel = make_django_model(UserModel)
    
    # Check that it's a proper Django model
    assert issubclass(DjangoUserModel, models.Model)
    
    # Check that fields were created
    fields = DjangoUserModel._meta.get_fields()
    field_names = {f.name for f in fields}
    
    assert "name" in field_names
    assert "age" in field_names
    assert "email" in field_names

def test_make_django_model_with_methods():
    """Test that methods from Pydantic model are preserved."""
    class UserModel(BaseModel):
        name: str
        age: int
        
        def get_display_name(self) -> str:
            return f"{self.name} ({self.age})"
    
    DjangoUserModel = make_django_model(UserModel)
    
    # Check that the method was preserved
    assert hasattr(DjangoUserModel, "get_display_name")
    
    # Create an instance and test the method
    user = DjangoUserModel(name="John", age=30)
    assert user.get_display_name() == "John (30)" 