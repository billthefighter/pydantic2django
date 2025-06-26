"""
Defines various generic Python classes to be used as source models for typedclass conversion tests.
"""

from typing import Optional, List, Dict, Any
import datetime
import uuid

# --- Simple Classes ---
class SimpleTypedClass:
    """A very basic class with class attributes and type hints."""
    name: str = "Default Name"
    count: int = 0
    is_active: bool = True
    unique_id: uuid.UUID = field(default_factory=uuid.uuid4) # Requires import dataclasses.field if used as default factory
    # For now, let's assume direct assignment or __init__ for simplicity in TypedClass
    # unique_id: uuid.UUID = uuid.uuid4() # This would be a class-level constant UUID

    def __init__(self, name: Optional[str] = None, count: Optional[int] = None):
        if name is not None:
            self.name = name
        if count is not None:
            self.count = count
        self.runtime_id = uuid.uuid4() # Will not be picked up unless __init__ has it or class annotation

class AnotherSimpleClass:
    description: Optional[str] = None
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)

    def __init__(self, description: Optional[str] = None):
        self.description = description
        if not hasattr(self, 'created_at'): # Ensure created_at is set if not by default_factory magic
             self.created_at = datetime.datetime.now()

# --- Classes with __init__ focus ---
class InitOnlyTypedClass:
    """Attributes are primarily defined by __init__ parameters."""
    def __init__(self, item_id: int, value: float, notes: Optional[str] = "No notes"):
        self.item_id = item_id
        self.value = value
        self.notes = notes
        self._internal_state = "secret" # Should be ignored by default

# --- Classes with Dependencies ---
class ChildTypedClass:
    child_name: str
    child_id: int

    def __init__(self, child_name: str, child_id: int):
        self.child_name = child_name
        self.child_id = child_id

class ParentTypedClass:
    parent_name: str
    child_instance: ChildTypedClass # Direct dependency
    optional_child: Optional[ChildTypedClass] = None
    # list_of_children: List[ChildTypedClass] = field(default_factory=list) # More complex

    def __init__(self, parent_name: str, child: ChildTypedClass, opt_child: Optional[ChildTypedClass] = None):
        self.parent_name = parent_name
        self.child_instance = child
        self.optional_child = opt_child

# --- Classes simulating pydantic-ai structures (simplified) ---
class MockAIClient:
    """Represents a complex, non-serializable client object."""
    def __init__(self, api_key: str, timeout: int = 60):
        self.api_key = api_key # This might be persisted
        self.timeout = timeout   # This might be persisted
        self._session = object() # This should NOT be persisted

    def connect(self):
        return f"Connected with key {self.api_key}"

class MockAIProviderConfig:
    provider_name: str
    retry_attempts: int
    # client: MockAIClient # Attribute that should be handled carefully (e.g. skipped or special logic)

    def __init__(self, provider_name: str, retry_attempts: int = 3, api_key_for_client: Optional[str] = None):
        self.provider_name = provider_name
        self.retry_attempts = retry_attempts
        if api_key_for_client:
            self.client = MockAIClient(api_key=api_key_for_client) # Instance of complex obj
        else:
            self.client = None # Or some default / placeholder

class MockAIModelSettings:
    model_identifier: str
    temperature: float = 0.7
    # config: MockAIProviderConfig # Nested typed class

    def __init__(self, model_id: str, temp: float = 0.7, provider_config: Optional[MockAIProviderConfig] = None):
        self.model_identifier = model_id
        self.temperature = temp
        self.config = provider_config

# Helper for dataclasses.field, not directly used by TypedClass but good for reference
# from dataclasses import field
