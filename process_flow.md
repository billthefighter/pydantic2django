# Pydantic2Django Static Model Generation Process Flow

## 1. Initialization
- Entry point: `StaticDjangoModelGenerator` class in `static_django_model_generator.py`
- Configures output path, packages to scan, app label, and logging
- Sets up Jinja2 environment for template rendering

## 2. Pydantic Model Discovery
### Discovery Process (`discover_models()`)
- Uses `ModelDiscovery` class from `discovery.py`
- Scans specified packages for Pydantic models
- Process:
  1. Imports target packages using `importlib`
  2. Walks through Python files in package directories
  3. Identifies classes that inherit from `BaseModel`
  4. Normalizes model names (adds "Django" prefix if needed)
  5. Stores discovered models in registry

### Model Storage
- Maintains dictionaries in `ModelDiscovery`:
  - `discovered_models`: Raw Pydantic models
  - `normalized_models`: Models with normalized names
  - `dependencies`: Model relationship dependencies
  - `django_models`: Generated Django models

## 3. Field Parsing and Type Resolution
### Field Type Resolution (`FieldTypeResolver` in `field_type_resolver.py`)
- Determines Django field type for each Pydantic field
- Handles:
  - Basic Python types (str, int, float, bool)
  - Optional/Union types
  - Complex types (dict, list)
  - Custom serializable types
  - Non-serializable types (converted to TextField)

### Field Attribute Handling (`FieldAttributeHandler` in `field_utils.py`)
- Processes field attributes and metadata
- Handles:
  - Null/blank settings
  - Default values
  - Help text and verbose names
  - Field constraints (validators)
  - Custom attributes

## 4. Relationship Field Identification
### Relationship Detection (`RelationshipFieldHandler` in `field_utils.py`)
- Identifies model relationships:
  - Direct Pydantic model references → ForeignKey
  - List of Pydantic models → ManyToManyField
  - Non-serializable types → TextField with is_relationship=True
- Handles relationship attributes:
  - on_delete behavior
  - related_name
  - through models for M2M relationships

### Dependency Analysis
- Analyzes model dependencies for proper ordering
- Creates dependency graph for model registration
- Ensures related models are created in correct order

## 5. Django Model Creation
### Model Generation (`setup_django_models()`)
- Uses `DjangoModelFactory` to create Django models
- Process for each model:
  1. Creates model class inheriting from Django's Model
  2. Adds fields with proper types and attributes
  3. Sets up Meta class with table name and app label
  4. Registers model with Django's app registry

### Field Creation
- Creates Django field instances
- Applies field attributes and validators
- Sets up relationship fields with proper references

## 6. Static File Generation
### Template Rendering
- Uses Jinja2 templates for model definitions
- Generates:
  - Import statements
  - Model class definitions
  - Field definitions
  - Meta class configurations
  - Model registration code

### File Writing
- Writes generated code to specified output path
- Includes:
  - Timestamp and generation metadata
  - Complete model definitions
  - All required imports
  - Model registration code

## 7. Context Storage
### State Management
- Maintains discovery context in `ModelDiscovery` singleton
- Stores:
  - Discovered models
  - Generated Django models
  - Model dependencies
  - Registration order

### Persistence
- Generates static Python file with all models
- Preserves model relationships and dependencies
- Maintains proper import order for dependencies

## Key Files and Classes
- `static_django_model_generator.py`: Main generator class
- `discovery.py`: Model discovery and registration
- `field_type_resolver.py`: Field type resolution
- `field_utils.py`: Field attribute handling
- `fields.py`: Field conversion and creation
- `templates/`: Jinja2 templates for code generation
