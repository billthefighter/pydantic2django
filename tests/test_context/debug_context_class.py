@dataclass
class MockDjangoModelForTypePreservationContext(ModelContext):
    """
    Context class for MockDjangoModelForTypePreservation.
    Contains non-serializable fields that need to be provided when converting from Django to Pydantic.
    """
    model_name: str = "MockDjangoModelForTypePreservation"
    pydantic_class: type = TestModel
    django_model: type[models.Model]
    context_fields: dict[str, FieldContext] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize context fields after instance creation."""
        self.add_field(
            field_name="input_transform",
            field_type="'Optional[Callable[ChainContext, Any, dict[str, Any]]]'",
            is_optional=True,
            is_list=False,
            additional_metadata={}
        )
        self.add_field(
            field_name="output_transform",
            field_type="'Callable[LLMResponse, Any]'",
            is_optional=False,
            is_list=False,
            additional_metadata={}
        )
    @classmethod
    def create(cls, django_model: Type[models.Model],        input_transform: "'Optional[Callable[ChainContext, Any, dict[str, Any]]]'",        output_transform: "'Callable[LLMResponse, Any]'"):
        """
        Create a context instance with the required field values.

        Args:
            django_model: The Django model class
            input_transform: Value for input_transform field
            output_transform: Value for output_transform field
        Returns:
            A context instance with all required field values set
        """
        context = cls(django_model=django_model)
        context.set_value("input_transform", input_transform)
        context.set_value("output_transform", output_transform)
        return context

    def to_dict(self) -> dict:
        """
        Convert this context to a dictionary.

        Returns:
            Dictionary representation of the context values
        """
        result = {}
        for field_name, field_context in self.context_fields.items():
            if field_context.value is not None:
                result[field_name] = field_context.value
        return result
