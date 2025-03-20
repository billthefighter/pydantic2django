import pytest
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, Generic

from pydantic2django.type_handler import TypeHandler

# Set up test logger
logger = logging.getLogger(__name__)


# Test data structure for parameterized tests
@dataclass
class TypeHandlerTestParams:
    """Test parameters for TypeHandler tests."""

    input_type: Any
    expected_output: Any  # Can be str or dict
    test_id: str
    description: str = ""


class TestTypeHandlerCleanTypeString:
    """Test the clean_type_string method of TypeHandler."""

    @pytest.mark.parametrize(
        "params",
        [
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Callable[[Dict[str, Any]], Optional[List[Dict[str, Any]]]]",
                    expected_output="Callable[[[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]",
                    test_id="complex-callable-with-nested-types",
                ),
                id="complex-callable-with-nested-types",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Callable[[[Any], Dict]]",
                    expected_output="Callable[[[[Any], Dict]]]",
                    test_id="nested-lists-in-callable-parameters",
                ),
                id="nested-lists-in-callable-parameters",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Callable[[[], Dict]]",
                    expected_output="Callable[[[[], Dict]]]",
                    test_id="double-nested-empty-list",
                ),
                id="double-nested-empty-list",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Optional[Union[Callable, None]]",
                    expected_output="Optional[Union[Callable, None]]",
                    test_id="optional-union-with-none",
                ),
                id="optional-union-with-none",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Callable[[], LLMResponse], T",
                    expected_output="Callable[[LLMResponse], T]",
                    test_id="callable-with-trailing-typevar",
                ),
                id="callable-with-trailing-typevar",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Callable[[], Dict], is_optional=False",
                    expected_output="Callable[[Dict], is_optional=False]",
                    test_id="callable-with-keyword-args",
                ),
                id="callable-with-keyword-args",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Union[Callable, None]",
                    expected_output="Union[Callable, None]",
                    test_id="union-with-none",
                ),
                id="union-with-none",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Type[PromptType]", expected_output="Type[PromptType]", test_id="type-with-typevar"
                ),
                id="type-with-typevar",
            ),
        ],
    )
    def test_clean_type_string(self, params: TypeHandlerTestParams):
        """Test cleaning and formatting type strings."""
        result = TypeHandler.clean_type_string(params.input_type)
        assert result == params.expected_output, f"Failed to clean type string for {params.test_id}"


class TestTypeHandlerFixCallableSyntax:
    """Test the fix_callable_syntax method of TypeHandler."""

    @pytest.mark.parametrize(
        "params",
        [
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Callable[[[Any], Dict]]",
                    expected_output="Callable[[[Any], Dict]]",
                    test_id="process-callable-nested-brackets",
                ),
                id="process-callable-nested-brackets",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Callable[[]]", expected_output="Callable[[], Any]", test_id="empty-params-no-return"
                ),
                id="empty-params-no-return",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Callable[[], LLMResponse], T",
                    expected_output="Callable[[], LLMResponse]",
                    test_id="trailing-typevar",
                ),
                id="trailing-typevar",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Callable[[Dict[str, Any]]]",
                    expected_output="Callable[[Dict[str, Any]], Any]",
                    test_id="missing-return-type",
                ),
                id="missing-return-type",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Callable[Dict]",
                    expected_output="Callable[[Dict], Any]",
                    test_id="missing-brackets-in-params",
                ),
                id="missing-brackets-in-params",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Callable[Any], Dict]",
                    expected_output="Callable[[Any], Dict]",
                    test_id="incorrect-bracket-placement",
                ),
                id="incorrect-bracket-placement",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Callable[[], LLMResponse]], T]",
                    expected_output="Callable[[], LLMResponse]",
                    test_id="trailing-type-variable-with-brackets",
                ),
                id="trailing-type-variable-with-brackets",
            ),
        ],
    )
    def test_fix_callable_syntax(self, params: TypeHandlerTestParams):
        """Test fixing various Callable syntax patterns."""
        result = TypeHandler.fix_callable_syntax(params.input_type)
        assert result == params.expected_output, f"Failed to fix Callable syntax for {params.test_id}"


class TestTypeHandlerProcessFieldType:
    """Test the process_field_type method of TypeHandler."""

    @pytest.mark.parametrize(
        "params",
        [
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Callable[[dict], Dict[str, Any]]",
                    expected_output="Callable[[dict], Dict[str, Any]]",
                    test_id="process-callable",
                ),
                id="process-callable",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type=Callable[[dict], Dict[str, Any]],
                    expected_output="Callable[[dict], Dict[str, Any]]",
                    test_id="process-actual-callable-type",
                ),
                id="process-actual-callable-type",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type=Optional[Union[Callable, None]],
                    expected_output="Optional[Union[Callable, None]]",
                    test_id="process-optional-union-callable",
                ),
                id="process-optional-union-callable",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type=Optional[Callable[[Any], Dict[str, Any]]],
                    expected_output="Optional[Callable[[Any], Dict[str, Any]]]",
                    test_id="process-optional-callable-with-args",
                ),
                id="process-optional-callable-with-args",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type=Union[Callable, None],
                    expected_output="Union[Callable, None]",
                    test_id="process-union-with-none",
                ),
                id="process-union-with-none",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Callable[[[Any], Dict]]",
                    expected_output="Callable[[Any], Dict]",
                    test_id="process-nested-list-in-callable",
                ),
                id="process-nested-list-in-callable",
            ),
        ],
    )
    def test_process_field_type(self, params: TypeHandlerTestParams):
        """Test processing field type to produce clean type string."""
        result, _ = TypeHandler.process_field_type(params.input_type)
        assert result == params.expected_output, f"Failed to process field type for {params.test_id}"


class TestTypeHandlerRequiredImports:
    """Test the get_required_imports method of TypeHandler."""

    @pytest.mark.parametrize(
        "params",
        [
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Optional[Union[Callable, None]]",
                    expected_output={"typing": ["Optional", "Union", "Callable"], "custom": [], "explicit": []},
                    test_id="imports-for-optional-union",
                ),
                id="imports-for-optional-union",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Callable[[ChainContext, Any], Dict[str, Any]]",
                    expected_output={"typing": ["Callable"], "custom": ["ChainContext"], "explicit": []},
                    test_id="imports-for-callable-with-custom-type",
                ),
                id="imports-for-callable-with-custom-type",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Type[PromptType]",
                    expected_output={"typing": ["Type"], "custom": ["PromptType"], "explicit": []},
                    test_id="imports-for-type-with-custom-type",
                ),
                id="imports-for-type-with-custom-type",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Union[Callable, None]",
                    expected_output={"typing": ["Union", "Callable"], "custom": [], "explicit": []},
                    test_id="imports-for-union-with-none",
                ),
                id="imports-for-union-with-none",
            ),
        ],
    )
    def test_get_required_imports(self, params: TypeHandlerTestParams):
        """Test extracting required imports from type strings."""
        result = TypeHandler.get_required_imports(params.input_type)
        assert result == params.expected_output, f"Failed to get required imports for {params.test_id}"


class TestTypeHandlerSpecificIssues:
    """Test specific issues identified in generated code."""

    def test_nested_lists_in_callable_parameters(self):
        """Test handling of nested list brackets in Callable parameters."""
        input_type = "Callable[[[Any], Dict]]"
        result = TypeHandler.clean_type_string(input_type)
        assert result == "Callable[[Any], Dict]", "Failed to remove nested list brackets"

    def test_list_as_callable_parameter(self):
        """Test handling list as a Callable parameter."""
        input_type = "Callable[[[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]"
        result = TypeHandler.clean_type_string(input_type)
        assert (
            result == "Callable[[Dict[str, Any]], Optional[List[Dict[str, Any]]]]"
        ), "Failed to clean nested list in Callable"

    def test_trailing_type_variable(self):
        """Test removal of trailing type variable in Callable."""
        input_type = "Callable[[], LLMResponse], T"
        result = TypeHandler.clean_type_string(input_type)
        assert result == "Callable[[], LLMResponse]", "Failed to remove trailing type variable"

    def test_specific_pattern_from_line_122(self):
        """Test the specific pattern from line 122 in generated_models.py."""
        input_type = "Callable[[], LLMResponse], T, is_optional=False"
        result = TypeHandler.clean_type_string(input_type)
        assert result == "Callable[[], LLMResponse]", "Failed to clean Callable with type var and keyword args"

    def test_callable_with_empty_list_parameter(self):
        """Test Callable with empty list parameter."""
        input_type = "Callable[[]]"
        result = TypeHandler.fix_callable_syntax(input_type)
        assert result == "Callable[[], Any]", "Failed to fix empty parameter list"

    def test_expected_return_type_for_callable(self):
        """Test ensuring proper return type for Callable."""
        input_type = "Callable[[Dict]], Any]"
        result = TypeHandler.fix_callable_syntax(input_type)
        assert result == "Callable[[Dict], Any]", "Failed to fix brackets in return type"

    def test_none_vs_nonetype_handling(self):
        """Test None is properly handled in type strings."""
        input_type = "Union[Callable, None]"
        result = TypeHandler.clean_type_string(input_type)
        assert result == "Union[Callable, None]", "Failed to retain None"

    def test_none_import_handling(self):
        """Test None is properly included in required imports."""
        input_type = "Optional[Union[Callable, None]]"
        required_imports = TypeHandler.get_required_imports(input_type)
        assert "Optional" in required_imports["typing"], "Failed to include Optional in required imports"
        assert "Union" in required_imports["typing"], "Failed to include Union in required imports"

    def test_callable_with_type_var_and_keyword_args(self):
        """Test Callable with TypeVar and keyword arguments."""
        input_type = "Callable[[], LLMResponse], T, is_optional=False, additional_metadata={}"
        result = TypeHandler.clean_type_string(input_type)
        assert result == "Callable[[], LLMResponse]", "Failed to clean type string with keyword args"

    def test_balance_brackets_for_complex_callable(self):
        """Test balancing brackets for complex Callable types."""
        input_type = "Callable[[Dict[str, Any]], Optional[List[Dict[str, Any]]]]"
        result = TypeHandler.balance_brackets(input_type)
        assert result == input_type, "Failed to correctly balance brackets"


class TestTypeHandlerBalanceBrackets:
    """Test the balance_brackets method of TypeHandler."""

    @pytest.mark.parametrize(
        "params",
        [
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Callable[[Dict[str, Any]], Optional[List[Dict[str, Any]]]]",
                    expected_output="Callable[[Dict[str, Any]], Optional[List[Dict[str, Any]]]]",
                    test_id="already-balanced-complex",
                ),
                id="already-balanced-complex",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Callable[[Dict[str, Any]]",
                    expected_output="Callable[[Dict[str, Any]], Any]",
                    test_id="missing-closing-brackets",
                ),
                id="missing-closing-brackets",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Callable[[Dict[str, Any]]]]]",
                    expected_output="Callable[[Dict[str, Any]], Any]",
                    test_id="excess-closing-brackets",
                ),
                id="excess-closing-brackets",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Callable[[Dict[str, Any]",
                    expected_output="Callable[[Dict[str, Any]], Any]",
                    test_id="severely-unbalanced",
                ),
                id="severely-unbalanced",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="Callable[[], Dict[str, Any]], T]",
                    expected_output="Callable[[], Dict[str, Any]]",
                    test_id="trailing-typevar-with-brackets",
                ),
                id="trailing-typevar-with-brackets",
            ),
        ],
    )
    def test_balance_brackets(self, params: TypeHandlerTestParams):
        """Test balancing brackets in type strings."""
        result = TypeHandler.balance_brackets(params.input_type)
        assert result == params.expected_output, f"Failed to balance brackets for {params.test_id}"


if __name__ == "__main__":
    pytest.main(["-v", "test_type_handler.py"])
