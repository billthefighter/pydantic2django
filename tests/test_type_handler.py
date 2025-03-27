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
                    input_type="Callable[[]]", expected_output="Callable[[]]", test_id="empty-params-no-return"
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
        result, _ = TypeHandler.process_field_type(input_type)
        assert result == "Callable[[Any], Dict]", "Failed to remove nested list brackets"

    def test_list_as_callable_parameter(self):
        """Test handling list as a Callable parameter."""
        input_type = "Callable[[[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]"
        result, _ = TypeHandler.process_field_type(input_type)
        assert (
            result == "Callable[[Dict[str, Any]], Optional[List[Dict[str, Any]]]]"
        ), "Failed to clean nested list in Callable"

    def test_trailing_type_variable(self):
        """Test removal of trailing type variable in Callable."""
        input_type = "Callable[[], LLMResponse], T"
        result, _ = TypeHandler.process_field_type(input_type)
        assert result == "Callable[[], LLMResponse]", "Failed to remove trailing type variable"

    def test_specific_pattern_from_line_122(self):
        """Test the specific pattern from line 122 in generated_models.py."""
        input_type = "Callable[[], LLMResponse], T, is_optional=False"
        result, _ = TypeHandler.process_field_type(input_type)
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

    def test_nonetype_definition(self):
        """Test NoneType is properly handled in type strings."""
        input_type = "Union[Callable, NoneType]"
        result, imports = TypeHandler.process_field_type(input_type)
        assert result == "Union[Callable, NoneType]", "Failed to retain NoneType"
        assert any("NoneType" in imp for imp in imports), "Failed to include NoneType in imports"

    def test_nonetype_import_handling(self):
        """Test NoneType is properly included in required imports."""
        input_type = "Optional[Union[Callable, NoneType]]"
        required_imports = TypeHandler.get_required_imports(input_type)
        assert "NoneType" in required_imports["typing"], "Failed to include NoneType in required imports"

    def test_callable_with_type_var_and_keyword_args(self):
        """Test Callable with TypeVar and keyword arguments."""
        input_type = "Callable[[], LLMResponse], T, is_optional=False, additional_metadata={}"
        result, _ = TypeHandler.process_field_type(input_type)
        assert result == "Callable[[], LLMResponse]", "Failed to clean type string with keyword args"

    def test_balance_brackets_for_complex_callable(self):
        """Test balancing brackets for complex Callable types."""
        input_type = "Callable[[Dict[str, Any]], Optional[List[Dict[str, Any]]]]"
        result = TypeHandler.balance_brackets(input_type)
        assert result == input_type, "Failed to correctly balance brackets"

    def test_callable_with_list_args_and_dict_return(self):
        """Test Callable with list arguments and Dict return type."""
        input_type = "Callable[[[Any], Dict]]"
        result, _ = TypeHandler.process_field_type(input_type)
        assert result == "Callable[[Any], Dict]", "Failed to clean nested list brackets in parameters"

    def test_specific_line_115_error(self):
        """Test for specific error on line 115 in generated_models.py."""
        input_type = "Callable[[[Any], Dict]]"
        result, _ = TypeHandler.process_field_type(input_type)
        assert "[[[" not in result, "Failed to remove triple nested brackets"
        assert result == "Callable[[Any], Dict]", "Failed to format Callable parameters correctly"

    def test_specific_line_122_error(self):
        """Test for specific error on line 122 in generated_models.py."""
        input_type = "Callable[[], LLMResponse], T"
        result, _ = TypeHandler.process_field_type(input_type)
        assert ", T" not in result, "Failed to remove trailing type variable"

    def test_specific_line_138_error(self):
        """Test for specific error on line 138 in generated_models.py."""
        input_type = "Callable[[[Any], Dict]]"
        result, _ = TypeHandler.process_field_type(input_type)
        assert result == "Callable[[Any], Dict]", "Failed to clean list expression in type annotation"


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


class TestTypeHandlerNoneType:
    """Test handling of NoneType in TypeHandler to address linter errors."""

    def test_union_callable_nonetype(self):
        """Test handling Union[Callable, NoneType] pattern seen in lines 51, 58."""
        input_type = "Union[Callable, NoneType]"
        result, imports = TypeHandler.process_field_type(input_type)
        assert result == "Union[Callable, NoneType]", "Failed to preserve NoneType in Union"
        assert any("NoneType" in imp for imp in imports), "Failed to include NoneType in import list"

    def test_optional_union_callable_nonetype(self):
        """Test handling Optional[Union[Callable, NoneType]] pattern."""
        input_type = "Optional[Union[Callable, NoneType]]"
        result, imports = TypeHandler.process_field_type(input_type)
        assert result == "Optional[Union[Callable, NoneType]]", "Failed to preserve Optional with NoneType"
        assert any("NoneType" in imp for imp in imports), "Failed to include NoneType in import list"

    def test_optional_callable_parameter_with_nonetype(self):
        """Test handling Optional parameter that has NoneType (lines 73, 74)."""
        input_type = "Optional[Union[Callable, NoneType]]"
        required_imports = TypeHandler.get_required_imports(input_type)
        assert "NoneType" in required_imports["typing"], "Failed to include NoneType in typing imports"
        assert "Optional" in required_imports["typing"], "Failed to include Optional in typing imports"
        assert "Union" in required_imports["typing"], "Failed to include Union in typing imports"


class TestGeneratedModelsLinterErrors:
    """
    Tests specifically targeting the linter errors in generated_models.py
    Each test focuses on a specific line number where errors occurred.
    """

    def test_line_115_error_nested_list(self):
        """
        Test for error on line 115: field_type=Callable[[[Any], Dict]]]
        Expected type expression but received "list[Any]"
        Expected return type as second type argument for "Callable"
        """
        input_type = "Callable[[[Any], Dict]]"
        result, imports = TypeHandler.process_field_type(input_type)
        # Verify no triple brackets
        assert "[[[" not in result, "Triple brackets still present in output"
        # Verify correct formatting
        assert result == "Callable[[Any], Dict]", "Failed to correctly format nested list in Callable"
        # Verify imports are correct
        assert "Callable" in imports[0] or any("Callable" in imp for imp in imports), "Callable import missing"

    def test_line_122_error_trailing_type_var(self):
        """
        Test for error on line 122: field_type=Callable[[], LLMResponse], T
        Positional argument cannot appear after keyword arguments
        Expected 2 positional arguments
        """
        input_type = "Callable[[], LLMResponse], T, is_optional=False"
        result, imports = TypeHandler.process_field_type(input_type)
        # Verify no trailing type var
        assert ", T" not in result, "Trailing type variable not removed"
        # Verify no keyword arguments in type string
        assert "is_optional" not in result, "Keyword arguments not removed from type string"
        # Verify correct formatting
        assert result == "Callable[[], LLMResponse]", "Failed to correctly format Callable with trailing type var"

    def test_line_138_error_nested_list_type(self):
        """
        Test for error on line 138: input_transform: Callable[[[Any], Dict]]
        List expression not allowed in type annotation
        Expected type expression but received "list[Any]"
        """
        input_type = "Callable[[[Any], Dict]]"
        result, _ = TypeHandler.process_field_type(input_type)
        assert result == "Callable[[Any], Dict]", "Failed to clean list expression in type annotation"
        # Verify brackets structure
        assert result.count("[") == 3, "Incorrect number of opening brackets"
        assert result.count("]") == 3, "Incorrect number of closing brackets"

    def test_line_139_error_trailing_comma_in_params(self):
        """
        Test for error on line 139: output_transform: Callable[[], LLMResponse], T
        SyntaxError: positional argument follows keyword argument
        """
        input_type = "Callable[[], LLMResponse], T"
        clean_type_result = TypeHandler.clean_type_string(input_type)
        fix_callable_result = TypeHandler.fix_callable_syntax(input_type)
        process_result, _ = TypeHandler.process_field_type(input_type)

        # All methods should handle this correctly
        assert ", T" not in clean_type_result, "clean_type_string failed to remove trailing T"
        assert ", T" not in fix_callable_result, "fix_callable_syntax failed to remove trailing T"
        assert ", T" not in process_result, "process_field_type failed to remove trailing T"

        # Final result should be cleaned properly
        assert (
            process_result == "Callable[[], LLMResponse]"
        ), "Failed to properly format Callable with trailing type var"


class TestTypeHandlerImportCategorization:
    """Test TypeHandler's ability to correctly categorize imports from different sources."""

    @pytest.mark.parametrize(
        "params",
        [
            pytest.param(
                TypeHandlerTestParams(
                    input_type="llmaestro.chains.chains.ChainNode",
                    expected_output={
                        "typing": [],
                        "custom": ["ChainNode"],
                        "explicit": ["from llmaestro.chains.chains import ChainNode"],
                    },
                    test_id="fully-qualified-module-path",
                ),
                id="fully-qualified-module-path",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="typing.Dict[str, llmaestro.chains.chains.ChainNode]",
                    expected_output={
                        "typing": ["Dict"],
                        "custom": ["ChainNode"],
                        "explicit": ["import typing", "from llmaestro.chains.chains import ChainNode"],
                    },
                    test_id="typing-with-module-path",
                ),
                id="typing-with-module-path",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="typing.Optional[typing.Dict[str, llmaestro.core.conversations.ConversationNode]]",
                    expected_output={
                        "typing": ["Optional", "Dict"],
                        "custom": ["ConversationNode"],
                        "explicit": ["import typing", "from llmaestro.core.conversations import ConversationNode"],
                    },
                    test_id="complex-typing-with-module-path",
                ),
                id="complex-typing-with-module-path",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="<class 'llmaestro.chains.chains.ChainContext'>",
                    expected_output={
                        "typing": [],
                        "custom": ["ChainContext"],
                        "explicit": ["from llmaestro.chains.chains import ChainContext"],
                    },
                    test_id="angle-bracket-class-string",
                ),
                id="angle-bracket-class-string",
            ),
        ],
    )
    def test_get_required_imports_for_module_paths(self, params: TypeHandlerTestParams):
        """Test that TypeHandler correctly identifies imports from fully qualified module paths."""
        # Get imports from the type string
        result = TypeHandler.get_required_imports(params.input_type)

        # Sort lists for consistent comparison
        for k in result:
            if isinstance(result[k], list):
                result[k] = sorted(result[k])

        expected = params.expected_output
        for k in expected:
            if isinstance(expected[k], list):
                expected[k] = sorted(expected[k])

        # Check each category of imports
        assert set(result["typing"]) == set(
            expected["typing"]
        ), f"Typing imports don't match: {result['typing']} != {expected['typing']}"
        assert set(result["custom"]) == set(
            expected["custom"]
        ), f"Custom type imports don't match: {result['custom']} != {expected['custom']}"

        # Explicit imports may contain module paths that need special handling
        for exp_import in expected["explicit"]:
            matching_imports = [
                imp for imp in result["explicit"] if self._normalize_import(imp) == self._normalize_import(exp_import)
            ]
            assert matching_imports, f"Expected import '{exp_import}' not found in {result['explicit']}"

    def _normalize_import(self, import_stmt: str) -> str:
        """Normalize import statements to handle slight format differences."""
        if import_stmt.startswith("from ") and " import " in import_stmt:
            parts = import_stmt.split(" import ")
            module = parts[0].replace("from ", "")
            imports = parts[1].split(", ")
            return f"from {module} import {', '.join(sorted(imports))}"
        return import_stmt


class TestTypeHandlerProcessFieldTypeWithComplexObjects:
    """Test the TypeHandler's process_field_type method with complex object references."""

    @pytest.mark.parametrize(
        "params",
        [
            pytest.param(
                TypeHandlerTestParams(
                    input_type="<class 'llmaestro.chains.chains.ChainNode'>",
                    expected_output=("ChainNode", ["from llmaestro.chains.chains import ChainNode"]),
                    test_id="class-object-angle-brackets",
                ),
                id="class-object-angle-brackets",
            ),
            pytest.param(
                TypeHandlerTestParams(
                    input_type="typing.Dict[str, typing.List[str]]",
                    expected_output=("Dict[str, List[str]]", ["import typing", "from typing import Dict, List"]),
                    test_id="nested-typing-with-module",
                ),
                id="nested-typing-with-module",
            ),
        ],
    )
    def test_process_field_type_with_angle_brackets(self, params: TypeHandlerTestParams):
        """Test that process_field_type correctly handles class references with angle brackets."""
        type_name, imports = TypeHandler.process_field_type(params.input_type)

        # Check type name
        assert (
            type_name == params.expected_output[0]
        ), f"Type name doesn't match: {type_name} != {params.expected_output[0]}"

        # Check imports (some flexibility in format allowed)
        expected_imports = sorted(params.expected_output[1])
        actual_imports = sorted(imports)

        for exp_import in expected_imports:
            found = False
            for act_import in actual_imports:
                # Normalize imports to handle format differences
                if self._normalize_import(exp_import) == self._normalize_import(act_import):
                    found = True
                    break
            assert found, f"Expected import '{exp_import}' not found in {actual_imports}"

    def _normalize_import(self, import_stmt: str) -> str:
        """Normalize import statements to handle slight format differences."""
        if import_stmt.startswith("from ") and " import " in import_stmt:
            parts = import_stmt.split(" import ")
            module = parts[0].replace("from ", "")
            imports = parts[1].split(", ")
            return f"from {module} import {', '.join(sorted(imports))}"
        return import_stmt


if __name__ == "__main__":
    pytest.main(["-v", "test_type_handler.py"])
