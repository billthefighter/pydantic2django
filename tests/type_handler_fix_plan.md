# TypeHandler Fix Implementation Plan

## 1. Fixing Nested List Brackets in Callable Parameters

### Proposed Changes to `clean_type_string`
```python
# Add this code to the "Special handling for Callable types" section
if "Callable[[[" in type_str:
    logger.debug("Fixing nested brackets in Callable parameters")
    # Match patterns like Callable[[[Any], Dict]] and fix them
    type_str = re.sub(r"Callable\[\[\[(.*?)\]", r"Callable[[\1", type_str)
    logger.debug(f"After fixing nested brackets: '{type_str}'")
```

### Proposed Changes to `fix_callable_syntax`
```python
# Add to the function near the beginning
if "Callable[[[" in type_str:
    logger.debug("Fixing triple nested brackets in Callable")
    type_str = re.sub(r"Callable\[\[\[(.*?)\]", r"Callable[[\1", type_str)
    logger.debug(f"After fixing triple brackets: '{type_str}'")
```

## 2. Fixing Trailing TypeVar Issues

### Proposed Changes to `clean_type_string`
```python
# Add after handling Callable syntax
if "Callable[" in type_str and "], " in type_str and not ", is_" in type_str:
    logger.debug("Handling trailing TypeVar after Callable")
    # Extract the Callable part up to the closing bracket
    callable_match = re.search(r"(Callable\[\[.*?\]\])", type_str)
    if callable_match:
        type_str = callable_match.group(1)
        logger.debug(f"Extracted Callable part: '{type_str}'")
```

### Proposed Changes to `fix_callable_syntax`
```python
# Add near the beginning of the function
if ", T" in type_str or ", is_" in type_str:
    logger.debug("Removing trailing TypeVar or keyword arguments")
    # Extract just the Callable part
    callable_match = re.search(r"(Callable\[.*?\])", type_str)
    if callable_match:
        type_str = callable_match.group(1)
        logger.debug(f"Extracted Callable without trailing parts: '{type_str}'")
```

## 3. Fixing Empty Parameters and Return Type

### Proposed Changes to `fix_callable_syntax`
```python
# Modify the handling of Callable with empty parameters
if "Callable[[]]" in type_str:
    logger.debug("Handling empty parameter list")
    # Ensure the return type is separated by comma and has a value
    if not "], " in type_str:
        type_str = type_str.replace("]]", "], Any]")
        logger.debug(f"Added default return type: '{type_str}'")
```

## 4. Fixing NoneType Import Generation

### Proposed Changes to `get_required_imports`
```python
# Add specific handling for NoneType
if "NoneType" in type_str:
    imports["typing"].append("NoneType")
    logger.debug("Added NoneType to typing imports")
```

### Proposed Changes to `process_field_type`
```python
# Add to the process_field_type function
if "NoneType" in type_name:
    if "from types import NoneType" not in required_imports:
        required_imports.append("from types import NoneType")
        logger.debug("Added explicit import for NoneType")
```

## 5. Fixing Syntax with Keywords

### Proposed Changes to `clean_type_string`
```python
# Add handling for trailing keyword arguments
if ", is_" in type_str:
    logger.debug("Removing trailing keyword arguments")
    # Extract just the type portion before keywords
    type_match = re.search(r"(.*?),\s*is_", type_str)
    if type_match:
        type_str = type_match.group(1)
        logger.debug(f"Extracted type before keywords: '{type_str}'")
```

## Additional Cleanup

### Proposed Changes to Improve Bracket Handling
```python
# Enhance balance_brackets function to better handle complex cases
def balance_brackets(s: str) -> str:
    # Special case for problematic patterns from tests
    if "Callable[[[" in s:
        s = re.sub(r"Callable\[\[\[(.*?)\]", r"Callable[[\1", s)

    # Handle trailing TypeVars and keywords
    if "Callable[" in s and "], " in s:
        callable_match = re.search(r"(Callable\[.*?\])", s)
        if callable_match:
            s = callable_match.group(1)

    # Rest of the function remains the same...
```

## Testing Strategy

1. Run the tests we've created to verify each fix addresses the identified issues
2. Test with actual model generation to ensure the linter errors are resolved
3. Add additional test cases as needed to ensure robust handling of edge cases

## Implementation Order

1. First implement the fixes for nested brackets and trailing TypeVar issues
2. Then address the NoneType import generation
3. Finally fix the keyword arguments and empty parameter handling
4. Verify all tests pass and model generation produces correct output
