# SQL Node Result Truncation

## Overview

The SQL Node now includes automatic result truncation to prevent token exhaustion when processing large query results. This feature uses a function wrapper pattern that intercepts and truncates large results before they reach the chat context.

## Features

### Automatic Truncation
- **Row-based limiting**: Truncates results exceeding a configurable row count (default: 50 rows)
- **Character-based limiting**: Truncates results exceeding a configurable character count (default: 5000 chars)
- **Transparent operation**: Works seamlessly with Gemini's automatic function calling
- **No configuration conflicts**: Uses standard SDK features without conflicts

### Intelligent Handling
- **Small queries**: Results under the threshold pass through unchanged
- **Large queries**: Automatically truncated with helpful metadata and instructions
- **Metadata preservation**: Column names and structure are maintained
- **Guidance included**: Truncated results include instructions for the model to use subqueries

## Configuration

### Environment Variables

You can configure the truncation thresholds using environment variables:

```bash
# Maximum number of rows to return (default: 50)
export SQL_NODE_MAX_ROWS=50

# Maximum character count for results (default: 5000)
export SQL_NODE_MAX_CHARS=5000
```

### Programmatic Configuration

You can also configure the thresholds when creating the SQLNode:

```python
from app.mirkat.node_sql import SQLNode

sql_node = SQLNode(
    llm="gemini-2.5-flash",
    instructions="Your SQL instructions...",
    functions=db_tools,
    max_result_rows=100,      # Custom row limit
    max_result_chars=10000    # Custom character limit
)
```

## How It Works

### 1. Function Wrapping

When the SQLNode is initialized, it wraps the `execute_query` function with truncation logic:

```python
# Original function is wrapped
wrapped_execute_query = _create_truncating_wrapper(original_execute_query)

# Other functions (list_tables, get_table_schema, etc.) pass through unchanged
```

### 2. Result Format

The `execute_query` function now returns a dictionary with metadata:

**Small Results** (no truncation):
```python
{
    'result': [['row1'], ['row2'], ['row3']],
    'columns': ['col1', 'col2']
}
```

**Large Results** (truncated):
```python
{
    'result': [['row1'], ['row2'], ...],  # First 50 rows
    'columns': ['col1', 'col2'],
    'total_count': 5000,                   # Original row count
    'showing': 50,                         # Number of rows shown
    '_truncated': True,                    # Truncation flag
    '_instruction': 'Results truncated...' # Guidance for the model
}
```

### 3. Model Guidance

When results are truncated, the model receives an instruction to use efficient querying:

```
Results truncated: showing 50 of 5000 total rows.
IMPORTANT: Do not use these literal values in the next query.
Instead, use a subquery approach:
WHERE column IN (SELECT ... FROM ... WHERE ...)
This prevents token exhaustion and is more efficient.
```

## Benefits

### 1. No Configuration Conflicts
- Works with Gemini's standard automatic function calling
- No conflicting `automatic_function_calling` settings
- No manual iteration required

### 2. Prevents Token Exhaustion
- Large results never enter the chat context
- Reduces API costs
- Prevents 429 (rate limit) errors

### 3. Efficient Multi-Step Queries
- Model learns to use subqueries instead of literal values
- Queries remain efficient even with large intermediate results
- Database handles filtering instead of LLM

### 4. Transparent to Users
- Small queries work exactly as before
- Large queries automatically adapt
- No user configuration required (unless customization is desired)

## Testing

### Unit Tests

Comprehensive unit tests are available in `tests/unit/test_sql_truncation.py`:

```bash
# Run the SQL truncation tests
uv run pytest tests/unit/test_sql_truncation.py -v
```

Test coverage includes:
- Small result handling (no truncation)
- Large result handling (truncation)
- Row-based truncation
- Character-based truncation
- Non-dict result pass-through
- Function wrapping behavior
- Metadata preservation

### Manual Testing

To test the truncation behavior manually:

```python
from app.mirkat.node_sql import SQLNode

# Create a SQLNode with low thresholds for testing
sql_node = SQLNode(
    instructions="test",
    max_result_rows=10,
    max_result_chars=1000
)

# Test truncation logic directly
large_result = {
    'result': [[f'row{i}'] for i in range(100)],
    'columns': ['col1']
}

truncated = sql_node._truncate_result_if_needed(large_result)
print(f"Original rows: {len(large_result['result'])}")
print(f"Truncated rows: {len(truncated['result'])}")
print(f"Truncated flag: {truncated['_truncated']}")
```

## Migration Notes

### Breaking Changes

The `execute_query` function signature has changed:

**Before:**
```python
def execute_query(self, sql: str, query_name: str) -> list[list[str]]:
    # Returns a list of rows
    return results
```

**After:**
```python
def execute_query(self, sql: str, query_name: str) -> dict:
    # Returns a dictionary with metadata
    return {
        'result': results,
        'columns': columns
    }
```

### Compatibility

- **Backward Compatible**: The wrapper handles both old and new formats gracefully
- **Test Updates**: Existing tests using `min` and `max` as mock functions continue to work
- **No SDK Changes**: Works with existing Gemini SDK version

## Troubleshooting

### Results Not Being Truncated

Check the configuration:
```python
# Verify thresholds
print(f"Max rows: {sql_node.max_result_rows}")
print(f"Max chars: {sql_node.max_result_chars}")

# Check if execute_query is being wrapped
print(f"Functions: {sql_node.functions}")
print(f"Original functions: {sql_node.original_functions}")
```

### Excessive Truncation

Increase the thresholds:
```bash
export SQL_NODE_MAX_ROWS=100
export SQL_NODE_MAX_CHARS=10000
```

Or configure programmatically:
```python
sql_node = SQLNode(..., max_result_rows=100, max_result_chars=10000)
```

### Model Not Using Subqueries

The model should automatically adapt based on the `_instruction` field in truncated results. If not:
1. Check that truncation is occurring (look for `_truncated: True`)
2. Verify the instruction is being passed to the model
3. Consider updating the SQL instructions to emphasize efficient querying

## Implementation Details

### Files Modified

1. **`app/mirkat/node_sql.py`**
   - Added `max_result_rows` and `max_result_chars` parameters to `__init__`
   - Added `_wrap_functions_with_truncation()` method
   - Added `_create_truncating_wrapper()` method
   - Added `_truncate_result_if_needed()` helper method

2. **`app/mirkat/sql_functions.py`**
   - Modified `execute_query()` to return dict instead of list
   - Added column names to the response
   - Updated TSV file writing to include headers

3. **`app/nodes.py`**
   - Updated SQLNode instantiation to include environment variable configuration

4. **`tests/unit/test_sql_truncation.py`**
   - Added comprehensive unit tests for truncation logic

### Design Decisions

**Why wrap at the function level?**
- Intercepts results before they reach the chat context
- Works with automatic function calling (no SDK conflicts)
- Transparent to the Gemini SDK
- Easy to test and maintain

**Why dictionary format?**
- Allows including metadata (columns, counts, flags)
- More extensible for future enhancements
- Standard pattern in database libraries
- Better type safety

**Why both row and character limits?**
- Row limits prevent too many records
- Character limits prevent records with huge string fields
- Combined approach handles all edge cases

## Future Enhancements

Potential improvements for the future:
- [ ] Configurable truncation strategies (first N, random sample, etc.)
- [ ] Per-query truncation overrides
- [ ] Smart sampling that preserves distribution
- [ ] Compression for very large text fields
- [ ] Caching of truncated results
- [ ] Metrics/logging for truncation events
