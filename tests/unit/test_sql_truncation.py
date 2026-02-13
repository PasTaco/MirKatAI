# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for SQL node truncation functionality.
Tests the wrapper pattern that intercepts and truncates large query results.
"""
import pytest
from app.mirkat.node_sql import SQLNode


def test_sql_node_with_truncation_params() -> None:
    """Check that SQLNode accepts and stores truncation parameters."""
    sql_node = SQLNode(
        instructions="you are a SQL expert",
        functions=[min, max],
        max_result_rows=100,
        max_result_chars=10000
    )
    assert sql_node is not None
    assert sql_node.max_result_rows == 100
    assert sql_node.max_result_chars == 10000
    assert sql_node.original_functions == [min, max]


def test_sql_node_default_truncation_params() -> None:
    """Check that SQLNode uses default truncation parameters."""
    sql_node = SQLNode(
        instructions="you are a SQL expert",
        functions=[min, max]
    )
    assert sql_node is not None
    assert sql_node.max_result_rows == 50
    assert sql_node.max_result_chars == 5000


def test_truncate_result_small_dataset() -> None:
    """Test that small results are not truncated."""
    sql_node = SQLNode(
        instructions="you are a SQL expert",
        max_result_rows=50
    )
    
    # Small result that should not be truncated
    small_result = {
        'result': [['row1'], ['row2'], ['row3']],
        'columns': ['col1']
    }
    
    truncated = sql_node._truncate_result_if_needed(small_result)
    
    assert truncated == small_result
    assert '_truncated' not in truncated
    assert 'total_count' not in truncated


def test_truncate_result_large_dataset() -> None:
    """Test that large results are truncated correctly."""
    sql_node = SQLNode(
        instructions="you are a SQL expert",
        max_result_rows=10
    )
    
    # Large result that should be truncated
    large_result = {
        'result': [[f'row{i}'] for i in range(100)],
        'columns': ['col1']
    }
    
    truncated = sql_node._truncate_result_if_needed(large_result)
    
    assert truncated is not None
    assert len(truncated['result']) == 10
    assert truncated['total_count'] == 100
    assert truncated['showing'] == 10
    assert truncated['_truncated'] is True
    assert '_instruction' in truncated
    assert 'subquery' in truncated['_instruction']


def test_truncate_result_non_dict() -> None:
    """Test that non-dictionary results are passed through unchanged."""
    sql_node = SQLNode(
        instructions="you are a SQL expert",
        max_result_rows=10
    )
    
    # Non-dict result
    non_dict_result = [['row1'], ['row2']]
    
    truncated = sql_node._truncate_result_if_needed(non_dict_result)
    
    assert truncated == non_dict_result


def test_truncate_result_missing_result_key() -> None:
    """Test that dictionaries without 'result' key are passed through."""
    sql_node = SQLNode(
        instructions="you are a SQL expert",
        max_result_rows=10
    )
    
    # Dict without 'result' key
    invalid_result = {'columns': ['col1'], 'data': []}
    
    truncated = sql_node._truncate_result_if_needed(invalid_result)
    
    assert truncated == invalid_result


def test_wrap_functions_with_truncation() -> None:
    """Test that execute_query function is wrapped while others are not."""
    def execute_query(sql: str, query_name: str) -> dict:
        """Mock execute_query function."""
        return {'result': [[1, 2]], 'columns': ['a', 'b']}
    
    def list_tables() -> list:
        """Mock list_tables function."""
        return ['table1', 'table2']
    
    sql_node = SQLNode(
        instructions="you are a SQL expert",
        functions=[execute_query, list_tables],
        max_result_rows=10
    )
    
    # Check that functions were wrapped
    assert len(sql_node.functions) == 2
    assert sql_node.original_functions == [execute_query, list_tables]
    
    # execute_query should be wrapped (check by calling it)
    # The wrapped function should have the same name
    wrapped_execute_query = sql_node.functions[0]
    assert wrapped_execute_query.__name__ == 'execute_query'
    
    # list_tables should not be wrapped (same object)
    assert sql_node.functions[1] == list_tables


def test_wrapped_execute_query_truncates() -> None:
    """Test that wrapped execute_query function truncates large results."""
    def mock_execute_query(sql: str, query_name: str) -> dict:
        """Mock execute_query that returns large dataset."""
        return {
            'result': [[f'row{i}'] for i in range(100)],
            'columns': ['col1']
        }
    
    sql_node = SQLNode(
        instructions="you are a SQL expert",
        functions=[mock_execute_query],
        max_result_rows=10
    )
    
    # Get the wrapped function
    wrapped_func = sql_node.functions[0]
    
    # Call the wrapped function
    result = wrapped_func("SELECT * FROM table", "query1")
    
    # Verify truncation occurred
    assert len(result['result']) == 10
    assert result['total_count'] == 100
    assert result['_truncated'] is True


def test_wrapped_execute_query_no_truncation() -> None:
    """Test that wrapped execute_query doesn't truncate small results."""
    def mock_execute_query(sql: str, query_name: str) -> dict:
        """Mock execute_query that returns small dataset."""
        return {
            'result': [['row1'], ['row2']],
            'columns': ['col1']
        }
    
    sql_node = SQLNode(
        instructions="you are a SQL expert",
        functions=[mock_execute_query],
        max_result_rows=10
    )
    
    # Get the wrapped function
    wrapped_func = sql_node.functions[0]
    
    # Call the wrapped function
    result = wrapped_func("SELECT * FROM table", "query1")
    
    # Verify no truncation occurred
    assert len(result['result']) == 2
    assert '_truncated' not in result
    assert 'total_count' not in result


def test_truncation_by_character_count() -> None:
    """Test that truncation occurs based on character count threshold."""
    def mock_execute_query(sql: str, query_name: str) -> dict:
        """Mock execute_query that returns data exceeding char threshold."""
        # Create rows with long strings to exceed character limit
        return {
            'result': [['x' * 1000] for i in range(10)],
            'columns': ['col1']
        }
    
    sql_node = SQLNode(
        instructions="you are a SQL expert",
        functions=[mock_execute_query],
        max_result_rows=50,  # High row limit
        max_result_chars=5000  # Low character limit
    )
    
    # Get the wrapped function
    wrapped_func = sql_node.functions[0]
    
    # Call the wrapped function
    result = wrapped_func("SELECT * FROM table", "query1")
    
    # Verify truncation occurred due to character count
    assert result['_truncated'] is True
    assert 'total_count' in result
