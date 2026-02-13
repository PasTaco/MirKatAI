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
Unit tests for SQL Node truncation and manual function calling.
Tests the fix for 429 RESOURCE_EXHAUSTED errors.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch

from app.mirkat.node_sql import SQLNode
from google.genai.types import (
    GenerateContentResponse,
    Candidate,
    Content,
    Part,
    FunctionCall,
    FunctionResponse
)
from langchain_core.messages import AIMessage


@pytest.fixture
def mock_sql_node():
    """Create a mock SQL node with mocked client."""
    with patch('app.mirkat.node_constructor.genai.Client'):
        node = SQLNode(
            instructions="test",
            functions=[],
            max_result_rows=50,
            max_result_chars=5000
        )
        node.chat = Mock()
        return node


class TestSQLNodeTruncation:
    """Test SQL Node result truncation functionality."""

    def test_truncation_with_large_result_by_rows(self, mock_sql_node):
        """Test that results are truncated when exceeding max_result_rows."""
        # Create a large result with 1000 rows
        large_result = {
            'result': [(f'mir-{i}',) for i in range(1000)]
        }

        truncated = mock_sql_node._truncate_result_if_needed(large_result)

        assert truncated['_truncated'] is True
        assert truncated['total_count'] == 1000
        assert truncated['showing'] == 50
        assert len(truncated['result']) == 50
        assert '_instruction' in truncated
        assert 'subquery' in truncated['_instruction'].lower()

    def test_truncation_with_large_result_by_chars(self, mock_sql_node):
        """Test that results are truncated when exceeding max_result_chars."""
        mock_sql_node.max_result_chars = 1000
        mock_sql_node.max_result_rows = 1000

        # Create a result with long strings that exceed char limit
        large_result = {
            'result': [('x' * 100,) for i in range(50)]
        }

        truncated = mock_sql_node._truncate_result_if_needed(large_result)

        # Verify truncation occurred due to character limit
        assert truncated['_truncated'] is True
        assert truncated['total_count'] == 50
        # Verify that original result data exceeded limit
        assert len(str(large_result)) > 1000

    def test_no_truncation_with_small_result(self, mock_sql_node):
        """Test that small results are not truncated."""
        small_result = {
            'result': [(f'mir-{i}',) for i in range(10)]
        }

        result = mock_sql_node._truncate_result_if_needed(small_result)

        assert '_truncated' not in result
        assert result == small_result

    def test_truncation_with_empty_result(self, mock_sql_node):
        """Test that empty results are handled correctly."""
        empty_result = {'result': []}
        result = mock_sql_node._truncate_result_if_needed(empty_result)

        assert result == empty_result

    def test_truncation_with_invalid_result(self, mock_sql_node):
        """Test that invalid results are returned unchanged."""
        # Test with non-dict result
        result1 = mock_sql_node._truncate_result_if_needed("not a dict")
        assert result1 == "not a dict"

        # Test with dict without 'result' key
        result2 = mock_sql_node._truncate_result_if_needed({'data': []})
        assert result2 == {'data': []}

    def test_configurable_thresholds(self):
        """Test that truncation thresholds are configurable."""
        with patch('app.mirkat.node_constructor.genai.Client'):
            sql_node = SQLNode(
                instructions="test",
                functions=[],
                max_result_rows=10,
                max_result_chars=100
            )

            assert sql_node.max_result_rows == 10
            assert sql_node.max_result_chars == 100

            # Test with result that exceeds custom threshold
            result = {
                'result': [(f'mir-{i}',) for i in range(20)]
            }

            truncated = sql_node._truncate_result_if_needed(result)
            assert truncated['_truncated'] is True
            assert len(truncated['result']) == 10


class TestSQLNodeFunctionCalling:
    """Test SQL Node manual function calling functionality."""

    def test_has_function_call_with_function(self, mock_sql_node):
        """Test detection of function calls in response."""
        # Create a mock response with a function call
        function_call = FunctionCall(
            name="execute_query",
            args={"sql": "SELECT * FROM mirna"}
        )
        part = Part(function_call=function_call)
        content = Content(parts=[part], role="model")
        candidate = Candidate(content=content)
        
        response = Mock(spec=GenerateContentResponse)
        response.candidates = [candidate]

        assert mock_sql_node._has_function_call(response) is True

    def test_has_function_call_without_function(self, mock_sql_node):
        """Test detection when no function call present."""
        # Create a mock response with just text
        part = Part(text="Here is your answer")
        content = Content(parts=[part], role="model")
        candidate = Candidate(content=content)
        
        response = Mock(spec=GenerateContentResponse)
        response.candidates = [candidate]

        assert mock_sql_node._has_function_call(response) is False

    def test_extract_function_call(self, mock_sql_node):
        """Test extraction of function call details."""
        # Create a mock response with a function call
        function_call = FunctionCall(
            name="execute_query",
            args={"sql": "SELECT * FROM mirna", "query_name": "test"}
        )
        part = Part(function_call=function_call)
        content = Content(parts=[part], role="model")
        candidate = Candidate(content=content)
        
        response = Mock(spec=GenerateContentResponse)
        response.candidates = [candidate]

        extracted = mock_sql_node._extract_function_call(response)
        
        assert extracted is not None
        assert extracted.name == "execute_query"
        assert extracted.args["sql"] == "SELECT * FROM mirna"

    def test_execute_function_success(self):
        """Test successful function execution."""
        # Create a mock function
        def mock_execute_query(sql, query_name):
            return [("mir-1",), ("mir-2",)]

        with patch('app.mirkat.node_constructor.genai.Client'):
            sql_node = SQLNode(
                instructions="test",
                functions=[mock_execute_query]
            )

            function_call = FunctionCall(
                name="mock_execute_query",
                args={"sql": "SELECT * FROM mirna", "query_name": "test"}
            )

            result = sql_node._execute_function(function_call)
            
            assert 'result' in result
            assert result['result'] == [("mir-1",), ("mir-2",)]

    def test_execute_function_not_found(self, mock_sql_node):
        """Test function execution when function not found."""
        function_call = FunctionCall(
            name="nonexistent_function",
            args={}
        )

        result = mock_sql_node._execute_function(function_call)
        
        assert 'error' in result
        assert 'not found' in result['error'].lower()

    def test_execute_function_with_error(self):
        """Test function execution when function raises an error."""
        def mock_function_with_error(**kwargs):
            raise ValueError("Database error")

        with patch('app.mirkat.node_constructor.genai.Client'):
            sql_node = SQLNode(
                instructions="test",
                functions=[mock_function_with_error]
            )

            function_call = FunctionCall(
                name="mock_function_with_error",
                args={}
            )

            result = sql_node._execute_function(function_call)
            
            assert 'error' in result
            assert 'Database error' in result['error']

    def test_format_function_response(self, mock_sql_node):
        """Test formatting of function response for next iteration."""
        function_call = FunctionCall(
            name="execute_query",
            args={"sql": "SELECT * FROM mirna"}
        )

        result = {'result': [("mir-1",), ("mir-2",)]}

        content = mock_sql_node._format_function_response(function_call, result)

        assert isinstance(content, Content)
        assert content.role == "user"
        assert len(content.parts) == 1
        assert content.parts[0].function_response is not None
        assert content.parts[0].function_response.name == "execute_query"
        assert content.parts[0].function_response.response == result

    def test_reconstruct_history(self, mock_sql_node):
        """Test reconstruction of function calling history."""
        function_call = FunctionCall(
            name="execute_query",
            args={"sql": "SELECT * FROM mirna"}
        )

        result = {'result': [("mir-1",), ("mir-2",)]}

        all_function_calls = [{
            'function_call': function_call,
            'result': result
        }]

        history = mock_sql_node._reconstruct_history(all_function_calls)

        assert len(history) == 2  # Call + Response
        assert history[0].role == "model"
        assert history[0].parts[0].function_call is not None
        assert history[1].role == "user"
        assert history[1].parts[0].function_response is not None


class TestSQLNodeIntegration:
    """Integration tests for SQL Node with truncation."""

    def test_truncation_prevents_large_context(self):
        """Test that truncation prevents large results from entering context."""
        def mock_execute_query(sql, query_name):
            # Return a large result
            return [(f'mir-{i}',) for i in range(5000)]

        with patch('app.mirkat.node_constructor.genai.Client'):
            sql_node = SQLNode(
                instructions="test",
                functions=[mock_execute_query],
                max_result_rows=50,
                max_result_chars=5000
            )

            function_call = FunctionCall(
                name="mock_execute_query",
                args={"sql": "SELECT * FROM mirna", "query_name": "test"}
            )

            # Execute function
            result = sql_node._execute_function(function_call)
            assert len(result['result']) == 5000

            # Truncate result
            truncated = sql_node._truncate_result_if_needed(result)
            
            # Verify truncation
            assert truncated['_truncated'] is True
            assert len(truncated['result']) == 50
            assert truncated['total_count'] == 5000
            
            # Verify instruction is present
            assert '_instruction' in truncated
            assert 'subquery' in truncated['_instruction'].lower()

    def test_small_results_unchanged(self):
        """Test that small results pass through unchanged."""
        def mock_execute_query(sql, query_name):
            return [("mir-1",), ("mir-2",)]

        with patch('app.mirkat.node_constructor.genai.Client'):
            sql_node = SQLNode(
                instructions="test",
                functions=[mock_execute_query],
                max_result_rows=50,
                max_result_chars=5000
            )

            function_call = FunctionCall(
                name="mock_execute_query",
                args={"sql": "SELECT * FROM mirna LIMIT 2", "query_name": "test"}
            )

            # Execute function
            result = sql_node._execute_function(function_call)
            
            # Truncate result (should not truncate)
            processed = sql_node._truncate_result_if_needed(result)
            
            # Verify no truncation
            assert '_truncated' not in processed
            assert processed == result
