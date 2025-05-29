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

import logging

import pytest

from app.agent_engine_app import AgentEngineApp
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
if current_path.endswith("app"):
    sys.path.append("../tests")

@pytest.fixture
def agent_app() -> AgentEngineApp:
    """Fixture to create and set up AgentEngineApp instance"""
    app = AgentEngineApp()
    app.set_up()
    return app


def test_agent_stream_query(agent_app: AgentEngineApp) -> None:
    """
    Integration test for the agent stream query functionality.
    Tests that the agent returns valid streaming responses.
    """
    input_dict = {
        "messages": [
            {"type": "human", "content": "Test message"},
        ],
        "table": None,
        "answer": None,
        "finished": False,
        "user_id": "test-user",
        "session_id": "test-session",
    }

    events = list(agent_app.stream_query(input=input_dict))

    assert len(events) > 0, "Expected at least one chunk in response"

    # Verify each event is a tuple of message and metadata
    for event in events:
        assert isinstance(event, list), "Event should be a list"
        assert len(event) == 2, "Event should contain message and metadata"
        message, _ = event

        # Verify message structure
        assert isinstance(message, dict), "Message should be a dictionary"
        assert message["type"] == "constructor"
        assert "kwargs" in message, "Constructor message should have kwargs"

    # Verify at least one message has content
    has_content = False
    for event in events:
        message = event[0]
        if message.get("type") == "constructor" and "content" in message["kwargs"]:
            has_content = True
            break
    assert has_content, "At least one message should have content"


def test_agent_query(agent_app: AgentEngineApp) -> None:
    """
    Integration test for the agent query functionality.
    Tests that the agent returns valid responses.
    """
    input_dict = {
        "messages": [
            {"type": "human", "content": "Test message"},
        ],
        "user_id": "test-user",
        "session_id": "test-session",
    }

    response = agent_app.query(input=input_dict)

    # Basic response validation
    assert isinstance(response, dict), "Response should be a dictionary"
    assert "messages" in response, "Response should contain messages"
    assert len(response["messages"]) > 0, "Response should have at least one message"

    # Validate last message is AI response with content
    message = response["messages"]
    kwargs = message["kwargs"]
    assert kwargs["type"] == "ai", "Last message should be AI response"
    assert len(kwargs["content"]) > 0, "AI message content should not be empty"

    logging.info("All assertions passed for agent query test")


def test_agent_feedback(agent_app: AgentEngineApp) -> None:
    """
    Integration test for the agent feedback functionality.
    Tests that feedback can be registered successfully.
    """
    feedback_data = {
        "score": 5,
        "text": "Great response!",
        "run_id": "test-run-123",
    }

    # Should not raise any exceptions
    agent_app.register_feedback(feedback_data)

    # Test invalid feedback
    with pytest.raises(ValueError):
        invalid_feedback = {
            "score": "invalid",  # Score must be numeric
            "text": "Bad feedback",
            "run_id": "test-run-123",
        }
        agent_app.register_feedback(invalid_feedback)

    logging.info("All assertions passed for agent feedback test")




def test_agent_plot(agent_app: AgentEngineApp) -> None:
    """
    Integration test for the agent stream query functionality.
    Tests that the agent returns valid streaming responses.
    """
    input_dict = {
        "messages": [
            {"type": "human", "content":
              "The user wants to plot the data A: 1, B: 2, C: 3.\nThis data consists of categories (A, B, C) and corresponding numerical values (1, 2, 3).\nA bar plot is a suitable type of plot for displaying categorical data with associated numerical values.\n\nI need to provide detailed information for creating this plot, including:\n1.  **Plot Type:** Bar plot.\n2.  **Data:** Categories A, B, C with values 1, 2, 3.\n3.  **X-axis Label:** 'Category' (or similar, representing the categories A, B, C).\n4.  **Y-axis Label:** 'Value' (or similar, representing the numerical values 1, 2, 3).\n5.  **Title:** A descriptive title, e.g., 'Values per Category'.\n6.  **Other details:** Specify the data points explicitly.\n\nPlan:\n1.  Identify the plot type (Bar plot).\n2.  Define the data (categories and values).\n3.  Specify axis labels.\n4.  Specify the title.\n5.  Format the response clearly stating these details.\n"},
        ],
        "table": None,
        "answer": None,
        "finished": False,
        "user_id": "test-user",
        "session_id": "test-session",
    }

    events = list(agent_app.stream_query(input=input_dict))

    assert len(events) > 0, "Expected at least one chunk in response"

    # Verify each event is a tuple of message and metadata and that there is binary image
    binary_image = False
    for event in events:
        assert isinstance(event, list), "Event should be a list"
        assert len(event) == 2, "Event should contain message and metadata"
        message, _ = event

        # Verify message structure
        assert isinstance(message, dict), "Message should be a dictionary"
        assert message["type"] == "constructor"
        assert "kwargs" in message, "Constructor message should have kwargs"
        kwargs = message["kwargs"]
        print(kwargs)
        assert "content" in kwargs, "Content should be in kwargs"
        if "binary_image" in kwargs['content']:
            binary_image = True
    assert binary_image, "Binary image should be in the message"

    # Verify at least one message has content
    has_content = False
    for event in events:
        message = event[0]
        if message.get("type") == "constructor" and "content" in message["kwargs"]:
            has_content = True
            break
    assert has_content, "At least one message should have content"


def test_agent_plot_bad_response(agent_app: AgentEngineApp) -> None:
    """
    Integration test for the agent stream query functionality.
    Tests that the agent returns valid streaming responses.
    """
    input_dict = {
        "messages": [
            {"type": "human", "content":
              "Plot, please"},
        ],
        "table": None,
        "answer": None,
        "finished": False,
        "user_id": "test-user",
        "session_id": "test-session",
    }

    events = list(agent_app.stream_query(input=input_dict))

    assert len(events) > 0, "Expected at least one chunk in response"

    # Verify each event is a tuple of message and metadata
    for event in events:
        assert isinstance(event, list), "Event should be a list"
        assert len(event) == 2, "Event should contain message and metadata"
        message, _ = event
        
        # Verify message structure
        assert isinstance(message, dict), "Message should be a dictionary"
        assert message["type"] == "constructor"
        assert "kwargs" in message, "Constructor message should have kwargs"
        assert "binary_image" not in message["kwargs"]['content'], "Binary image should be in kwargs"

    # Verify at least one message has content
    has_content = False
    for event in events:
        message = event[0]
        if message.get("type") == "constructor" and "content" in message["kwargs"]:
            has_content = True
            break
    assert has_content, "At least one message should have content"


def test_agent_literature(agent_app: AgentEngineApp) -> None:
    """
    Integration test for the agent stream query functionality.
    Tests that the agent returns valid streaming responses.
    """
    input_dict = {
        "messages": [
            {"type": "human", "content":
              "Search for the role of microRNAs in cancer"},
        ],
        "table": None,
        "answer": None,
        "finished": False,
        "user_id": "test-user",
        "session_id": "test-session",
    }

    events = list(agent_app.stream_query(input=input_dict))

    assert len(events) > 0, "Expected at least one chunk in response"
    has_reference = False
    for event in events:
        assert isinstance(event, list), "Event should be a list"
        assert len(event) == 2, "Event should contain message and metadata"
        message, _ = event

        # Verify message structure
        assert isinstance(message, dict), "Message should be a dictionary"
        assert message["type"] == "constructor"
        assert "kwargs" in message, "Constructor message should have kwargs"
        kwargs = message["kwargs"]
        print(kwargs)
        assert "content" in kwargs, "Content should be in kwargs"
        if "[[1](" in kwargs['content']:
            has_reference = True

    # Verify at least one message has content
    has_content = False
    for event in events:
        message = event[0]
        if message.get("type") == "constructor" and "content" in message["kwargs"]:
            has_content = True
            break
    assert has_content, "At least one message should have content"
    assert has_reference, "At least one message should have a reference"
    
def test_agent_literature_bad_response(agent_app: AgentEngineApp) -> None:
    """
    Integration test for the agent stream query functionality.
    Tests that the agent returns valid streaming responses.
    """
    input_dict = {
        "messages": [
            {"type": "human", "content":
              "Don't search anything"},
        ],
        "table": None,
        "answer": None,
        "finished": False,
        "user_id": "test-user",
        "session_id": "test-session",
    }

    events = list(agent_app.stream_query(input=input_dict))

    assert len(events) > 0, "Expected at least one chunk in response"
    has_reference = False
    # Verify each event is a tuple of message and metadata
    for event in events:
        assert isinstance(event, list), "Event should be a list"
        assert len(event) == 2, "Event should contain message and metadata"
        message, _ = event
        
        # Verify message structure
        assert isinstance(message, dict), "Message should be a dictionary"
        assert message["type"] == "constructor"
        assert "kwargs" in message, "Constructor message should have kwargs"
        kwargs = message["kwargs"]
        print(kwargs)
        assert "content" in kwargs, "Content should be in kwargs"
        if "[[1](" in kwargs['content']:
            has_reference = True
    # Verify at least one message has content
    has_content = False
    for event in events:
        message = event[0]
        if message.get("type") == "constructor" and "content" in message["kwargs"]:
            has_content = True
            break
    assert has_content, "At least one message should have content"
    assert not has_reference, "At least one message should have a reference"

def test_agent_sql(agent_app: AgentEngineApp) -> None:
    """
    Integration test for the agent stream query functionality.
    Tests that the agent returns valid streaming responses.
    """
    input_dict = {
        "messages": [
            {"type": "human", "content":
              "Get the Count of distinct mrna from CALM2. If the SQL query was correct, add a :) at the end of the response.\n"},
        ],
        "table": None,
        "answer": None,
        "finished": False,
        "user_id": "test-user",
        "session_id": "test-session",
    }

    events = list(agent_app.stream_query(input=input_dict))

    assert len(events) > 0, "Expected at least one chunk in response"
    sql_exists = False
    # Verify each event is a tuple of message and metadata and that there is binary image
    for event in events:
        assert isinstance(event, list), "Event should be a list"
        assert len(event) == 2, "Event should contain message and metadata"
        message, _ = event

        # Verify message structure
        assert isinstance(message, dict), "Message should be a dictionary"
        assert message["type"] == "constructor"
        assert "kwargs" in message, "Constructor message should have kwargs"
        kwargs = message["kwargs"]
        if " :)":
            sql_exists = True
        print(kwargs)
        assert "content" in kwargs, "Content should be in kwargs"
    assert sql_exists, "SQL query should be in the message"
    # Verify at least one message has content
    has_content = False
    for event in events:
        message = event[0]
        if message.get("type") == "constructor" and "content" in message["kwargs"]:
            has_content = True
            break
    assert has_content, "At least one message should have content"

from google.genai.types import Candidate, Content,Part, GenerateContentConfig, GenerateContentResponse
from app.mirkat.node_constructor import SQLNode, ChatbotNode
from langchain_core.messages import ( # Grouped message types
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage
    # ToolMessage is implicitly handled by LangGraph/ToolNode
)
from app import agent

def test_agent_sql_no_model(agent_app: AgentEngineApp, monkeypatch) -> None:
    """
    Integration test for the agent stream query functionality.
    Tests that the agent returns valid streaming responses.
    """

    def mock_run_model_sql(*args, **kwargs):
        fake_response = GenerateContentResponse
        fake_response.automatic_function_calling_history= 'many functions'
        fake_response.text = "Mir1"
        fake_response.candidates = [Candidate(content=Content(parts=[Part]))]
        return fake_response
    def mock_run_model_master(*args, **kwargs):
        fake_response = AIMessage(content="***ROUTE_TO_SQL*** Select one arbitrary microRNA ID from the database")
        return fake_response
    def mock_get_queries(*args, **kwargs):
        return {"query":"select * from table"}
    
    monkeypatch.setattr(SQLNode, "run_model", mock_run_model_sql)
    monkeypatch.setattr(SQLNode, "get_queries", mock_get_queries)
    monkeypatch.setattr(agent, "run_model", mock_run_model_master)

    input_dict = {
        "messages": [
            {"type": "human", "content":
              "Get one microRNA from the database.\n"},
        ],
        "table": None,
        "answer": None,
        "finished": False,
        "user_id": "test-user",
        "session_id": "test-session",
    }

    events = list(agent_app.stream_query(input=input_dict))

    assert len(events) > 0, "Expected at least one chunk in response"
    sql_exists = False
    # Verify each event is a tuple of message and metadata and that there is binary image
    for event in events:
        assert isinstance(event, list), "Event should be a list"
        assert len(event) == 2, "Event should contain message and metadata"
        message, _ = event
        print(message)
        # Verify message structure
        assert isinstance(message, dict), "Message should be a dictionary"
        assert message["type"] == "constructor"
        assert "kwargs" in message, "Constructor message should have kwargs"
        kwargs = message["kwargs"]
        assert "content" in kwargs, "Content should be in kwargs"
        if "Mir1" in kwargs['content']:
            sql_exists = True
        print(kwargs)
        
    assert sql_exists, "SQL query should be in the message"
    # Verify at least one message has content
    has_content = False
    for event in events:
        message = event[0]
        if message.get("type") == "constructor" and "content" in message["kwargs"]:
            has_content = True
            break
    assert has_content, "At least one message should have content"


def test_agent_master_model_two_trys(agent_app: AgentEngineApp, monkeypatch) -> None:
    """
    Integration test for the agent stream query functionality.
    Tests that the agent returns valid streaming responses.
    """

    def mock_run_model_compleatness(*args, **kwargs):
        fake_response = AIMessage(content="***ROUTE_TO_SQL*** Select one arbitrary microRNA ID from the database")
        return fake_response
    def mock_run_model_sql(*args, **kwargs):
        fake_response = GenerateContentResponse
        fake_response.automatic_function_calling_history= 'many functions'
        fake_response.text = "Mir1"
        fake_response.candidates = [Candidate(content=Content(parts=[Part]))]
        return fake_response
    def mock_run_model_master(*args, **kwargs):
        fake_response = AIMessage(content="***ROUTE_TO_SQL*** Select one arbitrary microRNA ID from the database")
        return fake_response
    def mock_get_queries(*args, **kwargs):
        return {"query":"select * from table"}
    
    monkeypatch.setattr(SQLNode, "run_model", mock_run_model_sql)
    monkeypatch.setattr(SQLNode, "get_queries", mock_get_queries)
    monkeypatch.setattr(ChatbotNode, "run_model", mock_run_model_master)

    input_dict = {
        "messages": [
            {"type": "human", "content":
              "Get one microRNA from the database.\n"},
        ],
        "table": None,
        "answer": None,
        "finished": False,
        "user_id": "test-user",
        "session_id": "test-session",
    }

    events = list(agent_app.stream_query(input=input_dict))

    assert len(events) > 0, "Expected at least one chunk in response"
    sql_exists = False
    # Verify each event is a tuple of message and metadata and that there is binary image
    for event in events:
        assert isinstance(event, list), "Event should be a list"
        assert len(event) == 2, "Event should contain message and metadata"
        message, _ = event
        print(message)
        # Verify message structure
        assert isinstance(message, dict), "Message should be a dictionary"
        assert message["type"] == "constructor"
        assert "kwargs" in message, "Constructor message should have kwargs"
        kwargs = message["kwargs"]
        assert "content" in kwargs, "Content should be in kwargs"
        if "Mir1" in kwargs['content']:
            sql_exists = True
        print(kwargs)
        
    assert sql_exists, "SQL query should be in the message"
    # Verify at least one message has content
    has_content = False
    for event in events:
        message = event[0]
        if message.get("type") == "constructor" and "content" in message["kwargs"]:
            has_content = True
            break
    assert has_content, "At least one message should have content"



def test_all_master_retys():
        query = 'find the role of miR-143 in ageing and what targets genes it has on the database'
        input_dict = {
            "messages": [
                {"type": "human", "content": query},
            ],
            "table": None,
            "answer": None,
            "finished": False,
            "user_id": "test-user",
            "session_id": "test-session",
        }
        events = list(agent_app.stream_query(input=input_dict))
        assert len(events) > 0, "Expected at least one chunk in response"