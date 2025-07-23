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

# mypy: disable-error-code="union-attr"
from app.agent import agent
from google.genai.types import GenerateContentResponse
from google.genai.types import Candidate, Content,Part, GenerateContentConfig
from app.mirkat.node_sql import SQLNode
import os
from app.agent_engine_app import AgentEngineApp
import pytest
current_path = os.path.dirname(os.path.abspath(__file__))
@pytest.fixture
def agent_app() -> AgentEngineApp:
    """Fixture to create and set up AgentEngineApp instance"""
    app = AgentEngineApp(project_id='mirkatdb')
    app.set_up()
    return app

def test_agent_stream() -> None:
    """
    Integration test for the agent stream functionality.
    Tests that the agent returns valid streaming responses.
    """
    input_dict = {
        "messages": [
            {"type": "human", "content": "Hi"},
            {"type": "ai", "content": "Hi there!"},
            {"type": "human", "content": "What's the weather in NY?"},
        ],
        "answer": None,
    }

    events = [
        message for message, _ in agent.stream(input_dict, stream_mode="messages")
    ]

    # Verify we get a reasonable number of messages
    assert len(events) > 0, "Expected at least one message"

    # First message should be an AI message
    assert events[0].type == "AIMessageChunk"

    # At least one message should have content
    has_content = False
    for event in events:
        if hasattr(event, "content") and event.content:
            has_content = True
            break
    assert has_content, "Expected at least one message with content"


def test_agent_sql(agent_app: AgentEngineApp, monkeypatch) -> None:
    """
    Integration test for the agent stream query functionality.
    Tests that the agent returns valid streaming responses.
    """

    def mock_run_model(*args, **kwargs):
        fake_response = GenerateContentResponse
        fake_response.automatic_function_calling_history= 'many functions'
        fake_response.text = "Mir1"
        fake_response.candidates = [Candidate(content=Content(parts=[Part]))]
        return fake_response
    def mock_get_queries(*args, **kwargs):
        return {"query":"select * from table"}
    
    monkeypatch.setattr(SQLNode, "run_model", mock_run_model)
    monkeypatch.setattr(SQLNode, "get_queries", mock_get_queries)

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

        # Verify message structure
        assert isinstance(message, dict), "Message should be a dictionary"
        assert message["type"] == "constructor"
        assert "kwargs" in message, "Constructor message should have kwargs"
        assert "content" in kwargs, "Content should be in kwargs"
        kwargs = message["kwargs"]
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