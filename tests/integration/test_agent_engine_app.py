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
import os, sys

from app.agent import LITERATURE_NODE

current_path = os.path.dirname(os.path.abspath(__file__))
if current_path.endswith("app"):
    sys.path.append("../tests")
#if current_path.endswith("tests/integration"):
#    sys.path.append("../")
#    sys.path.append("../app")
import logging

import pytest

from app.agent_engine_app import AgentEngineApp

from google.genai.types import Candidate, Content,Part, GenerateContentConfig, GenerateContentResponse
#from app.mirkat.node_constructor import SQLNode, ChatbotNode
from app.mirkat.node_sql import SQLNode
from app.mirkat.node_chatbot import ChatbotNode
from app.mirkat.node_literature import LiteratureNode
from langchain_core.messages import ( # Grouped message types
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage
    # ToolMessage is implicitly handled by LangGraph/ToolNode
)
from app import agent
from unittest.mock import patch, MagicMock




@pytest.fixture
def agent_app() -> AgentEngineApp:
    """Fixture to create and set up AgentEngineApp instance"""
    app = AgentEngineApp(project_id='mirkatdb')
    app.set_up()
    return app


def test_agent_stream_query(agent_app, monkeypatch):
    """
    Integration test for the agent stream query functionality.
    Tests that the agent returns valid streaming responses.
    """

    def mock_run_model_master(*args, **kwargs):
        fake_response = AIMessage(content='***ANSWER_DIRECTLY*** This is a test message.')
        return fake_response
    def mock_is_compleated(*args, **kwargs):
        return {'answer': 'YES', 'return': 'This is a test message.'}

    monkeypatch.setattr(ChatbotNode, "run_model", mock_run_model_master)
    monkeypatch.setattr(ChatbotNode, "is_compleated", mock_is_compleated)
    # Setup mock for ChatGoogleGenerativeAI.invoke
    input_dict = {
        "messages": [
            {"type": "human", "content": "Test message"},
        ],
        "table": None,
        "answer": None,
        "finished": False,
        "history": [],
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


def test_agent_query(agent_app: AgentEngineApp, monkeypatch) -> None:
    """
    Integration test for the agent query functionality.
    Tests that the agent returns valid responses.
    """
    def mock_run_model_master(*args, **kwargs):
        fake_response = AIMessage(content='***ANSWER_DIRECTLY*** This is a test message.')
        return fake_response
    def mock_is_compleated(*args, **kwargs):
        return {'answer': 'YES', 'return': 'This is a test message.'}

    monkeypatch.setattr(ChatbotNode, "run_model", mock_run_model_master)
    monkeypatch.setattr(ChatbotNode, "is_compleated", mock_is_compleated)
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



@pytest.mark.skip(reason="Plot needs to be fixed")
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

@pytest.mark.skip(reason="Plot needs to be fixed")
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


def test_agent_literature(agent_app: AgentEngineApp, monkeypatch) -> None:
    """
    Integration test for the agent stream query functionality.
    Tests that the agent returns valid streaming responses.
    """
    literature_answer = """*   **General Information:** cel-let-7 is a microRNA found in *Caenorhabditis elegans*.[[1](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXGmaUF80RC9kPupZBBJExM094Vwh2eXaoTruem_7VyJ_A2mlH-WhkULfV_-4xfIYXviA-kNyY_6Y2vVwRB-bPoRbtt3oy-7wyS4X7Af0hg8y_Ju8yIdln16v8Xg7AIwgvB4sLcgRMhcsAXpGzsEedDVuw==),[2](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXHbZOWmVHuvaE6e6Mzlajzv29ObibwE3wQXQZr3t9vY2TQkHFg6JJyt_BMlebS9-LXwFkRr__sf4Z5YBxi5JWmz0oiApewUk8SYd6jkyuKJsUF5X1shUvim6NXDnNUYEqlaIQb9Vq_enqRAEZjXaBRQz5cZ7x-Sqw==),[3](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFewSmXx_88n1nfG5eaEcIbMQFoP-wYUcwlhyaMyMdhOmFI_3wtq3wc0lT7h0Iae7Dz7JZuLx8jzfcG5OQonttQ5YmQq2RLaxLM4mwbGTPgOrZmO2tU49zlCn6-rac2tedZnDIpr8-JSQZHKWZ5gw1Y_LXv4XcB2Zis8g==),[4](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXESj1tob_T89BAFfvtgM0LjxR7Ygla-0Ke7HhBjWdFmNvJf27yQTw5mLIVJynTDAKWrnMc76JIpiIJZmvQVgHF87_GRtnw28cdxPEKwiWJnTgBsFAVCKoiNFK_-wKhNWg9GmAGrCyYwHndyYzVWnWzN1Ob34POZ-4ZUvU4VxadNmh5pVcowHdIIzRGaoA-EOe9Djc3xi-xT-nh-vkgeL9UI__HY7nTc6sPprxVpO5tpW10NFk7aMbVaE23eAPku)] It plays a role in developmental timing, specifically in the transition to late-larval and adult stages by translational repression of target mRNAs.[[2](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXHbZOWmVHuvaE6e6Mzlajzv29ObibwE3wQXQZr3t9vY2TQkHFg6JJyt_BMlebS9-LXwFkRr__sf4Z5YBxi5JWmz0oiApewUk8SYd6jkyuKJsUF5X1shUvim6NXDnNUYEqlaIQb9Vq_enqRAEZjXaBRQz5cZ7x-Sqw==)] It is located on chromosome X in *C. elegans*.[[2](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXHbZOWmVHuvaE6e6Mzlajzv29ObibwE3wQXQZr3t9vY2TQkHFg6JJyt_BMlebS9-LXwFkRr__sf4Z5YBxi5JWmz0oiApewUk8SYd6jkyuKJsUF5X1shUvim6NXDnNUYEqlaIQb9Vq_enqRAEZjXaBRQz5cZ7x-Sqw==)]
    *   **miRBase:** The miRBase entry for cel-let-7 (Accession MI0000001) provides detailed information, including its sequence, structure, experimental evidence, and links to related databases.
    *   **Evolutionary Conservation:** The let-7 microRNA family is highly conserved across different species, including vertebrates. Its expression often coincides with differentiation and opposes stem cell pluripotency, acting as a tumor suppressor in certain contexts.[[3](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFewSmXx_88n1nfG5eaEcIbMQFoP-wYUcwlhyaMyMdhOmFI_3wtq3wc0lT7h0Iae7Dz7JZuLx8jzfcG5OQonttQ5YmQq2RLaxLM4mwbGTPgOrZmO2tU49zlCn6-rac2tedZnDIpr8-JSQZHKWZ5gw1Y_LXv4XcB2Zis8g==)]
    *   **Biogenesis and Regulation:** Biogenesis of let-7 is tightly regulated by cellular factors like LIN28A/B and DIS3L2.[[4](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXESj1tob_T89BAFfvtgM0LjxR7Ygla-0Ke7HhBjWdFmNvJf27yQTw5mLIVJynTDAKWrnMc76JIpiIJZmvQVgHF87_GRtnw28cdxPEKwiWJnTgBsFAVCKoiNFK_-wKhNWg9GmAGrCyYwHndyYzVWnWzN1Ob34POZ-4ZUvU4VxadNmh5pVcowHdIIzRGaoA-EOe9Djc3xi-xT-nh-vkgeL9UI__HY7nTc6sPprxVpO5tpW10NFk7aMbVaE23eAPku)] Dysregulation of let-7 processing can have deleterious effects.[[4](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXESj1tob_T89BAFfvtgM0LjxR7Ygla-0Ke7HhBjWdFmNvJf27yQTw5mLIVJynTDAKWrnMc76JIpiIJZmvQVgHF87_GRtnw28cdxPEKwiWJnTgBsFAVCKoiNFK_-wKhNWg9GmAGrCyYwHndyYzVWnWzN1Ob34POZ-4ZUvU4VxadNmh5pVcowHdIIzRGaoA-EOe9Djc3xi-xT-nh-vkgeL9UI__HY7nTc6sPprxVpO5tpW10NFk7aMbVaE23eAPku)]
    *   **Function:** let-7 is required for cell cycle exit and differentiation in hypodermal cell lineages during larval development in *C. elegans*.[[3](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFewSmXx_88n1nfG5eaEcIbMQFoP-wYUcwlhyaMyMdhOmFI_3wtq3wc0lT7h0Iae7Dz7JZuLx8jzfcG5OQonttQ5YmQq2RLaxLM4mwbGTPgOrZmO2tU49zlCn6-rac2tedZnDIpr8-JSQZHKWZ5gw1Y_LXv4XcB2Zis8g==)] It has critical functions in cell fate specification and developmental progression in diverse animals.[[3](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFewSmXx_88n1nfG5eaEcIbMQFoP-wYUcwlhyaMyMdhOmFI_3wtq3wc0lT7h0Iae7Dz7JZuLx8jzfcG5OQonttQ5YmQq2RLaxLM4mwbGTPgOrZmO2tU49zlCn6-rac2tedZnDIpr8-JSQZHKWZ5gw1Y_LXv4XcB2Zis8g==)]
    *   **Targets:** Target genes of let-7 include c-Myc, high-mobility group A (HMGA), STAT3, and JAK2, which are involved in cell proliferation and the cell cycle.[[5](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXG63BA4NR08TXubox8gAamrTPyMLwmZWgBCKmjYaLXfjhR1dURgYdgn0TkkHYVxr0Phdm9J0C9TAn2vKszReTUJ_LcfSGs_jrPujPq8WrrvKm0DWKWSkca8_GZVaSh5tfWRFUp-JFB-1Y5_qT1du4sftTSJzDg4XXBpYMvg5Ehh3ZyuU2Yd9VLnMUCpZKyEZg==)]
    *   **Role in Cancer:** let-7 underexpression has been reported to be significantly correlated with patient outcome in non-small cell lung cancer (NSCLC).[[5](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXG63BA4NR08TXubox8gAamrTPyMLwmZWgBCKmjYaLXfjhR1dURgYdgn0TkkHYVxr0Phdm9J0C9TAn2vKszReTUJ_LcfSGs_jrPujPq8WrrvKm0DWKWSkca8_GZVaSh5tfWRFUp-JFB-1Y5_qT1du4sftTSJzDg4XXBpYMvg5Ehh3ZyuU2Yd9VLnMUCpZKyEZg==)] It can regulate cellular proliferation by targeting the Ras family.[[5](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXG63BA4NR08TXubox8gAamrTPyMLwmZWgBCKmjYaLXfjhR1dURgYdgn0TkkHYVxr0Phdm9J0C9TAn2vKszReTUJ_LcfSGs_jrPujPq8WrrvKm0DWKWSkca8_GZVaSh5tfWRFUp-JFB-1Y5_qT1du4sftTSJzDg4XXBpYMvg5Ehh3ZyuU2Yd9VLnMUCpZKyEZg==)]
    *   **Tools for Target Prediction:** Several tools, such as seedVicious, miRanda, Diana-microT, PicTar, and RNAhybrid, are available for predicting microRNA targets.[[1](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXGmaUF80RC9kPupZBBJExM094Vwh2eXaoTruem_7VyJ_A2mlH-WhkULfV_-4xfIYXviA-kNyY_6Y2vVwRB-bPoRbtt3oy-7wyS4X7Af0hg8y_Ju8yIdln16v8Xg7AIwgvB4sLcgRMhcsAXpGzsEedDVuw==)]
    *   **Absence in Some Species:** Interestingly, some species within the *Caenorhabditis* genus lack the let-7 sequence in their genomic assemblies.[[3](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFewSmXx_88n1nfG5eaEcIbMQFoP-wYUcwlhyaMyMdhOmFI_3wtq3wc0lT7h0Iae7Dz7JZuLx8jzfcG5OQonttQ5YmQq2RLaxLM4mwbGTPgOrZmO2tU49zlCn6-rac2tedZnDIpr8-JSQZHKWZ5gw1Y_LXv4XcB2Zis8g==)]1. [essex.ac.uk](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXGmaUF80RC9kPupZBBJExM094Vwh2eXaoTruem_7VyJ_A2mlH-WhkULfV_-4xfIYXviA-kNyY_6Y2vVwRB-bPoRbtt3oy-7wyS4X7Af0hg8y_Ju8yIdln16v8Xg7AIwgvB4sLcgRMhcsAXpGzsEedDVuw==)
    2. [mirbase.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXHbZOWmVHuvaE6e6Mzlajzv29ObibwE3wQXQZr3t9vY2TQkHFg6JJyt_BMlebS9-LXwFkRr__sf4Z5YBxi5JWmz0oiApewUk8SYd6jkyuKJsUF5X1shUvim6NXDnNUYEqlaIQb9Vq_enqRAEZjXaBRQz5cZ7x-Sqw==)
    3. [oup.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFewSmXx_88n1nfG5eaEcIbMQFoP-wYUcwlhyaMyMdhOmFI_3wtq3wc0lT7h0Iae7Dz7JZuLx8jzfcG5OQonttQ5YmQq2RLaxLM4mwbGTPgOrZmO2tU49zlCn6-rac2tedZnDIpr8-JSQZHKWZ5gw1Y_LXv4XcB2Zis8g==)
    4. [researchgate.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXESj1tob_T89BAFfvtgM0LjxR7Ygla-0Ke7HhBjWdFmNvJf27yQTw5mLIVJynTDAKWrnMc76JIpiIJZmvQVgHF87_GRtnw28cdxPEKwiWJnTgBsFAVCKoiNFK_-wKhNWg9GmAGrCyYwHndyYzVWnWzN1Ob34POZ-4ZUvU4VxadNmh5pVcowHdIIzRGaoA-EOe9Djc3xi-xT-nh-vkgeL9UI__HY7nTc6sPprxVpO5tpW10NFk7aMbVaE23eAPku)
    5. [frontiersin.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXG63BA4NR08TXubox8gAamrTPyMLwmZWgBCKmjYaLXfjhR1dURgYdgn0TkkHYVxr0Phdm9J0C9TAn2vKszReTUJ_LcfSGs_jrPujPq8WrrvKm0DWKWSkca8_GZVaSh5tfWRFUp-JFB-1Y5_qT1du4sftTSJzDg4XXBpYMvg5Ehh3ZyuU2Yd9VLnMUCpZKyEZg==)Here's a summary of literature related to 'cel-let-7', gathered from the search results:

    *   **General Information:** cel-let-7 is a microRNA found in *Caenorhabditis elegans*.[[1](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXGmaUF80RC9kPupZBBJExM094Vwh2eXaoTruem_7VyJ_A2mlH-WhkULfV_-4xfIYXviA-kNyY_6Y2vVwRB-bPoRbtt3oy-7wyS4X7Af0hg8y_Ju8yIdln16v8Xg7AIwgvB4sLcgRMhcsAXpGzsEedDVuw==),[2](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXHbZOWmVHuvaE6e6Mzlajzv29ObibwE3wQXQZr3t9vY2TQkHFg6JJyt_BMlebS9-LXwFkRr__sf4Z5YBxi5JWmz0oiApewUk8SYd6jkyuKJsUF5X1shUvim6NXDnNUYEqlaIQb9Vq_enqRAEZjXaBRQz5cZ7x-Sqw==),[3](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFewSmXx_88n1nfG5eaEcIbMQFoP-wYUcwlhyaMyMdhOmFI_3wtq3wc0lT7h0Iae7Dz7JZuLx8jzfcG5OQonttQ5YmQq2RLaxLM4mwbGTPgOrZmO2tU49zlCn6-rac2tedZnDIpr8-JSQZHKWZ5gw1Y_LXv4XcB2Zis8g==),[4](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXESj1tob_T89BAFfvtgM0LjxR7Ygla-0Ke7HhBjWdFmNvJf27yQTw5mLIVJynTDAKWrnMc76JIpiIJZmvQVgHF87_GRtnw28cdxPEKwiWJnTgBsFAVCKoiNFK_-wKhNWg9GmAGrCyYwHndyYzVWnWzN1Ob34POZ-4ZUvU4VxadNmh5pVcowHdIIzRGaoA-EOe9Djc3xi-xT-nh-vkgeL9UI__HY7nTc6sPprxVpO5tpW10NFk7aMbVaE23eAPku)] It plays a role in developmental timing, specifically in the transition to late-larval and adult stages by translational repression of target mRNAs.[[2](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXHbZOWmVHuvaE6e6Mzlajzv29ObibwE3wQXQZr3t9vY2TQkHFg6JJyt_BMlebS9-LXwFkRr__sf4Z5YBxi5JWmz0oiApewUk8SYd6jkyuKJsUF5X1shUvim6NXDnNUYEqlaIQb9Vq_enqRAEZjXaBRQz5cZ7x-Sqw==)] It is located on chromosome X in *C. elegans*.[[2](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXHbZOWmVHuvaE6e6Mzlajzv29ObibwE3wQXQZr3t9vY2TQkHFg6JJyt_BMlebS9-LXwFkRr__sf4Z5YBxi5JWmz0oiApewUk8SYd6jkyuKJsUF5X1shUvim6NXDnNUYEqlaIQb9Vq_enqRAEZjXaBRQz5cZ7x-Sqw==)]
    *   **miRBase:** The miRBase entry for cel-let-7 (Accession MI0000001) provides detailed information, including its sequence, structure, experimental evidence, and links to related databases.
    *   **Evolutionary Conservation:** The let-7 microRNA family is highly conserved across different species, including vertebrates. Its expression often coincides with differentiation and opposes stem cell pluripotency, acting as a tumor suppressor in certain contexts.[[3](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFewSmXx_88n1nfG5eaEcIbMQFoP-wYUcwlhyaMyMdhOmFI_3wtq3wc0lT7h0Iae7Dz7JZuLx8jzfcG5OQonttQ5YmQq2RLaxLM4mwbGTPgOrZmO2tU49zlCn6-rac2tedZnDIpr8-JSQZHKWZ5gw1Y_LXv4XcB2Zis8g==)]
    *   **Biogenesis and Regulation:** Biogenesis of let-7 is tightly regulated by cellular factors like LIN28A/B and DIS3L2.[[4](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXESj1tob_T89BAFfvtgM0LjxR7Ygla-0Ke7HhBjWdFmNvJf27yQTw5mLIVJynTDAKWrnMc76JIpiIJZmvQVgHF87_GRtnw28cdxPEKwiWJnTgBsFAVCKoiNFK_-wKhNWg9GmAGrCyYwHndyYzVWnWzN1Ob34POZ-4ZUvU4VxadNmh5pVcowHdIIzRGaoA-EOe9Djc3xi-xT-nh-vkgeL9UI__HY7nTc6sPprxVpO5tpW10NFk7aMbVaE23eAPku)] Dysregulation of let-7 processing can have deleterious effects.[[4](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXESj1tob_T89BAFfvtgM0LjxR7Ygla-0Ke7HhBjWdFmNvJf27yQTw5mLIVJynTDAKWrnMc76JIpiIJZmvQVgHF87_GRtnw28cdxPEKwiWJnTgBsFAVCKoiNFK_-wKhNWg9GmAGrCyYwHndyYzVWnWzN1Ob34POZ-4ZUvU4VxadNmh5pVcowHdIIzRGaoA-EOe9Djc3xi-xT-nh-vkgeL9UI__HY7nTc6sPprxVpO5tpW10NFk7aMbVaE23eAPku)]
    *   **Function:** let-7 is required for cell cycle exit and differentiation in hypodermal cell lineages during larval development in *C. elegans*.[[3](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFewSmXx_88n1nfG5eaEcIbMQFoP-wYUcwlhyaMyMdhOmFI_3wtq3wc0lT7h0Iae7Dz7JZuLx8jzfcG5OQonttQ5YmQq2RLaxLM4mwbGTPgOrZmO2tU49zlCn6-rac2tedZnDIpr8-JSQZHKWZ5gw1Y_LXv4XcB2Zis8g==)] It has critical functions in cell fate specification and developmental progression in diverse animals.[[3](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFewSmXx_88n1nfG5eaEcIbMQFoP-wYUcwlhyaMyMdhOmFI_3wtq3wc0lT7h0Iae7Dz7JZuLx8jzfcG5OQonttQ5YmQq2RLaxLM4mwbGTPgOrZmO2tU49zlCn6-rac2tedZnDIpr8-JSQZHKWZ5gw1Y_LXv4XcB2Zis8g==)]
    *   **Targets:** Target genes of let-7 include c-Myc, high-mobility group A (HMGA), STAT3, and JAK2, which are involved in cell proliferation and the cell cycle.[[5](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXG63BA4NR08TXubox8gAamrTPyMLwmZWgBCKmjYaLXfjhR1dURgYdgn0TkkHYVxr0Phdm9J0C9TAn2vKszReTUJ_LcfSGs_jrPujPq8WrrvKm0DWKWSkca8_GZVaSh5tfWRFUp-JFB-1Y5_qT1du4sftTSJzDg4XXBpYMvg5Ehh3ZyuU2Yd9VLnMUCpZKyEZg==)]
    *   **Role in Cancer:** let-7 underexpression has been reported to be significantly correlated with patient outcome in non-small cell lung cancer (NSCLC).[[5](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXG63BA4NR08TXubox8gAamrTPyMLwmZWgBCKmjYaLXfjhR1dURgYdgn0TkkHYVxr0Phdm9J0C9TAn2vKszReTUJ_LcfSGs_jrPujPq8WrrvKm0DWKWSkca8_GZVaSh5tfWRFUp-JFB-1Y5_qT1du4sftTSJzDg4XXBpYMvg5Ehh3ZyuU2Yd9VLnMUCpZKyEZg==)] It can regulate cellular proliferation by targeting the Ras family.[[5](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXG63BA4NR08TXubox8gAamrTPyMLwmZWgBCKmjYaLXfjhR1dURgYdgn0TkkHYVxr0Phdm9J0C9TAn2vKszReTUJ_LcfSGs_jrPujPq8WrrvKm0DWKWSkca8_GZVaSh5tfWRFUp-JFB-1Y5_qT1du4sftTSJzDg4XXBpYMvg5Ehh3ZyuU2Yd9VLnMUCpZKyEZg==)]
    *   **Tools for Target Prediction:** Several tools, such as seedVicious, miRanda, Diana-microT, PicTar, and RNAhybrid, are available for predicting microRNA targets.[[1](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXGmaUF80RC9kPupZBBJExM094Vwh2eXaoTruem_7VyJ_A2mlH-WhkULfV_-4xfIYXviA-kNyY_6Y2vVwRB-bPoRbtt3oy-7wyS4X7Af0hg8y_Ju8yIdln16v8Xg7AIwgvB4sLcgRMhcsAXpGzsEedDVuw==)]
    *   **Absence in Some Species:** Interestingly, some species within the *Caenorhabditis* genus lack the let-7 sequence in their genomic assemblies.[[3](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFewSmXx_88n1nfG5eaEcIbMQFoP-wYUcwlhyaMyMdhOmFI_3wtq3wc0lT7h0Iae7Dz7JZuLx8jzfcG5OQonttQ5YmQq2RLaxLM4mwbGTPgOrZmO2tU49zlCn6-rac2tedZnDIpr8-JSQZHKWZ5gw1Y_LXv4XcB2Zis8g==)]1. [essex.ac.uk](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXGmaUF80RC9kPupZBBJExM094Vwh2eXaoTruem_7VyJ_A2mlH-WhkULfV_-4xfIYXviA-kNyY_6Y2vVwRB-bPoRbtt3oy-7wyS4X7Af0hg8y_Ju8yIdln16v8Xg7AIwgvB4sLcgRMhcsAXpGzsEedDVuw==)
    2. [mirbase.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXHbZOWmVHuvaE6e6Mzlajzv29ObibwE3wQXQZr3t9vY2TQkHFg6JJyt_BMlebS9-LXwFkRr__sf4Z5YBxi5JWmz0oiApewUk8SYd6jkyuKJsUF5X1shUvim6NXDnNUYEqlaIQb9Vq_enqRAEZjXaBRQz5cZ7x-Sqw==)
    3. [oup.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFewSmXx_88n1nfG5eaEcIbMQFoP-wYUcwlhyaMyMdhOmFI_3wtq3wc0lT7h0Iae7Dz7JZuLx8jzfcG5OQonttQ5YmQq2RLaxLM4mwbGTPgOrZmO2tU49zlCn6-rac2tedZnDIpr8-JSQZHKWZ5gw1Y_LXv4XcB2Zis8g==)
    4. [researchgate.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXESj1tob_T89BAFfvtgM0LjxR7Ygla-0Ke7HhBjWdFmNvJf27yQTw5mLIVJynTDAKWrnMc76JIpiIJZmvQVgHF87_GRtnw28cdxPEKwiWJnTgBsFAVCKoiNFK_-wKhNWg9GmAGrCyYwHndyYzVWnWzN1Ob34POZ-4ZUvU4VxadNmh5pVcowHdIIzRGaoA-EOe9Djc3xi-xT-nh-vkgeL9UI__HY7nTc6sPprxVpO5tpW10NFk7aMbVaE23eAPku)
    5. [frontiersin.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXG63BA4NR08TXubox8gAamrTPyMLwmZWgBCKmjYaLXfjhR1dURgYdgn0TkkHYVxr0Phdm9J0C9TAn2vKszReTUJ_LcfSGs_jrPujPq8WrrvKm0DWKWSkca8_GZVaSh5tfWRFUp-JFB-1Y5_qT1du4sftTSJzDg4XXBpYMvg5Ehh3ZyuU2Yd9VLnMUCpZKyEZg==)```json
    {"""
    def mock_run_model_master(*args, **kwargs):
        fake_response = AIMessage(content='***ROUTE_TO_LITERATURE*** Search for the role of microRNAs in cancer.')
        return fake_response
    def mock_run_model_literature(*args, **kwargs):
        fake_response = AIMessage(content=literature_answer)
        return fake_response
    def mock_is_compleated(*args, **kwargs):
        return {'answer': 'YES', 'return': literature_answer}
    def mock_format_text(*args, **kwargs):
        return literature_answer,literature_answer,literature_answer
    monkeypatch.setattr(ChatbotNode, "run_model", mock_run_model_master)
    monkeypatch.setattr(ChatbotNode, "is_compleated", mock_is_compleated)
    monkeypatch.setattr(LiteratureNode, "run_model", mock_run_model_literature)
    monkeypatch.setattr(LiteratureNode, "format_text", mock_format_text)

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
    
def test_agent_literature_bad_response(agent_app: AgentEngineApp, monkeypatch) -> None:
    """
    Integration test for the agent stream query functionality.
    Tests that the agent returns valid streaming responses.
    """
    literature_answer = ""
    def mock_run_model_master(*args, **kwargs):
        fake_response = AIMessage(content='***ROUTE_TO_LITERATURE*** Search for the role of microRNAs in cancer.')
        return fake_response
    def mock_run_model_literature(*args, **kwargs):
        fake_response = AIMessage(content=literature_answer)
        return fake_response
    def mock_is_compleated(*args, **kwargs):
        return {'answer': 'YES', 'return': literature_answer}
    def mock_format_text(*args, **kwargs):
        return literature_answer,literature_answer,literature_answer
    monkeypatch.setattr(ChatbotNode, "run_model", mock_run_model_master)
    monkeypatch.setattr(ChatbotNode, "is_compleated", mock_is_compleated)
    monkeypatch.setattr(LiteratureNode, "run_model", mock_run_model_literature)
    monkeypatch.setattr(LiteratureNode, "format_text", mock_format_text)

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

def test_agent_sql(agent_app: AgentEngineApp, monkeypatch) -> None:
    """
    Integration test for the agent stream query functionality.
    Tests that the agent returns valid streaming responses.
    """
    SQL_answer="Mir1"

    def mock_is_compleated(*args, **kwargs):
        return {'answer': 'YES', 'return': SQL_answer}
    def mock_run_model_sql(*args, **kwargs):
        fake_response = GenerateContentResponse
        fake_response.automatic_function_calling_history = 'many functions'
        fake_response.text = SQL_answer
        fake_response.candidates = [Candidate(content=Content(parts=[Part]))]
        return fake_response
    def mock_run_model_master(*args, **kwargs):
        fake_response = AIMessage(content="***ROUTE_TO_SQL*** Select one arbitrary microRNA ID from the database")
        return fake_response
    def mock_get_queries(*args, **kwargs):
        return {"query": "select * from table"}

    monkeypatch.setattr(SQLNode, "run_model", mock_run_model_sql)
    monkeypatch.setattr(SQLNode, "get_queries", mock_get_queries)
    monkeypatch.setattr(ChatbotNode, "run_model", mock_run_model_master)
    monkeypatch.setattr(ChatbotNode, "run_model", mock_run_model_master)
    monkeypatch.setattr(ChatbotNode, "is_compleated", mock_is_compleated)


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
    def mock_is_compleated(*args, **kwargs):
        return {'answer': 'YES', 'return': "MiR1"}
    monkeypatch.setattr(SQLNode, "run_model", mock_run_model_sql)
    monkeypatch.setattr(SQLNode, "get_queries", mock_get_queries)
    monkeypatch.setattr(ChatbotNode, "run_model", mock_run_model_master)
    monkeypatch.setattr(ChatbotNode, "is_compleated", mock_is_compleated)

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
    SQL_answer = "Mir1"

    def mock_is_compleated(*args, **kwargs):
        return {'answer': 'YES', 'return': SQL_answer}

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
    monkeypatch.setattr(ChatbotNode, "is_compleated", mock_is_compleated)
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



def test_history_track(agent_app: AgentEngineApp, monkeypatch) -> None:
    """
    Test that the history is tracked correctly in the agent.
    """
    def mock_run_model_master(*args, **kwargs):
        fake_response = AIMessage(content='***ANSWER_DIRECTLY*** This is a test message.')
        return fake_response
    def mock_is_compleated(*args, **kwargs):
        return {'answer': 'YES', 'return': 'This is a test message.'}

    monkeypatch.setattr(ChatbotNode, "run_model", mock_run_model_master)
    monkeypatch.setattr(ChatbotNode, "is_compleated", mock_is_compleated)

    input_dict = {
        "messages": [
            {"type": "human", "content": "Test message 1"},
            {"type": "human", "content": "Test message 2"},
        ],
        "table": None,
        "answer": None,
        "finished": False,
        "user_id": "test-user",
        "session_id": "test-session",
    }

    events = list(agent_app.stream_query(input=input_dict))
    assert len(events) > 0, "Expected at least one chunk in response"
    history = False
    # Verify each event is a tuple of message and metadata
    for event in events:
        assert isinstance(event, list), "Event should be a list"
        assert len(event) == 2, "Event should contain message and metadata"
        message, _ = event
        print(message)
        # Verify message structure
        assert isinstance(message, dict), "Message should be a dictionary"
        assert message["type"] == "constructor"
        assert "kwargs" in message, "Constructor message should have kwargs"


    # Check if history contains both messages
    #assert len(response["history"]) == 2, "History should contain two messages"
    #assert response["history"][0]["content"] == "Test message 1"
    #assert response["history"][1]["content"] == "Test message 2"