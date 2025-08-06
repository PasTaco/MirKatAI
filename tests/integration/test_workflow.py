import os, sys

from app.agent import LITERATURE_NODE

current_path = os.path.dirname(os.path.abspath(__file__))
if current_path.endswith("app"):
    sys.path.append("../tests")

import pytest
from app.agent import agent
from unittest.mock import patch, MagicMock



current_path = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def workflow():
    """Fixture to create and set up the agent workflow."""
    return agent



def test_agent_direct_answer(workflow) -> None:
    state = {
        "messages":["who are you?"]
    }
    result = workflow.invoke(state)
    response = result.get("messages", "").content
    source = result.get("answer_source", "")
    history = result.get("history", [])
    trys = result.get("trys", 0)
    assert isinstance(response, str) # check response is a string
    assert response.startswith("****FINAL_RESPONSE**** ") # check response starts with expected text
    assert source == "ChatbotNode" # check source is chatbot_node
    assert isinstance(history, list) # check history is a list
    assert len(history) > 0 # check history is not empty
    assert "***ANSWER_DIRECTLY***" in history[1].content
    assert trys == 2 # check trys is 2, as it should be a direct answer



def test_agent_wrong_question(workflow) -> None:
    state = {
        "messages":["How to make a proper carbonara?"]
    }
    result = workflow.invoke(state)
    response = result.get("messages", "").content
    source = result.get("answer_source", "")
    history = result.get("history", [])
    trys = result.get("trys", 0)
    assert isinstance(response, str) # check response is a string
    assert response.startswith("****FINAL_RESPONSE**** ") # check response starts with expected text
    assert source == "ChatbotNode" # check source is chatbot_node
    assert isinstance(history, list) # check history is a list
    assert len(history) > 0 # check history is not empty
    assert "***ANSWER_DIRECTLY***" in history[1].content
    assert trys == 2 # check trys is 2, as it should be a direct answer


def test_agent_sql_query(workflow) -> None:
    state = {
        "messages":["From the SQL database, how many targets has hsa-miR-1-p5?"]
    }
    result = workflow.invoke(state)
    response = result.get("messages", "").content
    history = result.get("history", [])
    assert isinstance(response, str) # check response is a string
    assert response.startswith("****FINAL_RESPONSE**** ") # check response starts with expected text
    assert "1021" in response # check response contains the expected number of targets
    assert isinstance(history, list) # check history is a list
    assert len(history) > 0 # check history is not empty
    assert "***ROUTE_TO_SQL***" in history[1].content



def test_agent_literature_query(workflow) -> None:
    state = {
        "messages":["Search in the literature if hsa-miR-1-p5 is involved in cancer."]
    }
    result = workflow.invoke(state)
    response = result.get("messages", "").content
    history = result.get("history", [])
    assert isinstance(response, str) # check response is a string
    assert response.startswith("****FINAL_RESPONSE**** ") # check response starts with expected text
    assert isinstance(history, list) # check history is a list
    assert len(history) > 0 # check history is not empty
    assert "***ROUTE_TO_LITERATURE***" in history[1].content


def test_agent_plot_query(workflow) -> None:
    state = {
        "messages":["Plot a barplot with miR1=5 and miR2=10."]
    }
    result = workflow.invoke(state)
    response = result.get("messages", "").content
    history = result.get("history", [])
    assert isinstance(response, str) # check response is a string
    assert response.startswith("****FINAL_RESPONSE**** ") # check response starts with expected text
    assert "<image>" in response # check if the response contains an image
    assert isinstance(history, list) # check history is a list
    assert len(history) > 0 # check history is not empty
    assert "***PLOT***" in history[1].content
