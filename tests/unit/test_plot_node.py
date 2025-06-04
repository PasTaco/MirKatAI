import pytest

from app.mirkat.node_constructor import (
    node,
    HumanNode
)
from app.mirkat.node_chatbot import ChatbotNode
from app.mirkat.node_sql import SQLNode
from app.mirkat.node_plot import PlotNode
from app.mirkat.node_literature import LiteratureNode
import app.mirkat.plot_functions as plot_functions

from langchain_core.messages import (  # Grouped message types
    AIMessage, HumanMessage
)
from langchain_google_genai import ChatGoogleGenerativeAI

from google.genai.client import Client
from google.genai.chats import Chat

from google.genai.types import GenerateContentResponse
from google.genai.types import Candidate, Content,Part, GenerateContentConfig
import pickle
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
if current_path.endswith("tests/unit"):
    # change path to tests directory
    sys.path.append("../../app")
    sys.path.append("../../tests")
if current_path.endswith("app"):
    sys.path.append("../tests")

def test_plot_node() -> None:
    """Check that the node is created correctly."""
    plot_node = PlotNode(instructions="you are a plot expert", functions=[min, max])
    assert plot_node is not None
    assert plot_node.llm is None
    assert plot_node.welcome is None
    assert type(plot_node.client) == Client
    assert plot_node.model is not None
    assert type(plot_node.model) == Chat
    assert plot_node.instructions == "you are a plot expert"
    assert plot_node.functions == [min, max]


def test_plot_get_node(monkeypatch) -> None:
    """Check that the node is a responsive llm node"""
    plot_node = PlotNode(instructions="you are a plot expert", functions=[min, max])
    # On this architecture, messages always comes as "" since we try to skip
    # the extra plotting and to make the models request lighter.
    # The request have the requested action from the node
    # Table is empty, it will be filled
    # Answer will be filled with the result of this node
    # Original_query is filled out during the master node.
    #
    status = {
        "messages": "",
        "request":  AIMessage(content="***ROUTE_TO_PLOT_NODE*** Plot A=1, B=2"),
        "table": None,
        "answer": "",
        "original_query": "Plot A=1, B=2",
        "finished": False
    }
    print(current_path)
    try:
        plot = pickle.load(open("tests/dummy_files/plot.pkl", "rb"))
    except FileNotFoundError as e:
        plot = pickle.load(open("../dummy_files/plot.pkl", "rb"))

    def mock_run_model(*args, **kwargs):
        fake_response = GenerateContentResponse
        fake_response.automatic_function_calling_history = 'many functions'
        fake_response.text = "Risotto alla Milanese"
        fake_response.candidates = [Candidate(content=Content(parts=[Part]))]
        return fake_response

    monkeypatch.setattr(plot_node, "run_model",
                        mock_run_model)
    monkeypatch.setattr(plot_functions.PlotFunctons,
                        "handle_response",
                        lambda *args, **kwargs: plot)
    result = plot_node.get_node(status)
    print(result)
    # check messages is AIMessage
    result_message = result['messages'].content

    assert "binary_image" in result_message
    assert result['table'] == 'data'
