import pytest
from app.mirkat.node_plot import PlotNode
import app.nodes as nodes
import app.mirkat.plot_functions as plot_functions

from langchain_core.messages import (  # Grouped message types
    AIMessage, HumanMessage
)

from google.genai.client import Client
from google.genai.chats import Chat

from google.genai.types import GenerateContentResponse
from google.genai.types import Candidate, Content,Part
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

def test_plot_node_initialization() -> None:
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
    plot_node = nodes.plot_node
    # On this architecture, messages always comes as "" since we try to skip
    # the extra plotting and to make the models request lighter.
    # The request have the requested action from the node
    # Table is empty, it will be filled
    # Answer will be filled with the result of this node
    # Original_query is filled out during the master node.
    #
    plot_response = r'''{
              "caption" : "Barplot of values a=1 and b=3",
              "code" : "import matplotlib.pyplot as plt\n\nvalues = [1, 3]\nlabels = ['a', 'b']\n\nfig, ax = plt.subplots()\nax.bar(labels, values)\nax.set_xlabel('Variables')\nax.set_ylabel('Values')\nax.set_title('Barplot of a and b')\n\nfigure = fig",
              "notes" : "The code generates a barplot with 'a' and 'b' on the x-axis and their corresponding values (1 and 3) on the y-axis."
            }'''
    status = {
        "messages": "",
        "request":  AIMessage(content="***ROUTE_TO_PLOT_NODE*** Plot A=1, B=2"),
        "table": "data",
        "answer": "",
        "original_query": "Plot A=1, B=2",
        "finished": False
    }
    print(current_path)
    try:
        plot = pickle.load(open("tests/dummy_files/plot.pkl", "rb"))
        response_plot = pickle.load(open("test/dummy_files/plot_result.pkl", "rb"))
    except FileNotFoundError as e:
        plot = pickle.load(open("../dummy_files/plot.pkl", "rb"))
        response_plot = pickle.load(open("../dummy_files/plot_result.pkl", "rb"))


    def mock_run_model(*args, **kwargs):
        fake_response = response_plot
        return fake_response

    monkeypatch.setattr(plot_node, "run_model",
                        mock_run_model)
    monkeypatch.setattr(plot_functions.PlotFunctons,
                        "handle_response",
                        lambda *args, **kwargs: plot)
    result = plot_node.get_node(status)
    print(result)
    # check messages is AIMessage
    result_message = result['answer'].content

    assert "binary_image" in result_message
    assert result['table'] == 'data'
