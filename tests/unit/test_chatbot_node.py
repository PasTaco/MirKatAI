import pytest
from app.mirkat.node_chatbot import ChatbotNode
import app.nodes as nodes

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


def test_plot_get_node_complete(monkeypatch) -> None:
    """Check that the node is a responsive llm node"""
    chatbot_node = nodes.master_node
    status = { 'messages':"",
                'answer': "The code generates a barplot with 'a' and 'b' on the x-axis and their corresponding values (1 and 3) on the y-axis.",
               'original_query': 'Barplot of values a=1 and b=3',
               'request': AIMessage(content='Barplot of values a=1 and b=3 <image_save>file</image_save>'),
               'answer_source': 'PlotNode',
               'trys': 1,
               'try_limit': 5,
               'history': [HumanMessage(content='Barplot of values a=1 and b=3',
                                        additional_kwargs={},
                                        response_metadata={},
                                        id='344c2e75-a150-4a19-9a11-bfd2f8863691'),
                           AIMessage(content='***PLOT*** Plot A=1, B=3',
                                     additional_kwargs={},
                                     response_metadata={},
                                     id='84e6698f-9e1b-43cf-a323-c70a066420d6'),
                           AIMessage(content='Barplot of values a=1 and b=3 <image_save>file</image_save> using code: Placeholder ',
                                     additional_kwargs={},
                                     response_metadata={},
                                     id='cad2e8d8-c9b4-45c9-9e1f-e194a2ffe791')
                           ]
               }
    print(current_path)
    try:
        complete = pickle.load(open("tests/dummy_files/completes_plot_result.pkl", "rb"))
    except FileNotFoundError as e:
        complete = pickle.load(open("../dummy_files/completes_plot_result.pkl", "rb"))

    monkeypatch.setattr(chatbot_node, "run_model_for_compleatness",
                        lambda *args, **kwargs: complete)

    result = chatbot_node.get_node(status)
    print(result)
    # check messages is AIMessage
    result_message = result['answer'].content
    assert "image_save" in result_message

    assert result['table'] == 'data'

