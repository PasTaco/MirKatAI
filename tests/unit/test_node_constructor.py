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
You can add your unit tests here.
This is where you test your business logic, including agent functionality,
data processing, and other core components of your application.
"""
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
    # change the path to the parent directory
    sys.path.append("../../tests")
    
if current_path.endswith("app"):
    sys.path.append("../tests")
## Test class node:
def test_create_node() -> None:
    """Check that the node is created correctly."""
    node_obj = node()
    assert node_obj is not None
    assert node_obj.llm is None
    assert node_obj.instructions is None
    assert node_obj.functions is None
    assert node_obj.welcome is None
    assert type(node_obj.client) == Client

def test_create_node_with_llm() -> None:
    """Check that the node is created correctly when setting a llm."""
    node_obj = node(llm="flash")
    assert node_obj is not None
    assert node_obj.llm == "flash"

def test_create_node_with_instructions() -> None:
    """Check that the node is created correctly when setting a llm and instructions."""
    node_obj = node(instructions="test")
    assert node_obj is not None
    assert node_obj.instructions == "test"

def test_create_node_with_functions() -> None:
    """Check that the node is created correctly when setting a llm, instructions and functions."""
    node_obj = node(functions=["test"])
    assert node_obj is not None
    assert node_obj.functions == ["test"]


def test_create_node_with_welcome() -> None:
    """Check that the node is created correctly when setting a llm, instructions and functions."""
    node_obj = node(welcome="hi")
    assert node_obj is not None
    assert node_obj.welcome == "hi"




### Test class ChatbotNode:

def test_chatbot_node() -> None:
    """Check that the node is created correctly."""
    chatbot = ChatbotNode(llm="gemini-1.5-flash")
    assert chatbot is not None
    assert chatbot.llm is not None
    assert chatbot.llm == "gemini-1.5-flash"
    assert chatbot.instructions is None
    assert chatbot.functions is None
    assert chatbot.welcome is None
    assert type(chatbot.client) == Client
    assert chatbot.llm_master is not None
    assert type(chatbot.llm_master) == ChatGoogleGenerativeAI


def test_chatbot_set_model() -> None:
    """Check that the node is created correctly."""
    chatbot = ChatbotNode(llm="gemini-1.5-flash")
    chatbot.set_model(model="gemini-2.0-flash")
    assert chatbot.llm is not None
    assert chatbot.llm == "gemini-1.5-flash"
    assert chatbot.llm_master is not None
    assert type(chatbot.llm_master) == ChatGoogleGenerativeAI
    assert chatbot.llm_master.model == 'models/gemini-2.0-flash'


def test_chatbot_run_model(monkeypatch) -> None:
    """Check that the node is a responsive llm node"""
    chatbot = ChatbotNode(llm="gemini-1.5-flash")
    monkeypatch.setattr(chatbot, "run_model", lambda *args, **kwargs: AIMessage(content="hi"))
    assert chatbot.llm_master is not None
    result = chatbot.run_model("hi")
    # check messages is AIMessage
    assert isinstance(result, AIMessage)
    assert result.content == "hi"


def test_chatbot_get_node_with_messages(monkeypatch) -> None:
    """Check that the node is a responsive llm node"""
    chatbot = ChatbotNode(llm="gemini-1.5-flash")
    monkeypatch.setattr(chatbot, "run_model", lambda *args, **kwargs: AIMessage(content="hi"))
    assert chatbot.llm_master is not None
    status = {
    "messages": [HumanMessage(content="Hi")], # <-- Empty list
    "table": None,
    "answer": "",
    "finished": False}
    result = chatbot.get_node(status)
    print(result)
    # check messages is AIMessage
    assert isinstance(result['messages'], AIMessage)
    assert isinstance(result['request'].content, str)
    assert result['request'].content == "hi"
    assert result['answer'] is None
    assert result['finished'] is False
    
def test_chatbot_get_node_with_messages_answer_directly(monkeypatch) -> None:
    """Check that the node is a responsive llm node"""
    chatbot = ChatbotNode(llm="gemini-1.5-flash")
    monkeypatch.setattr(chatbot, "run_model", lambda *args, **kwargs: AIMessage(content="***ANSWER_DIRECTLY***"))
    assert chatbot.llm_master is not None
    status = {
        "messages": [HumanMessage(content="Hi")],  # <-- Empty list
        "table": None,
        "answer": "",
        "finished": False}
    result = chatbot.get_node(status)
    # check messages is AIMessage
    assert isinstance(result['messages'], AIMessage)
    assert result['messages'].content == ""
    assert result['request'].content == "***ANSWER_DIRECTLY***"
    assert result['answer'] == None
    assert result['finished'] is False

#### Test class SQLNode:


def test_sql_node() -> None:
    """Check that the node is created correctly."""
    sql_node = SQLNode(instructions="you are a SQL expert", functions=[min, max])
    assert sql_node is not None
    assert sql_node.llm is None
    assert sql_node.welcome is None
    assert type(sql_node.client) == Client
    assert sql_node.chat is not None
    assert type(sql_node.chat) == Chat
    assert sql_node.instructions == "you are a SQL expert"
    assert sql_node.functions == [min, max]



def test_sql_without_messages() -> None:
    """Check that the node is a responsive llm node"""
    sql_node = SQLNode(instructions="you are a SQL expert", functions=[min, max])
    status = {
        "messages": [None],
        "request": None,
        "table": None,
        "answer": "",
        "finished": False}
    result = sql_node.get_node(status)
    assert result['messages'] == [None]


def test_sql_with_messages_str(monkeypatch) -> None:
    """Check that the node is a responsive llm node"""
    sql_node = SQLNode(instructions="you are a SQL expert", functions=[min, max])
    status = {
        "messages": [HumanMessage(content="Hi")],
        "request":  AIMessage(content="hi"),
        "table": None,
        "answer": "",
        "original_query": "Hi",
        "finished": False}
    def mock_run_model(*args, **kwargs):
        fake_response = GenerateContentResponse
        fake_response.automatic_function_calling_history= 'many functions'
        fake_response.text = "Risotto alla Milanese"
        return fake_response

    monkeypatch.setattr(sql_node, "run_model", 
                        mock_run_model)
                        
    monkeypatch.setattr(sql_node, "get_queries", 
                        lambda *args, **kwargs: {"query":"result"})
    
    result = sql_node.get_node(status)

    # check messages is AIMessage
    assert isinstance(result['messages'], AIMessage)
    assert result['request'].content == "Risotto alla Milanese"
    assert result['answer'] == AIMessage(content='Risotto alla Milanese')
    assert result['table'] == {"query":"result"}
    assert result['finished'] is False




def test_sql_with_messages_GenerateContentResponse(monkeypatch) -> None:
    """Check that the node is a responsive llm node"""
    sql_node = SQLNode(instructions="you are a SQL expert", functions=[min, max])
    status = {"messages":GenerateContentResponse, "request":"Run a query", "original_query":"question"}
    def mock_run_model(*args, **kwargs):
        fake_response = GenerateContentResponse
        fake_response.automatic_function_calling_history= 'many functions'
        fake_response.text = "Risotto alla Milanese"
        return fake_response

    monkeypatch.setattr(sql_node, "run_model", 
                        mock_run_model)
                        
    monkeypatch.setattr(sql_node, "get_queries", 
                        lambda *args, **kwargs: {"query":"result"})
    
    result = sql_node.get_node(status)

    # check messages is AIMessage
    assert isinstance(result['messages'], AIMessage)
    assert result['request'].content == "Risotto alla Milanese"
    assert result['answer'] == AIMessage(content='Risotto alla Milanese')
    assert result['table'] == {"query":"result"}
    assert result['finished'] is False


#### Test class PlotNode:



@pytest.mark.skip("Plot node must be fixed")
def test_plot_get_node(monkeypatch) -> None:
    """Check that the node is a responsive llm node"""
    plot_node = PlotNode(instructions="you are a plot expert", functions=[min, max])
    status = {"messages":"", 'request':AIMessage(content="Plot A=1, B=2"), "table":'data'}
    # see current directory 

    plot = pickle.load(open("tests/dummy_files/plot.pkl", "rb"))
    def mock_run_model(*args, **kwargs):
        fake_response = GenerateContentResponse
        fake_response.automatic_function_calling_history= 'many functions'
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


#### Test class LiteratureNode:

def test_literature_node() -> None:
    """Check that the node is created correctly."""
    literature_node = LiteratureNode(instructions="you are a literature expert", functions=[min, max])
    assert literature_node is not None
    assert literature_node.llm is None
    assert literature_node.welcome is None
    assert type(literature_node.client) == Client
    assert literature_node.instructions == "you are a literature expert"
    assert literature_node.functions == [min, max]
    assert literature_node.config_with_search is not None
    assert type(literature_node.config_with_search) == GenerateContentConfig

def test_literature_get_node(monkeypatch) -> None:
    """Check that the node is a responsive llm node"""
    literature_node = LiteratureNode(instructions="you are a literature expert", functions=[min, max])
    status = {"messages":"","request":AIMessage(content="hi")}

    monkeypatch.setattr(literature_node, "run_model", 
                            lambda *args, **kwargs: "Research done")
    monkeypatch.setattr(literature_node, "format_text",
                        lambda *args, **kwargs: ("# markdown text","bibliography", "queries"))
    result = literature_node.get_node(status)
    # check messages is AIMessage
    assert result['messages'] == AIMessage(content='# markdown textbibliography', additional_kwargs={}, response_metadata={})


def test_sql_get_node(monkeypatch) -> None:
    """Check that the node is a responsive llm node"""
    sql_node = SQLNode(instructions="you are a SQL expert,", functions=[min, max])
    status = {"messages":"",
              "request":AIMessage(content="Get one microRNA from the databse"),
              "original_query":[{'type': 'text', 'text': 'tell me one microRNA from the database'}]
              }
    def mock_run_model(*args, **kwargs):
        fake_response = GenerateContentResponse
        fake_response.automatic_function_calling_history= 'many functions'
        fake_response.text = "Mir1"
        fake_response.candidates = [Candidate(content=Content(parts=[Part]))]
        return fake_response
    def mock_get_queries(*args, **kwargs):
        return {"query":"select * from table"}
    monkeypatch.setattr(sql_node, "run_model", mock_run_model)
    monkeypatch.setattr(sql_node, "get_queries", mock_get_queries)
    result = sql_node.get_node(status)
    # check messages is AIMessage
    assert "Mir1" in result['request'].content
    assert result["answer_source"] == "SQL_NODE"
    assert result["answer"].content == "Mir1"



