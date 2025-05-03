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

from app.mirkat.node_constructor import (
    node,
    HumanNode, 
    ChatbotNode,
    SQLNode,
    PlotNode,
    LiteratureNode

)

import app.mirkat.plot_functions as plot_functions

from langchain_core.messages import ( # Grouped message types
    AIMessage
)
from langchain_google_genai import ChatGoogleGenerativeAI

from google.genai.client import Client
from google.genai.chats import Chat

from google.genai.types import GenerateContentResponse

import pickle

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



### test HumanNode:

def test_human_node() -> None:
    """Check that the node is created correctly."""
    human_node = HumanNode()
    assert human_node is not None
    assert human_node.llm is None
    assert human_node.instructions is None
    assert human_node.functions is None
    assert human_node.welcome is None
    assert type(human_node.client) == Client


def test_human_get_node_text_message() -> None:
    """Check that the node correctly handle response"""
    human_node = HumanNode()
    state = {"messages":"hi","answer":None}
    result = human_node.get_node(state)
    # check messages is AIMessage
    assert result['messages'] == "hi"
    assert result['answer'] is None


def test_human_get_node_ai_message_no_answer() -> None:
    """Check that the node correctly handle response"""
    human_node = HumanNode()
    state = {"messages":AIMessage(content="hi"),"answer":None}
    result = human_node.get_node(state)
    # check messages is AIMessage
    assert result['messages'].content == "hi"
    assert type(result['messages']) == AIMessage
    assert result['answer'] is None


def test_human_get_node_GenerateContentResponse_no_answer() -> None:
    """Check that the node correctly handle response"""
    human_node = HumanNode()
    state = {"messages":GenerateContentResponse(),"answer":None}
    result = human_node.get_node(state)
    # check messages is AIMessage
    assert result['messages'].text is None
    assert type(result['messages']) == GenerateContentResponse
    assert result['answer'] is None



def test_human_get_node_ai_message_with_answer() -> None:
    """Check that the node correctly handle response"""
    human_node = HumanNode()
    state = {"messages":AIMessage(content="hi"),"answer":"hi"}
    result = human_node.get_node(state)
    # check messages is AIMessage
    assert result['messages'].content == "hi"
    assert type(result['messages']) == AIMessage
    assert result['answer'] is None


def test_human_get_node_GenerateContentResponse_with_answer() -> None:
    """Check that the node correctly handle response"""
    human_node = HumanNode()
    state = {"messages":GenerateContentResponse(),"answer":"hi"}
    result = human_node.get_node(state)
    # check messages is AIMessage
    assert result['messages'].text is None
    assert type(result['messages']) == GenerateContentResponse
    assert result['answer'] is None

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
    status = {"messages":"hi"}
    result = chatbot.get_node(status)
    # check messages is AIMessage
    assert isinstance(result['messages'], AIMessage)
    assert result['messages'].content == "hi"
    assert result['answer'] is None
    assert result['finished'] is False
    
def test_chatbot_get_node_with_messages_answer_directly(monkeypatch) -> None:
    """Check that the node is a responsive llm node"""
    chatbot = ChatbotNode(llm="gemini-1.5-flash")
    monkeypatch.setattr(chatbot, "run_model", lambda *args, **kwargs: AIMessage(content="***ANSWER_DIRECTLY***"))
    assert chatbot.llm_master is not None
    status = {"messages":"hi"}
    result = chatbot.get_node(status)
    # check messages is AIMessage
    assert isinstance(result['messages'], AIMessage)
    assert result['messages'].content == ""
    assert result['answer'] == ""
    assert result['finished'] is False

def test_chatbot_get_node_without_messages() -> None:
    """Check that the node is a responsive llm node"""
    chatbot = ChatbotNode(llm="gemini-1.5-flash")
    status = {"messages":None}
    result = chatbot.get_node(status)
    # check messages is AIMessage
    assert isinstance(result['messages'], AIMessage)
    assert result['messages'].content == "Welcome to the chatbot! How can I assist you today?"
    assert result['answer'] is None
    assert result['finished'] is False


def test_chatbot_get_node_without_messages_with_welcome() -> None:
    """Check that the node is a responsive llm node"""
    chatbot = ChatbotNode(llm="gemini-1.5-flash", welcome="Buongiorno dottore!")
    status = {"messages":None}
    result = chatbot.get_node(status)
    # check messages is AIMessage
    assert isinstance(result['messages'], AIMessage)
    assert result['messages'].content == "Buongiorno dottore!"
    assert result['answer'] is None
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
    status = {"messages":None}
    result = sql_node.get_node(status)
    assert result['messages'] is None


def test_sql_with_messages_str(monkeypatch) -> None:
    """Check that the node is a responsive llm node"""
    sql_node = SQLNode(instructions="you are a SQL expert", functions=[min, max])
    status = {"messages":"hi"}
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
    assert result['messages'].content == "This was the answer from SQL node, please format and give to the user: Risotto alla Milanese"
    assert result['answer'] == ""
    assert result['table'] == {"query":"result"}
    assert result['finished'] is False




def test_sql_with_messages_GenerateContentResponse(monkeypatch) -> None:
    """Check that the node is a responsive llm node"""
    sql_node = SQLNode(instructions="you are a SQL expert", functions=[min, max])
    status = {"messages":GenerateContentResponse}
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
    assert result['messages'].content == "This was the answer from SQL node, please format and give to the user: Risotto alla Milanese"
    assert result['answer'] == ""
    assert result['table'] == {"query":"result"}
    assert result['finished'] is False


#### Test class PlotNode:

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


from google.genai.types import Candidate, Content,Part, GenerateContentConfig
def test_plot_get_node(monkeypatch) -> None:
    """Check that the node is a responsive llm node"""
    plot_node = PlotNode(instructions="you are a plot expert", functions=[min, max])
    status = {"messages":AIMessage(content="hi"), "table":'data'}
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

from types import SimpleNamespace
def test_literature_get_node(monkeypatch) -> None:
    """Check that the node is a responsive llm node"""
    literature_node = LiteratureNode(instructions="you are a literature expert", functions=[min, max])
    status = {"messages":None}

    monkeypatch.setattr(literature_node, "run_model", 
                            lambda *args, **kwargs: "Research done")
    monkeypatch.setattr(literature_node, "format_text",
                        lambda *args, **kwargs: ("# markdown text","bibliography", "queries"))
    result = literature_node.get_node(status)
    # check messages is AIMessage
    assert result['messages'] == "# markdown text"
    assert result['answer'] == "# markdown text"
    assert result['bibliography'] == "bibliography"
    assert result['research_queries'] == "queries"
