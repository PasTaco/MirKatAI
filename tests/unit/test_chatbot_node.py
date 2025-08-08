import pytest
from app.mirkat.node_chatbot import ChatbotNode
import app.nodes as nodes
import json
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
dummy_path = current_path.split('MirKatAI')[0]
dummy_path = dummy_path + "MirKatAI/tests/dummy_files/"

# TODO: FIx plot node json output.
@pytest.mark.skip("Plot node needs to be attended")
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
               'history': ['Barplot of values a=1 and b=3',
                           'Barplot of values a=1 and b=3 <image_save>file</image_save> using code: Placeholder '
                           ]
               }
    print(current_path)
    complete = pickle.load(open(dummy_path+"completes_plot_result.pkl", "rb"))


    monkeypatch.setattr(chatbot_node, "run_model_for_compleatness",
                        lambda *args, **kwargs: complete)

    result = chatbot_node.get_node(status)
    print(result)
    # check messages is AIMessage
    result_message = result['messages'].content
    result_media = result['']
    assert "image" in result_message


def test_extract_json_from_markdown(monkeypatch):
    """
    Test to check that the extract_json can deal with the jsons
    """
    chatbot_node = nodes.master_node
    is_complete_content = '{"answer": "YES", "return": "The tissues in which miR-1, miR-199a-5p, and miR-181a-5p are expressed are muscle, brain, and lung.", "media": null}'
    r = chatbot_node.extract_json_from_markdown(content=is_complete_content)
    print(r)
    assert r
    assert r == is_complete_content
    res = json.loads(r)
    assert res

def test_extract_json_from_json_tag(monkeypatch):
    """This test will evaluate for the old way of tag jsons"""
    content = """```json
            {
              "answer": "YES",
              "return": "The code generates a Venn diagram showing the overlap between two microRNAs and their target genes. The sizes of the sets and their overlap are specified, and the diagram is labeled accordingly.Venn diagram showing the overlap between the target genes of hsa-mir-1-5p and hsa-mir-24-3p. <image_save>plot_CZaNNzdDr4.svg</image_save>",
              "media": "plot_CZaNNzdDr4.svg"
            }
            ```"""
    is_complete_content = '{"answer": "YES", "return": "The code generates a Venn diagram showing the overlap between two microRNAs and their target genes. The sizes of the sets and their overlap are specified, and the diagram is labeled accordingly.Venn diagram showing the overlap between the target genes of hsa-mir-1-5p and hsa-mir-24-3p. <image_save>plot_CZaNNzdDr4.svg</image_save>","media": "plot_CZaNNzdDr4.svg"}'
    chatbot_node = nodes.master_node
    r = chatbot_node.extract_json_from_markdown(content=content)
    print(r)
    assert r
    res = json.loads(r)
    assert res
    assert res['answer']=='YES'

def test_extract_json_from_json_tag(monkeypatch):
    """This test will evaluate for the old way of tag jsons"""
    content = """```json
{
 "caption": "Barplot of values a=1 and b=3",
 "code": "import matplotlib.pyplot as plt\n\nvalues = [1, 3]\nlabels = ['a', 'b']\n\nfig, ax = plt.subplots()\nax.bar(labels, values)\nax.set_xlabel('Variables')\nax.set_ylabel('Values')\nax.set_title('Barplot of a and b')\n\nfigure = fig",
 "notes": "The code generates a barplot with 'a' and 'b' on the x-axis and their corresponding values (1 and 3) on the y-axis."
}
```"""
    chatbot_node = nodes.master_node
    r = chatbot_node.extract_json_from_markdown(content=content)
    print(r)
    assert r
    res = json.loads(r)
    assert res
    assert res['caption']=="Barplot of values a=1 and b=3"

def test_is_complete():
    node = nodes.master_node
    response = node.is_complete(original_query="What is the seed of miR1?", history=["UGGA is the seed of hsa-miR-1"],
                     trys=2, answer_source='SQL_NODE')
    print(response)
    assert response['answer']=='YES'