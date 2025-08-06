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
import pickle
import json
import re
import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage
)
import app.nodes as nodes

master_node = nodes.master_node
literature_search_node = nodes.literature_search_node
plot_node = nodes.plot_node
sql_node = nodes.sql_node

current_path = os.path.dirname(os.path.abspath(__file__))
dummy_path = current_path.split('MirKatAI')[0]
dummy_path = dummy_path + "MirKatAI/tests/dummy_files/"


def test_plot_run_model_json():
    """
    This test will make sure that the output from the gemini model is in json fromat
    """
    messages = "***PLOT*** Plot a barplot of values a=1 and b=3."
    result = plot_node.run_model(messages)
    print(result)
    assert result
    result_content = result.text
    # convert rest of the content to json
    result_json = json.loads(result_content)
    # check that result_json has the keys caption, code and notes
    assert "caption" in result_json
    assert "code" in result_json
    assert "notes" in result_json
    assert "figure" in result_json['code'], "The final figure must be saved on variable figure"
    # Save the result to a pickle file for later use
    with open(dummy_path+"plot_result.pkl", "wb") as f:
        pickle.dump(result, f)
    
def test_sql_run_model():
    """
    This test will make sure that the output from the gemini model is in json fromat
    """
    messages = "***SQL*** Check how many targets hsa-mir1-5p has in the database. Provide only the number"
    ai_message = AIMessage(content=messages)
    result = sql_node.run_model(ai_message)
    print(result)
    assert result
    result_content = result.text
    if "error" in result_content and "database" in result_content:
        raise "Connection to the database error."
    # convert rest of the content to json
    # check that result_json has the keys caption, code and notes
    assert "1021" in result_content
    
    # Save the result to a pickle file for later use
    #with open(dummy_path + "sql_result.pkl", "wb") as f:
    #    pickle.dump(result, f)

def test_check_compleatness_model_from_plot_json():
    """
    This test will make sure that the output from the gemini model after getting and
    answer is correctly formated.
    """

    def extract_json_from_markdown(content: str) -> dict:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON block found in markdown.")
        return json.loads(match.group(1))



    status = {'answer': AIMessage(content='Barplot of values a=1 and b=3 <image_save>file</image_save>',
                                  additional_kwargs={},
                                  response_metadata={},
                                  id='d16f8ff3-8d61-4645-b34a-72154900b8af'),
              'answer_source': 'PlotNode',
              'finished': False,
              'history':
                  [HumanMessage(content='Barplot of values a=1 and b=3',
                                additional_kwargs={},
                                response_metadata={},
                                id='344c2e75-a150-4a19-9a11-bfd2f8863691'),
                   AIMessage(content='***PLOT*** Plot A=1, B=3',
                             additional_kwargs={},
                             response_metadata={},
                             id='84e6698f-9e1b-43cf-a323-c70a066420d6'),
                   AIMessage(content='Barplot of values a=1 and b=3 using code: Placeholder ',
                             additional_kwargs={},
                             response_metadata={},
                             id='cad2e8d8-c9b4-45c9-9e1f-e194a2ffe791')],
              'messages': AIMessage(content='',
                                    additional_kwargs={},
                                    response_metadata={},
                                    id='82ecf564-698d-46ad-a1c5-ee4547e7e880'),
              'original_query': HumanMessage(content='Barplot of values a=1 and b=3',
                                             additional_kwargs={},
                                             response_metadata={},
                                             id='344c2e75-a150-4a19-9a11-bfd2f8863691'),
              'request': AIMessage(content="The code generates a barplot with 'a' and 'b' on the x-axis and their corresponding values (1 and 3) on the y-axis.",
                                   additional_kwargs={},
                                   response_metadata={},
                                   id='19d5f82b-80cc-40f7-a736-31085cd7e437'),
              'trys': 1}

    message = {'answer': "The code generates a barplot with 'a' and 'b' on the x-axis and their corresponding values (1 and 3) on the y-axis.",
               'original_query': 'Barplot of values a=1 and b=3',
               'message': 'Barplot of values a=1 and b=3 <image_save>file</image_save>',
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
    message_str = str(message)
    result = master_node.run_model_for_compleatness(message_str=message_str)

    print(result)
    assert hasattr(result, 'content'), "Result should have a 'content' attribute."
    response_data = result.content
    # Try to parse the content as JSON
    assert "```json" in response_data
    response_data = extract_json_from_markdown(response_data)
    # Check for required keys
    assert "answer" in response_data, "'answer' key is missing in the response JSON."
    assert "return" in response_data, "'return' key is missing in the response JSON."

    # Check that answer is a string and has an expected value
    assert isinstance(response_data["answer"], str), "'answer' must be a string."
    assert response_data["answer"] in {"YES", "NO"}, "'answer' must be either 'YES' or 'NO'."

    # Check that return is a string
    assert isinstance(response_data["return"], str), "'return' must be a string."

    # Optional: check expected substrings in return
    assert "image" in response_data["return"], "'return' explanation must mention 'binary_image'."

    # Save the result to a pickle file for later use
    with open(dummy_path + "/completes_plot_result.pkl", "wb") as f:
        pickle.dump(result, f)


@pytest.mark.skip("This test don't do anything.")
def test_temporal():
    result = pickle.load(open(dummy_path + "/plot_result.pkl", "rb"))
    result = pickle.load(open(dummy_path + "/completes_plot_result.pkl", "rb"))
    result_text = result.text
    print(result)


