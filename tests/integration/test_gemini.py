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

import pytest

import app.nodes as nodes

master_node = nodes.master_node
literature_search_node = nodes.literature_search_node
plot_node = nodes.plot_node




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



#@pytest.mark.skip("This test don't do anything.")
def test_temporal():
    result = pickle.load(open("../dummy_files/plot_result.pkl", "rb"))
    result_text = result.text
    print(result)


