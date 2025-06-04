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
import pandas as pd
from app.mirkat.node_chatbot import ChatbotNode
from app.mirkat.node_plot import PlotNode
from app.mirkat.node_sql import SQLNode
from app.mirkat.node_literature import LiteratureNode
from dotenv import load_dotenv
from app.mirkat.instructions import Instructions

from app.mirkat.sql_functions import (
    MySqlConnection,
    DBTools
)
from app.mirkat.literature_functions import LiteratureTools
from langgraph.prebuilt import ToolNode   



# Load .env file
load_dotenv()
# Get the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# set the models
LOCATION = "europe-west1"
LLM_ROUTE = "gemini-1.5-flash"
LLM = "gemini-2.0-flash"
LLM_SQL = "gemini-2.5-flash-preview-04-17"
LLM_PLOT = "gemini-2.0-flash"
###### define instructions for nodes


MIRNA_ASSISTANT_SYSTEM_MESSAGE, WELCOME_MSG = Instructions.router.get_instruction()
MIRNA_COMPLETE_ANSWER = Instructions.format_answer.get_instruction()
SQL_INSTRUCTIONS = Instructions.sql.get_instruction()
PLOT_INSTRUCTIONS = Instructions.plot.get_instruction()


##### Specific for SQL NODE

### SQL tables descriptions

# import tables with descriptonf of MiRkat DB
mirkat_columns_desctiption = pd.read_csv('tables/mirkat_tables_columns_descriptions.csv')
mirkat_tables_desctiption = pd.read_csv('tables/mirkat_tables_descriptions.csv')

#### SQL connection

config = {
    'user': os.getenv('MIRKAT_USER'),
    'password': os.getenv('MIRKAT_PASSWORD'),
    'host': os.getenv('MIRKAT_HOST'),
    'database': os.getenv('MIRKAT_DATABASE'),
    'raise_on_warnings': True
}

mysql_connection = MySqlConnection(config)
db_conn = mysql_connection.connect_sql()
db_tools_instance = DBTools(db_conn, mirkat_tables_desctiption, mirkat_columns_desctiption)
db_tools = [
    db_tools_instance.list_tables,
    db_tools_instance.get_table_schema,
    db_tools_instance.describe_columns,
    db_tools_instance.describe_tabes,
    db_tools_instance.execute_query
]


##### Creating the nodes

master_node = ChatbotNode(llm=LLM, instructions=MIRNA_ASSISTANT_SYSTEM_MESSAGE)
literature_search_node = LiteratureNode(llm=LLM, functions=LiteratureTools)
plot_node = PlotNode(llm=LLM_PLOT, instructions=PLOT_INSTRUCTIONS)
sql_node = SQLNode(llm=LLM_SQL, instructions=SQL_INSTRUCTIONS, functions=db_tools)
all_tools = db_tools # Add literature search tools here if they were LangChain tools
tool_node = ToolNode(all_tools)