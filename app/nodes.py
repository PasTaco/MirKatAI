
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


current_path = os.path.dirname(os.path.abspath(__file__))
mirkat_path = current_path.split('MirKatAI')[0]
mirkat_path = mirkat_path + "MirKatAI/"


# Load .env file
load_dotenv()
# Get the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# set the models
LOCATION = "europe-west1"
LLM_ROUTE = "gemini-2.-flash-lite"
LLM = "gemini-2.5-flash-lite"
LLM_SQL = "gemini-2.0-flash-lite"
LLM_PLOT = "gemini-2.5-flash-lite"
###### define instructions for nodes

GENERAL_INSTRUCTIONS = Instructions.general.get_instruction()

MIRNA_ASSISTANT_SYSTEM_MESSAGE, WELCOME_MSG = Instructions.router.get_instruction()
MIRNA_COMPLETE_ANSWER = GENERAL_INSTRUCTIONS + Instructions.format_answer.get_instruction()
SQL_INSTRUCTIONS = GENERAL_INSTRUCTIONS + Instructions.sql.get_instruction()
PLOT_INSTRUCTIONS = GENERAL_INSTRUCTIONS + Instructions.plot.get_instruction()
LITERATURE_INSTRUCTIONS = GENERAL_INSTRUCTIONS + Instructions.literature.get_instruction()
##### Specific for SQL NODE

### SQL tables descriptions
pwd = os.getcwd()
# import tables with description of MiRKat DB
# the if else is so it can work on the pyCharm tests

mirkat_columns_desctiption = pd.read_csv(mirkat_path+'tables/mirkat_tables_columns_descriptions.csv')
mirkat_tables_desctiption = pd.read_csv(mirkat_path+'tables/mirkat_tables_descriptions.csv')


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
    db_tools_instance.describe_tables,
    db_tools_instance.execute_query
]


##### Creating the nodes

master_node = ChatbotNode(llm=LLM, instructions=MIRNA_ASSISTANT_SYSTEM_MESSAGE)
literature_search_node = LiteratureNode(llm=LLM, functions=LiteratureTools, instructions=LITERATURE_INSTRUCTIONS)
plot_node = PlotNode(llm=LLM_PLOT, instructions=PLOT_INSTRUCTIONS)
sql_node = SQLNode(llm=LLM_SQL, instructions=SQL_INSTRUCTIONS, functions=db_tools)
all_tools = db_tools # Add literature search tools here if they were LangChain tools
tool_node = ToolNode(all_tools)

def reset_sql_node():
    global sql_node
    sql_node = SQLNode(llm=LLM_SQL, instructions=SQL_INSTRUCTIONS, functions=db_tools)


def create_nodes():
    master_node = ChatbotNode(llm=LLM, instructions=MIRNA_ASSISTANT_SYSTEM_MESSAGE)
    literature_search_node = LiteratureNode(llm=LLM, functions=LiteratureTools, instructions=LITERATURE_INSTRUCTIONS)
    plot_node = PlotNode(llm=LLM_PLOT, instructions=PLOT_INSTRUCTIONS)
    sql_node = SQLNode(llm=LLM_SQL, instructions=SQL_INSTRUCTIONS, functions=db_tools)
    all_tools = db_tools # Add literature search tools here if they were LangChain tools
    tool_node = ToolNode(all_tools)
    return master_node, literature_search_node, plot_node, sql_node, tool_node
