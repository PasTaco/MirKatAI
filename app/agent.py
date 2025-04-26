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

# mypy: disable-error-code="union-attr"
# original libraries
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

# import libraries
import pandas as pd
import json
import os
import re
from pprint import pprint
from typing import Any, Dict, List, Literal, Optional, TypedDict # Grouped and sorted alphabetically
import mysql.connector
from mysql.connector import errorcode # Specific import from the same library
from pprint import pformat
import collections


# langchain and google ai specific libraries
from google.genai.types import GenerateContentResponse
from google.genai import types
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ( # Grouped message types
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage
    # ToolMessage is implicitly handled by LangGraph/ToolNode
)
from langchain_core.tools import tool # Decorator for creating LangChain tools
from langgraph.graph import END, StateGraph # Core graph builder and end state marker
from langgraph.prebuilt import ToolNode   

## load env variables and set up gemini API key:

from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Get the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

LOCATION = "europe-west1"
LLM = "gemini-2.0-flash"



# import tables with descriptonf of MiRkat DB
mirkat_columns_desctiption = pd.read_csv('tables/mirkat_tables_columns_descriptions.csv')
mirkat_tables_desctiption = pd.read_csv('tables/mirkat_tables_descriptions.csv')



# istantiate llms for nodes

llm_master = ChatGoogleGenerativeAI(model=LLM)



###### define instructions for nodes

### master node

ORIGINAL_MIRNA_SYSINT_CONTENT = """
You are MiRNA Researcher Assistant, basically another high level researcher. You can help with information regarding the microRNAs and their context. You have access to the miRKatDB with includes the infromation from miRBase, targetScan, mirnaTissueAtlas and other relevant microRNA databases. 
Aditionally, you can search the web to increase the context of the microRNAs, their functions, mechanisms of actions or any related to the biology. 
If the conversation is getting off topic, you must inform the user. If there is no more microRNA releted queries, finish the conversation.
"""

ROUTING_INSTRUCTIONS = """
## Routing Instructions:
Based on the user's latest message, analyze the request and decide the *next immediate step*. Respond ONLY with ONE of the following keywords,and the full instruction for the next model to take the task:

1.  `***ROUTE_TO_SQL***`: If the question *clearly* requires specific data retrieval from the miRKat database (e.g., list targets, find miRNA by seed, check expression levels, database schema questions).
3.  `***ANSWER_DIRECTLY***`: If you can answer the question directly based on the conversation history or general knowledge appropriate for this assistant, OR if you need to ask the user a clarifying question before proceeding.
5.  `***FINISH***`: If the user indicates they want to end the conversation (e.g., "thanks, that's all", "goodbye").

**Example Decisions:**
- User: "What are the validated targets of hsa-let-7a?" -> `***ROUTE_TO_SQL***`
- User: "Which database tables store tissue expression?" -> `***ROUTE_TO_SQL***`
- User: "Okay, thank you! Bye" -> `***FINISH***`
- User: "What's the weather?" -> `***ANSWER_DIRECTLY***` (Acknowledge off-topic, maybe offer to return to miRNAs)
"""

# Combine them (adjust formatting as needed for your LLM)
MIRNA_ASSISTANT_SYSINT_CONTENT = ORIGINAL_MIRNA_SYSINT_CONTENT + ROUTING_INSTRUCTIONS
ORIGINAL_MIRNA_SYSINT_CONTENT_MESSAGE = SystemMessage(ORIGINAL_MIRNA_SYSINT_CONTENT)
MIRNA_ASSISTANT_SYSTEM_MESSAGE = SystemMessage(content=MIRNA_ASSISTANT_SYSINT_CONTENT) # Create SystemMessage object

WELCOME_MSG = "Hello there. Please ask me your microRNA related questions. I have access to miRKat database and general web search."


### sql node

instruction = """You interact with an MySQL database
of microRNAs and its targets called mirkat. You will take the users questions and turn them into SQL
queries. Once you have the information you need, you will
return a Json object. 

If you need additional information use list_tables to see what tables are present, get_table_schema to understand the
schema, describe_tabes is you need to know what a table represents, describe_columns if you need to know biological 
meaning of the columns, and execute_query to issue an SQL SELECT query. If you don't find the table or the columns at the first
try use describe_columns again.

Avoid select all since the tables are huge. 

Examples:

human query: how many mirs are there?
sql query: SELECT count(*) FROM mirna

human query: Which is the most common seed?
sql query: SELECT seed, count(*) AS count FROM mirna_seeds GROUP BY seed ORDER BY count DESC LIMIT 1

human query: How many mirnas have seed GAGGUAG?
sql query: SELECT count(*) FROM mirna_seeds WHERE seed = 'GAGGUAG'

human query: How many human microRNAs have the seed GAGGUAG
sql query: SELECT COUNT(DISTINCT mm.mature_name) FROM mirna_seeds ms JOIN mirna_mature mm ON ms.auto_mature = mm.mature_name JOIN mirna_pre_mature mpm ON mm.auto_mature = mpm.auto_mature JOIN mirna m ON mpm.auto_mirna = m.auto_mirna JOIN mirna_species sp ON m.auto_species = sp.auto_id WHERE ms.seed = 'GAGGUAG' AND sp.name = 'Homo sapiens'

human query: What are the differences in targets of human mir 106a and mir-106b separed by source?
sql query:  SELECT gm.mrna, gm.mirna_mature, gm.source FROM gene_mirna gm WHERE gm.mirna_mature IN ('hsa-miR-106a-5p', 'hsa-miR-106b-5p') 

human query: What are the seeds of the microRNAs that expressed in muscle and how many of those mirnas have said seed?
sql_query: execute_query(SELECT ms.seed, COUNT(DISTINCT mt.mirna) FROM mirna_seeds ms JOIN mirna_mature mm ON ms.auto_mature = mm.mature_name JOIN mirna_pre_mature mpm ON mm.auto_mature = mpm.auto_mature JOIN mirna m ON mpm.auto_mirna = m.auto_mirna JOIN mirna_tissues mt ON m.mirna_ID = mt.mirna WHERE mt.organ = 'muscle' GROUP BY ms.seed



"""



#### SQL connection

config = {
    'user': os.getenv('MIRKAT_USER'),
    'password': os.getenv('MIRKAT_PASSWORD'),
    'host': os.getenv('MIRKAT_HOST'),
    'database': os.getenv('MIRKAT_DATABASE'),
    'raise_on_warnings': True
}

cnx = None


def connect_sql():
    try:
        cnx = mysql.connector.connect(**config)
        return cnx
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
        return None

    
db_conn = connect_sql()




#### functions for SQL nodes

def list_tables() -> list[str]:
    """Retrieve the names of all tables in the database."""
    # Include print logging statements so you can see when functions are being called.
    print(' - DB CALL: list_tables()')

    cursor = db_conn.cursor()

    # Fetch the table names.
    cursor.execute("SHOW TABLES;")

    tables = cursor.fetchall()
    return [t[0] for t in tables]


def get_table_schema(table_name: str) -> list[tuple[str, str]]:
    """Look up the table schema.

    Returns:
      List of columns, where each entry is a tuple of (column, type).
    """
    print(f' - DB CALL: describe_table({table_name})')

    cursor = db_conn.cursor()

    cursor.execute(f"DESCRIBE `{table_name}`;")
    
    schema = cursor.fetchall()
    # MySQL returns (Field, Type, Null, Key, Default, Extra), so we extract the first two columns.
    return [(col[0], col[1]) for col in schema]

def describe_columns(table_name:str) -> list[tuple[str,str]]:
    """ Looks for the columns in the table table_name and gets the 
        biological description of the table
        Args:
            table_name (str): Name of the table to describe
        Returns:
            list[tuple[str,str]]: List of tuples containing column names and their descriptions
    """
    # Check if the table name exists in the DataFrame
    if table_name not in mirkat_columns_desctiption['Table'].values:
        print(f"Error: Table '{table_name}' not found.")
        return []

    # Filter the DataFrame for the specified table name
    filtered_df = mirkat_columns_desctiption[mirkat_columns_desctiption['Table'] == table_name]

    # Extract column names and descriptions
    columns = list(zip(filtered_df['Column Name'], filtered_df['Description']))
    
    return columns
    
def describe_tabes() -> list[tuple[str,str]]:
    """ Looks for the biological description 
    and returns the description of all the tables
    """
    # Extract table names and descriptions
    tables = list(zip(mirkat_tables_desctiption['Table'], mirkat_tables_desctiption['Description']))
    
    return tables

def execute_query(sql: str, query_name:str) -> list[list[str]]:
    """Execute an SQL statement, returning the results.
        params sql: is the formated mySQL query
        params query_name: name of the query only alfanumeric characters.
        """
    print(f' - DB CALL: execute_query({sql})')

    cursor = db_conn.cursor()

    cursor.execute(sql)
    results = cursor.fetchall()
    with open(f"{query_name}.tsv", "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(results) 
    
    return results


db_tools = [list_tables, get_table_schema, describe_columns, describe_tabes, execute_query]



#### functions for literature search

# literature search node
import google.ai.generativelanguage as genai_types
from google.genai import types

def process_chunk(chunk,n):
    title=chunk.web.title
    uri=chunk.web.uri
    reference=f"""
{n}. [{title}]({uri})"""
    return reference

def create_bibliography(chunks):
    bibliography=""
    if chunks is not None:
        for i,chunk in enumerate(chunks):
            bibliography = bibliography + process_chunk(chunk,i+1)
    return bibliography


def process_indices(indices,chunks):
    indices_text=""
    for i,indice in enumerate(indices):
        uri=chunks[indice].web.uri
        indices_text = indices_text + f"[{indice+1}]({uri})"
        if i < len(indices)-1:
            indices_text = indices_text +","
    indices_text= f"[{indices_text}]"
    return indices_text


def process_text_ref(support, chunks):
    indices = support.grounding_chunk_indices
    text = support.segment.text
    return text,text + process_indices(indices, chunks)


def process_references(answer, supports,chunks):
    if supports is not None:
        for support in supports:
            text,ref=process_text_ref(support,chunks)
            answer=answer.replace(text,ref)
    return answer

def process_paragraph(answer, supports, chunks):
    answer=process_references(answer,supports,chunks)
    bibliography=create_bibliography(chunks)
    paragraph = f"**Answer**\n{answer}\n\n**Bibliography**\n{bibliography}"
    return paragraph



#### Functions plotting node

# def handle_response(msg, tool_impl=None):
#     """Stream output and handle any tool calls during the session."""
#     msg = msg.candidates[0].content
#     for part in msg.parts:
#           if result := part.code_execution_result:
#             display(Markdown(f'### Result: {result.outcome}\n'
#                              f'```\n{pformat(result.output)}\n```'))

#           elif code := part.executable_code:
#             #display(Markdown(
#             #    f'### Code\n```\n{code.code}\n```'))
#             exec(code.code) 
          
#           elif img := part.inline_data:
#             display(Image(img.data))




def get_queries(callings):
    """ From the response.automatic_function_calling_history, 
    get the queries that were exceuted with good results and saving it on a dictionary
    query:result.
    """
    queries = {}
    query_queue = collections.deque()
    for call in callings:
        for part in call.parts:#.content.parts:
        #call = [0]
            #print (part)
            if hasattr(part, 'function_call') and part.function_call:
                #print(part.function_call.name)
                if part.function_call.name == 'execute_query':
                    query = part.function_call.args['sql']
                    query_queue.append(query)
            if hasattr(part, 'function_response') and part.function_response:
                if part.function_response.name =='execute_query':
                    #print(part.function_response.response)
                    popped_query = query_queue.popleft()
                    if 'result' in part.function_response.response:
                        queries[popped_query] = part.function_response.response
                    query= ''
    return queries




##### Define SQL agent

client = genai.Client(api_key=GOOGLE_API_KEY)

config_tools = types.GenerateContentConfig(
    system_instruction=instruction,
    tools=db_tools,
    temperature=0.0,
    )

# Start a chat with automatic function calling enabled.
chat = client.chats.create(
    model=LLM,
    config=config_tools,
)


##### Configure literature research agent

config_with_search = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
)


#### configure plotting node

config_with_code = types.GenerateContentConfig(
    tools=[types.Tool(code_execution=types.ToolCodeExecution())],
    temperature=0.0,
)

plotter_model = client.chats.create(model=LLM, config=config_with_code)



# start defininf graph


class GraphState(TypedDict):
    messages: List[BaseMessage]
    table: Optional[Dict[str, Any]]
    answer: str
    bibliography: list
    research_queries: list
    finished: bool
    



# define nodes

def human_node(state: GraphState) -> GraphState:
    """Display the last model message to the user, and rec
    eive the user's input."""
    #print("\n--- ENTERING: human_node ---")
    last_msg = state["messages"][-1]
    answer = state["answer"]
    support = None
    chunks = None
    #print(F"----- ANSWER: {answer} -------")
    if isinstance(last_msg, AIMessage) or isinstance(last_msg, GenerateContentResponse):
        if answer:
            #print("Assistant:", answer)
            display(Markdown(answer))
            state["answer"] = None
            #print()
        else:
            #print("Assistant:", last_msg.content)
            display(Markdown(last_msg.content))
    print("="*30)

    user_input = input("User: ")

    # If it looks like the user is trying to quit, flag the conversation
    # as over.
    if user_input.strip().lower() in {"q", "quit", "exit", "goodbye", "bye", "thanks that's all"}:
        state["finished"] = True

    #return state | {"messages": [("user", user_input)]}
    return {
        "messages": state["messages"] +[HumanMessage(content=user_input)],
        "table": state["table"],
        "answer": state["answer"],
        "finished": state["finished"]}

def chatbot_with_tools(state: GraphState) -> GraphState:
    """The chatbot with tools. A simple wrapper around the model's own chat interface."""
    print("\n--- ENTERING: master_node ---")

    
    messages = state['messages']

    # Check if this is the very first turn (no messages yet)
    if not messages:
        # Generate the welcome message directly
        print("--- Generating Welcome Message ---")
        response = AIMessage(content=WELCOME_MSG)
    else:
        # Normal operation: Invoke the master LLM for routing/response
        print("--- Calling Master Router LLM ---")
        # Always invoke with the system message + current history
        print(f"--- Message going to the llm_master: {messages}---")
        response = llm_master.invoke([MIRNA_ASSISTANT_SYSTEM_MESSAGE] + messages)
        if response.content.strip() == "***ANSWER_DIRECTLY***":
            response = llm_master.invoke([ORIGINAL_MIRNA_SYSINT_CONTENT_MESSAGE] + messages)
            answer = response.content
        print(f"--- Master Router Raw Response: {response.content} ---")

    # Update state
    return {
        **state, # Preserve other state fields
        "messages": state["messages"] + [response] # Add the router's decision/response
    }

def sql_processor_node(state: GraphState) -> GraphState:
    """The sql llm that will check for the sql questions and get a json file in response."""
    print("--- Calling SQL Processor Node ---")
    messages = [state['messages'][-2].content]
    if not messages:
        # Should ideally not happen if routing is correct
        #print("Warning: SQL processor called with no messages.")
        # Return unchanged state or add an error message? For now, return unchanged.
        return state
    #print("The message sent to the SQL node is: ", messages)
    response = chat.send_message(messages)

    queries = get_queries(response.automatic_function_calling_history)
    #handle_response(response)
    #response = sql_llm_with_db_tools.invoke([SQL_SYSTEM_INSTRUCTION] + messages)
    #print(f"--- SQL Processor LLM Response: {response} ---")
    
    
    new_answer = state.get("answer", "")
    
    if isinstance(response, AIMessage) and response.content and not response.tool_calls:
         new_answer = response.content # Update answer if it's a direct text response
    elif isinstance(response, GenerateContentResponse):
        new_answer = response.text
    elif isinstance(response, str):
        new_answer = response
    new_messages = messages + [AIMessage(content=new_answer)]
    #print(f"--- Answer from SQL Processor LLM Response: {new_answer} ---")
    return {
        "messages": new_messages,
        "table": queries, # Use .get for safety
        "answer": new_answer, # Return the potentially updated answer
        "finished": state.get("finished", False), # Use .get for safety
    }
    


def literature_search_node(state: GraphState) -> GraphState:
    """Perform GroundSearch with UserQuery
    Returns:
    - Answer: Markdown formatted text answer from Ground Search iwth clickable references
    - Bibliography: References, link and website use to obtain the answer
    - ResearcQueries: Quesries used to perform GroundSearch"""
    #print("\n--- ENTERING: Literature node ---")
    user_query = state["messages"][-1].content
    #print(f"\n--- SEARCHING {user_query} with GroundSearch model ---")

    # --- Grounding Setup ---
    # Use the native Google Search tool for grounding
  

    # --- Model Selection ---
    ## gemini-2.0-flash is faster and return less issues compared to gemini-1.5-flash
    model_name = LLM 
    print("\n--- Performing: GroundSearch ---")

    response = client.models.generate_content(
                model=model_name,
                contents=user_query,               # Pass the user's query here
                config=config_with_search, # Apply the grounding config
                # system_instruction=LITERATURE_SYSTEM_INSTRUCTION_CONTENT, # Apply system instruction
            )

    answer = response.text
    chunks = response.candidates[0].grounding_metadata.grounding_chunks
    supports = response.candidates[0].grounding_metadata.grounding_supports
    research_queries=response.candidates[0].grounding_metadata.web_search_queries

    answer=process_references(answer,supports,chunks)
    bibliography=create_bibliography(chunks)


    
    #print(F"----- ANSWER: {answer} -------")


    return {
        "messages": state["messages"] +[AIMessage(content=answer)],
        "table" : state["table"],
        "answer": answer,
        "bibliography": bibliography,
        "research_queries": research_queries,
        "finished": state["finished"]}


def plot_node(state:GraphState) -> GraphState:
    messages = state['messages'][-1].content
    queries = state['table']

    response_plot = plotter_model.send_message(str(queries) + messages)
    handle_response(response_plot)
    answer = ''
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            answer = answer + f"{part.text}\n"
    
    return {**state,
           "messages":state["messages"] + [AIMessage(content=answer)]
           }
    
    

all_tools = db_tools # Add literature search tools here if they were LangChain tools
tool_node = ToolNode(all_tools)





# Define node names for clarity
HUMAN_NODE = "human_node"
CHATBOT_NODE = "chatbot_router"
SQL_NODE = "sql_processor_node"
LITERATURE_NODE = "literature_search_node"
TOOL_NODE = "execute_tools" # Name for the ToolNode instance
PLOT_NODE = "plot_node"


### Routing criteria

# edges
def maybe_exit_human_node(state: GraphState) -> Literal["chatbot", "__end__"]:
    """Route to the chatbot, unless it looks like the user is exiting."""
    if state.get("finished", False):
        return END
    else:
        return "chatbot"

def route_after_human(state: GraphState) -> Literal["chatbot_router", "__end__"]:
    """
    Determines the next step after the human node.
    If the 'finished' flag is set, ends the graph.
    Otherwise, directs the conversation to the main chatbot.
    """
    #print("\n--- ROUTING: route_after_human ---")
    if state.get("finished", False):
        #print("--- Routing: Human to END ---")
        return END
    else:
        #print("--- Routing: Human to Chatbot ---")
        return CHATBOT_NODE

# Router 2: After the Main Chatbot/Router (`chatbot_with_tools`)
def route_chatbot_decision(state: GraphState) -> Literal["sql_processor_node", "literature_search_node","human_node", "__end__"]:
    """
    Inspects the last message from the main chatbot (`chatbot_with_tools`)
    and decides where to route the conversation next.
    """
    #print("\n--- ROUTING: route_chatbot_decision ---")
    messages: List[BaseMessage] = state['messages']
    if not messages:
        #print("--- Routing Error: No messages found in route_chatbot_decision ---")
        return END # Or raise error

    last_message = messages[-1]
    
    if not isinstance(last_message, AIMessage) :
        #print(f"--- Routing Warning: Expected AIMessage, got {type(last_message)}. Routing to Human. ---")
        return HUMAN_NODE
        
    content = last_message.content.strip()
    
    # Check for routing keywords first
    if "***ROUTE_TO_SQL***" in content:
        #print("--- Routing: Master Router to SQL Processor ---")
        return SQL_NODE
    elif "***ROUTE_TO_LITERATURE***" in content:
        #print("--- Routing: Master Router to Literature Searcher ---")
        # state['messages'][-1].content = "Okay, I need to search the literature for that."
        return LITERATURE_NODE
    elif "***FINISH***" in content or state.get("finished"): # Check flag too
        #print("--- Routing: Master Router to END ---")
        return END
    elif "***ANSWER_DIRECTLY***" in content:
         #print("--- Routing: Master Router to Human (Direct Answer) ---")
         # Remove the keyword itself before showing to human
         state['messages'][-1].content = content.replace("***ANSWER_DIRECTLY***", "").strip()
         # If the content is *only* the keyword, maybe add a placeholder?
         #if not state['messages'][-1].content:
         x = state['messages'][-2].content#.candidates[0].content.parts[0].text
         #print (f"--- The messages directly was: {x}")
         answer = llm_master.invoke(x) #"Okay, let me answer that." # Or similar
         state['messages'][-1].content = re.sub(r'[*_`~#\[\]()]', '', answer.content)
         #print (f"--- The answer directly was: {answer}")
         state['answer'] = answer#.response.candidates[0].content.parts[0].text
         return HUMAN_NODE
    elif "***PLOT***" in content:
        #print("---- Routing to plot node ----")
        return PLOT_NODE
    else:
         # Assume it's a direct answer or clarification question if no keyword is found
         #print("--- Routing: Master Router to Human (Direct Answer) ---")
         # Remove potential keywords just in case they were missed but shouldn't be shown
         state['messages'][-1].content = content.replace("***ROUTE_TO_SQL***", "").replace("***ROUTE_TO_LITERATURE***", "").replace("***FINISH***", "").replace("***ANSWER_DIRECTLY***", "").strip()
         return HUMAN_NODE

# Router 3: After a Specialist Processor Node (`sql_processor_node`, `literature_search_node`)
def route_processor_output(state: GraphState) -> Literal["human_node", "__end__"]:
    """
    Inspects the last message from a specialist processor node.
    Routes to 'tools' if a tool call was made (e.g., query_database, ground_search).
    Routes to 'human_node' if a final synthesized answer was provided.
    """
    #print("\n--- ROUTING: route_processor_output ---")
    messages: List[BaseMessage] = state['messages']
    if not messages:
        #print("--- Routing Error: No messages found in route_processor_output ---")
        return END

    last_message = messages[-1]

    #if not isinstance(last_message, AIMessage):
    #    print(f"--- Routing Warning: Expected AIMessage from processor, got {type(last_message)}. Routing to Human. ---")
    #    return HUMAN_NODE
    # Otherwise, the processor provided its final synthesized answer
    #else:
    #    print("--- Routing: Processor to Human ---")
    return HUMAN_NODE
        
def route_after_tools(state: GraphState) -> Literal["sql_processor_node", "literature_search_node","human_node"]:
    """ Routes back to the specialist node that originally called the tool OR to human if unclear."""
    #print("\n--- ROUTING: route_after_tools ---")
    messages = state['messages']
    # The last message is the ToolMessage with results
    # The second to last message *should* be the AIMessage that made the tool call
    if len(messages) < 2:
         #print("--- Routing Warning: Tool execution happened without prior AI message? Routing to Human. ---")
         return HUMAN_NODE # Should not happen

    ai_message_that_called_tool = messages[-2]

    # This is heuristic: Check which LLM likely generated the tool call
    # A more robust way might be to add metadata to the state indicating the caller.
    # Simple approach: Check the tools called. If DB tools, assume SQL node. If search, assume Literature.

    # Check if the tool call originated from SQL LLM (by checking tool names)
    db_tool_names = {t.name for t in db_tools}
    called_tool_names = {call['name'] for call in ai_message_that_called_tool.tool_calls}

    if any(name in db_tool_names for name in called_tool_names):
         #print("--- Routing: Tools back to SQL Processor ---")
         return SQL_NODE
    # Check if the tool call was Google Search (handled implicitly by LangChain for native tools)
    elif any(call['name'].lower() == 'googlesearchretrieval' for call in ai_message_that_called_tool.tool_calls):
         #print("--- Routing: Tools back to Literature Searcher ---")
         return LITERATURE_NODE
    else:
         # Fallback if the origin is unclear
         #print(f"--- Routing Warning: Tool caller unclear ({called_tool_names}). Routing to Human. ---")
         # Add a message indicating confusion?
         state['messages'].append(SystemMessage(content="(System: Unclear which process should handle the tool results. Displaying results directly.)"))
         return HUMAN_NODE
    



# --- Build the Graph ---
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node(HUMAN_NODE, human_node)
workflow.add_node(CHATBOT_NODE, chatbot_with_tools)
workflow.add_node(SQL_NODE, sql_processor_node)
# workflow.add_node(LITERATURE_NODE, literature_search_node)
workflow.add_node(TOOL_NODE, tool_node)
# workflow.add_node(PLOT_NODE, plot_node)


# --- Define Edges ---

# 1. Entry Point (Where the graph starts)
workflow.set_entry_point(CHATBOT_NODE) # Start with a hello input

# 2. From Human Node
workflow.add_conditional_edges(
    HUMAN_NODE,
    route_after_human, # Function to decide next step
    {
        CHATBOT_NODE: CHATBOT_NODE, # If function returns "chatbot_with_tools", go there
        END: END                   # If function returns "__end__", finish
    }
)

# 3. From Main Chatbot Node
workflow.add_conditional_edges(
    CHATBOT_NODE,
    route_chatbot_decision, # Function to decide based on chatbot output
    {
        SQL_NODE: SQL_NODE,                 # Route to SQL processor
        # LITERATURE_NODE: LITERATURE_NODE,   # Route to Literature searcher
        TOOL_NODE: TOOL_NODE,               # Route to execute chatbot's tools (get_menu)
        HUMAN_NODE: HUMAN_NODE,             # Route to show chatbot's direct answer
        # PLOT_NODE: PLOT_NODE,              # Route to plot node
        END: END                           # Route to end (though usually handled via human)
    }
)

# 4. From SQL Processor Node
workflow.add_conditional_edges(
    SQL_NODE,
    route_processor_output, # Function to decide based on SQL processor output
    {
        TOOL_NODE: TOOL_NODE,   # Route to execute SQL tools (query_database)
        HUMAN_NODE: HUMAN_NODE  # Route to show final SQL answer
    }
)

# 5. From Literature Search Node
# workflow.add_conditional_edges(
#    LITERATURE_NODE,
#    route_processor_output, # Function to decide based on Literature processor output
#    {
#        TOOL_NODE: TOOL_NODE,   # Route to execute literature tools (ground_search)
#        HUMAN_NODE: HUMAN_NODE  # Route to show final literature answer
#    }
# )

# 6. From Tool Node - Route back to the appropriate processor
# workflow.add_conditional_edges(
#     TOOL_NODE,
#     route_after_tools,
#     {
#         SQL_NODE: SQL_NODE,
#         LITERATURE_NODE: LITERATURE_NODE,
#         HUMAN_NODE: HUMAN_NODE # Fallback route
#     }
# )


# workflow.add_edge(PLOT_NODE, HUMAN_NODE)



# define state

# Initial state with a welcome message
initial_state = {
    "messages": [], # <-- Empty list
    "table": None,
    "answer": "",
    "finished": False
}
current_state = initial_state
config = {"recursion_limit": 100}


agent = workflow.compile()



# # 1. Define tools
# @tool
# def search(query: str) -> str:
#     """Simulates a web search. Use it get information on weather"""
#     if "sf" in query.lower() or "san francisco" in query.lower():
#         return "It's 60 degrees and foggy."
#     return "## Weather is:\nIt's 90 degrees and sunny."


# tools = [search]

# # 2. Set up the language model
# llm = ChatVertexAI(
#     model=LLM, location=LOCATION, temperature=0, max_tokens=1024, streaming=True
# ).bind_tools(tools)


# # 3. Define workflow components
# def should_continue(state: MessagesState) -> str:
#     """Determines whether to use tools or end the conversation."""
#     last_message = state["messages"][-1]
#     return "tools" if last_message.tool_calls else END


# def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
#     """Calls the language model and returns the response."""
#     system_message = "You are a helpful AI assistant."
#     messages_with_system = [{"type": "system", "content": system_message}] + state[
#         "messages"
#     ]
#     # Forward the RunnableConfig object to ensure the agent is capable of streaming the response.
#     response = llm.invoke(messages_with_system, config)
#     return {"messages": response}


# # 4. Create the workflow graph
# workflow = StateGraph(MessagesState)
# workflow.add_node("agent", call_model)
# workflow.add_node("tools", ToolNode(tools))
# workflow.set_entry_point("agent")

# # 5. Define graph edges
# workflow.add_conditional_edges("agent", should_continue)
# workflow.add_edge("tools", "agent")

# # 6. Compile the workflow
# agent = workflow.compile()
