# Standard libraries
import os
import logging
from typing import Any, Dict, List, Literal, Optional, TypedDict
import sys


# Make sure stdout/stderr use UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mirkat.log'),
        logging.StreamHandler()
    ],
    encoding='utf-8'  # Ensure log messages are UTF-8 encoded
)

# Environment variables
from dotenv import load_dotenv

# LangChain and Google AI specific libraries
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage
)
from google.genai.types import GenerateContentResponse
from langchain_core.tools import tool

# LangGraph specific libraries
from langgraph.graph import END, StateGraph

# Application-specific imports
import app.nodes as nodes
from  app.mirkat.graph_state import GraphState

# Current path
current_path = os.path.dirname(os.path.abspath(__file__))
print(f"Current path: {current_path}")
# Load .env file
load_dotenv()



# Nodes

master_node = nodes.master_node
literature_search_node = nodes.literature_search_node
plot_node = nodes.plot_node
sql_node = nodes.sql_node
tool_node = nodes.tool_node


# start defininf graph


# define nodes

def human_node(state: GraphState) -> GraphState:
    """Display the last model message to the user, and rec
    eive the user's input."""
    print("\n--- ENTERING: human_node ---")
    last_msg = state["messages"]
    answer = state["answer"]
    support = None
    chunks = None
    if isinstance(last_msg, AIMessage) or isinstance(last_msg, GenerateContentResponse):
        if answer:
            state["answer"] = None

    print("="*30)
    return state
   

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
    return END


# Router 2: After the Main Chatbot/Router (`chatbot_with_tools`)
def route_chatbot_decision(state: GraphState) -> Literal["sql_processor_node", "literature_search_node","human_node", "chatbot_router",  "__end__"]:
    """
    Inspects the last message from the main chatbot (`chatbot_with_tools`)
    and decides where to route the conversation next.
    """
    logging.info("Routing decision based on the last message from the chatbot.")
    logging.info(f"--- Current state: {state} ---")
    messages: List[BaseMessage] = state['request']
    logging.info(f"--- Messages in state: {messages} ---")
    if not messages:
        #print("--- Routing Error: No messages found in route_chatbot_decision ---")
        return END # Or raise error
    if isinstance(messages, list):
        last_message = messages[-1]
    else:
        last_message = messages
    if not isinstance(last_message, AIMessage) :
        #return HUMAN_NODE
        content = last_message
    else:
        content = last_message.content.strip()
    
    # Check for routing keywords first
    
    if "***ROUTE_TO_SQL***" in content:
        print("--- Routing: Master Router to SQL Processor ---")
        return SQL_NODE
    elif "***ROUTE_TO_LITERATURE***" in content:
        #print("--- Routing: Master Router to Literature Searcher ---")
        # state['messages'][-1].content = "Okay, I need to search the literature for that."
        return LITERATURE_NODE
    elif "***PLOT***" in content:
        #print("---- Routing to plot node ----")
        return PLOT_NODE
    
    elif "***ANSWER_DIRECTLY***" in content:
        content = content.replace("***ANSWER_DIRECTLY***", "")
        state['messages'].content = str(content)
        #  #print (f"--- The answer directly was: {answer}")
        state['answer'] = str(content)#.response.candidates[0].content.parts[0].text
        print(f"\n\n\n After Answer directly before returning to finihs \n\n\n\n")
        return CHATBOT_NODE
    elif "***FINISH***" in content or state.get("finished"): # Check flag too
        # remove the finish keyword
        content = content.replace("***FINISH***", "")
        
        
        #print("--- Routing: Master Router to END ---")
        return END
    else:
        print(f"\n\n\n BEFORE ENDING \n\n\n\n")
        return CHATBOT_NODE

# Router 3: After a Specialist Processor Node (`sql_processor_node`, `literature_search_node`)
def route_processor_output(state: GraphState) -> Literal["chatbot_router","human_node", "__end__"]:
    """
    Inspects the last message from a specialist processor node.
    Routes to 'tools' if a tool call was made (e.g., query_database, ground_search).
    Routes to 'human_node' if a final synthesized answer was provided.
    """
    logging.info("\n--- ROUTING: route_processor_output ---")
    messages: List[BaseMessage] = state['messages']
    if not messages:
        #print("--- Routing Error: No messages found in route_processor_output ---")
        return END

    last_message = messages
    return CHATBOT_NODE
        
def route_after_tools(state: GraphState) -> Literal["sql_processor_node", "literature_search_node","human_node"]:
    """ Routes back to the specialist node that originally called the tool OR to human if unclear."""
    logging.info("\n--- ROUTING: route_after_tools ---")
    messages = state['messages']
    # The last message is the ToolMessage with results
    # The second to last message *should* be the AIMessage that made the tool call
    if len(messages) < 2:
         return HUMAN_NODE # Should not happen

    ai_message_that_called_tool = messages[-2]

    # This is heuristic: Check which LLM likely generated the tool call
    # A more robust way might be to add metadata to the state indicating the caller.
    # Simple approach: Check the tools called. If DB tools, assume SQL node. If search, assume Literature.

    # Check if the tool call originated from SQL LLM (by checking tool names)
    db_tool_names = {t.name for t in db_tools}
    called_tool_names = {call['name'] for call in ai_message_that_called_tool.tool_calls}

    if any(name in db_tool_names for name in called_tool_names):
        logging.info(f"--- Routing: Tool call was from SQL LLM. Called tools: {called_tool_names} ---")
        return SQL_NODE
    # Check if the tool call was Google Search (handled implicitly by LangChain for native tools)
    elif any(call['name'].lower() == 'googlesearchretrieval' for call in ai_message_that_called_tool.tool_calls):
        logging.info(f"--- Routing: Tool call was from Literature Search LLM. Called tools: {called_tool_names} ---")
        return LITERATURE_NODE
    else:
         # Fallback if the origin is unclear
         # Add a message indicating confusion?
         logging.warning(f"--- Routing Warning: Unclear which process should handle the tool results. Called tools: {called_tool_names} ---")
         state['messages'].append(SystemMessage(content="(System: Unclear which process should handle the tool results. Displaying results directly.)"))
         return HUMAN_NODE
    



# --- Build the Graph ---
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node(HUMAN_NODE, human_node)
workflow.add_node(CHATBOT_NODE, master_node.get_node)
workflow.add_node(SQL_NODE, sql_node.get_node)
workflow.add_node(LITERATURE_NODE, literature_search_node.get_node)
workflow.add_node(TOOL_NODE, tool_node)
workflow.add_node(PLOT_NODE, plot_node.get_node)


# --- Define Edges ---

# 1. Entry Point (Where the graph starts)
workflow.set_entry_point(CHATBOT_NODE) # Start with a hello input

# Add direct edges
workflow.add_edge(PLOT_NODE, CHATBOT_NODE)
workflow.add_edge(SQL_NODE, CHATBOT_NODE)
workflow.add_edge(LITERATURE_NODE, CHATBOT_NODE)
workflow.add_edge(HUMAN_NODE,END)
# 2. From Human Node


# 3. From Main Chatbot Node
workflow.add_conditional_edges(
    CHATBOT_NODE,
    route_chatbot_decision, # Function to decide based on chatbot output
    {
        SQL_NODE: SQL_NODE,                 # Route to SQL processor
        LITERATURE_NODE: LITERATURE_NODE,   # Route to Literature searcher
        TOOL_NODE: TOOL_NODE,               # Route to execute chatbot's tools (get_menu)
        HUMAN_NODE: HUMAN_NODE,             # Route to show chatbot's direct answer
        PLOT_NODE: PLOT_NODE,              # Route to plot node
        CHATBOT_NODE: CHATBOT_NODE,         # Route back to chatbot for further processing
        END: END                           # Route to end (though usually handled via human)
    }
)

# 6. From Tool Node - Route back to the appropriate processor
workflow.add_conditional_edges(
    TOOL_NODE,
    route_after_tools,
    {
        SQL_NODE: SQL_NODE,
        LITERATURE_NODE: LITERATURE_NODE,
        HUMAN_NODE: HUMAN_NODE # Fallback route
    }
)


# define state

# Initial state with a welcome message
initial_state = {
    "messages": [], # <-- Empty list
    "table": None,
    "answer": "",
    "finished": False,
    "request": None,
    "original_query": "",
    "answer_source": "Human",
    "trys": 0,
    "history": [] 
}
current_state = initial_state
config = {"recursion_limit": 100}


agent = workflow.compile()

