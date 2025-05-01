# original libraries
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

# import libraries
import pandas as pd
import os
from pprint import pprint
from typing import Any, Dict, List, Literal, Optional, TypedDict # Grouped and sorted alphabetically
from pprint import pformat
from app.mirkat.instructions import Instructions
from app.mirkat.sql_functions import (
    MySqlConnection,
    DBTools
)
from app.mirkat.literature_functions import LiteratureTools
from app.mirkat.plot_functions import PlotFunctons

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
import google.ai.generativelanguage as genai_types
import base64
import io
## load env variables and set up gemini API key:

from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

LOCATION = "europe-west1"
LLM = "gemini-2.0-flash"
LLM = "gemini-2.5-flash-preview-04-17"
#LLM = "gemini-2.0-flash-lite"





# istantiate llms for nodes

llm_master = ChatGoogleGenerativeAI(model=LLM)


###### define instructions for nodes


MIRNA_ASSISTANT_SYSTEM_MESSAGE, WELCOME_MSG = Instructions.router.get_instruction()
SQL_INSTRUCTIONS = Instructions.sql.get_instruction()


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




# Assume db_conn, mirkat_columns_description, and mirkat_tables_description are available
db_tools_instance = DBTools(db_conn, mirkat_tables_desctiption, mirkat_columns_desctiption)

# OR you can make a list of methods if you want:
db_tools = [
    db_tools_instance.list_tables,
    db_tools_instance.get_table_schema,
    db_tools_instance.describe_columns,
    db_tools_instance.describe_tabes,
    db_tools_instance.execute_query
]



##### Define SQL agent

client = genai.Client(api_key=GOOGLE_API_KEY)

config_tools = types.GenerateContentConfig(
    system_instruction=SQL_INSTRUCTIONS,
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
    print("\n--- ENTERING: human_node ---")
    last_msg = state["messages"]
    answer = state["answer"]
    support = None
    chunks = None
    print(F"----- ANSWER: {answer} -------")
    if isinstance(last_msg, AIMessage) or isinstance(last_msg, GenerateContentResponse):
        if answer:
            print("Assistant:", answer)
            #display(Markdown(answer))
            state["answer"] = None
            #print()
        #else:
            #print("Assistant:", last_msg.content)
            #display(Markdown(last_msg.content))
    print("="*30)
    return state
    #user_input = input("User: ")

    # If it looks like the user is trying to quit, flag the conversation
    # as over.
    #if user_input.strip().lower() in {"q", "quit", "exit", "goodbye", "bye", "thanks that's all"}:
    #    state["finished"] = True

    #return state | {"messages": [("user", user_input)]}
    #return {
    #    "messages": state["messages"] +[HumanMessage(content=user_input)],
    #    "table": state["table"],
    #    "answer": state["answer"],
    #    "finished": state["finished"]}

def chatbot_with_tools(state: GraphState) -> GraphState:
    """The chatbot with tools. A simple wrapper around the model's own chat interface."""
    print("\n--- ENTERING: master_node ---")

    
    messages = state['messages']
    answer = None

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
        #messages_with_system = [{"type": "system", "content": MIRNA_ASSISTANT_SYSTEM_MESSAGE}] + state["messages"]
        response = llm_master.invoke(str(MIRNA_ASSISTANT_SYSTEM_MESSAGE) + str(state["messages"]))
        if "***ANSWER_DIRECTLY***" in response.content.strip():
            #response = llm_master.invoke([ORIGINAL_MIRNA_SYSINT_CONTENT_MESSAGE] + messages)
            response.content = response.content.replace("***ANSWER_DIRECTLY***", "")
            answer = response.content
        print(f"--- Master Router Raw Response: {response.content} ---")

    # Update state
    state = state | {
        #"messages": response.content , # Add the router's decision/response
        "messages": AIMessage(content=response.content), # Add the router's decision/response
        "answer": answer, # Update answer with the router's response
        "finished": state.get("finished", False), # Use .get for safety
    }
    return state
SQL_QUERIES = {}
def sql_processor_node(state: GraphState) -> GraphState:
    """The sql llm that will check for the sql questions and get a json file in response."""
    print("--- Calling SQL Processor Node ---")
    messages = state['messages']
    if not messages:
        # Should ideally not happen if routing is correct
        #print("Warning: SQL processor called with no messages.")
        # Return unchanged state or add an error message? For now, return unchanged.
        return state

    print("The type of the message is: ", type(messages))
    # check if it is GenerateContentResponse
    if isinstance(messages, GenerateContentResponse):
        print("The message is GenerateContentResponse, changing to AIMessage")
        messages = AIMessage(content=messages.candidates[0].content)
    elif isinstance(messages, str):
        print("The message is str, changing to AIMessage")
        messages = AIMessage(content=messages)

    #print("The message sent to the SQL node is: ", messages)
    print("The message sent to the SQL node is: ", messages)
    response = chat.send_message(messages.content)
    print(f"--- SQL Processor LLM Response: {response} ---")
    print("Run get_queries")
    callings = response.automatic_function_calling_history
    plotting_tools_instance = PlotFunctons(callings, '')
    queries = plotting_tools_instance.get_queries()
    SQL_QUERIES.update(queries)
    #handle_response(response)
    #response = sql_llm_with_db_tools.invoke([SQL_SYSTEM_INSTRUCTION] + messages)
    #print(f"--- SQL Processor LLM Response: {response} ---")
    
    
    new_answer = state.get("answer", "")
    
    if isinstance(response, AIMessage) and response.content and not response.tool_calls:
         print("The response is AIMessage")
         new_answer = response.content # Update answer if it's a direct text response
    elif isinstance(response, GenerateContentResponse):
        print("The response is GenerateContentResponse")
        new_answer = response.text
    elif isinstance(response, str):
        print("The response is str")
        new_answer = response
    new_messages = messages + [AIMessage(content=new_answer)]
    #print(f"--- Answer from SQL Processor LLM Response: {new_answer} ---")
    return {
        #"messages": response.content,
        "messages": AIMessage(content="This was the answer from SQL node, please format and give to the user: "+response.text), # Add the router's decision/response
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
    user_query = state["messages"]
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
    lit_tools_instance = LiteratureTools(chunks, supports, answer)

    answer=lit_tools_instance.process_references()
    bibliography=lit_tools_instance.create_bibliography()


    
    #print(F"----- ANSWER: {answer} -------")


    return {
        "messages": answer,
        "table" : state["table"],
        "answer": answer,
        "bibliography": bibliography,
        "research_queries": research_queries,
        "finished": state["finished"]}

buf = io.BytesIO()
def plot_node(state:GraphState) -> GraphState:
    print("\n--- ENTERING: plot_node ---")
    print(f"State values: {state.keys()}")

    messages = state['messages'].content
    queries = SQL_QUERIES # state['table']

    response_plot = plotter_model.send_message(str(queries) + "The code to plot, should save the final figure on variable figure." + messages)
    plotting_tools_instance = PlotFunctons('', response_plot)
    content = response_plot.candidates[0].content
    print(content)
    
    plot = plotting_tools_instance.handle_response()
    
    plot.savefig(buf, format='png') # Or another format like 'jpeg'
    plot.savefig("plot", format='svg')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    #print(f"Image base64: {image_base64}")
    buf.close()
    response_plot.candidates[0].content.parts[0].text =  f"binary_image: {image_base64}"
    #mime_type = "image/png"
    #data_uri = f"data:{mime_type};base64,{image_base64}"
    #image_content_part = {
    #"type": "image_url",
    #"image_url": {
    #    "url": data_uri
    #}
#}
    answer = ''
    for part in response_plot.candidates[0].content.parts:
        if part.text is not None:
            answer = answer + f"{part.text}\n"
    print(f"--- Plot Node LLM Response: {answer} ---")
    
    
    # if isinstance(response_plot.candidates[0].content, str):
    #     response_plot.candidates[0].content = [
    #         {"type": "text", "text": response_plot.candidates[0].content},
    #         image_content_part
    #         ]
    # elif isinstance(response_plot.candidates[0].content, list):
    #     response_plot.candidates[0].content.append(image_content_part)
    # else: # Handle unexpected content type or create new
    #     response_plot.candidates[0].content = [image_content_part]

    print("--Leaving plot node---")
    return {**state,
            "messages": AIMessage(content=answer + f"binary_image: {image_base64}"),
            "answer": answer
           #"messages":state["messages"] + [AIMessage(content=answer)]
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
    return END
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
    print("\n--- ROUTING: route_chatbot_decision ---")
    messages: List[BaseMessage] = state['messages']
    if not messages:
        #print("--- Routing Error: No messages found in route_chatbot_decision ---")
        return END # Or raise error

    last_message = messages
    if not isinstance(last_message, AIMessage) :
        print(f"--- Routing Warning: Expected AIMessage, got {type(last_message)}. Routing to Human. ---")
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
    elif "***FINISH***" in content or state.get("finished"): # Check flag too
        #print("--- Routing: Master Router to END ---")
        return END
    elif "***ANSWER_DIRECTLY***" in content:
        content = content.replace("***ANSWER_DIRECTLY***", "")
        print(f"--- The answer directly was: {content}")
         #print("--- Routing: Master Router to Human (Direct Answer) ---")
         # Remove the keyword itself before showing to human
        #  state['messages'][-1].content = content.replace("***ANSWER_DIRECTLY***", "").strip()
        #  # If the content is *only* the keyword, maybe add a placeholder?
        #  #if not state['messages'][-1].content:
        #  x = state['messages'][-2].content[0]['text']
        #  print(f"\n\n\n\nThis is the message puto!!!{x}\n\n\n\n")
        #  #print (f"--- The messages directly was: {x}")
        #  answer = llm_master.invoke(x) #"Okay, let me answer that." # Or similar
        state['messages'].content = str(content)
        #  #print (f"--- The answer directly was: {answer}")
        state['answer'] = str(content)#.response.candidates[0].content.parts[0].text
        print(f"\n\n\n BEFORE CALLING HUMAN NODE \n\n\n\n")
        return HUMAN_NODE
    elif "***PLOT***" in content:
        #print("---- Routing to plot node ----")
        return PLOT_NODE
    else:
         # Assume it's a direct answer or clarification question if no keyword is found
         #print("--- Routing: Master Router to Human (Direct Answer) ---")
         # Remove potential keywords just in case they were missed but shouldn't be shown
         state['messages'].content = content.replace("***ROUTE_TO_SQL***", "").replace("***ROUTE_TO_LITERATURE***", "").replace("***FINISH***", "").replace("***ANSWER_DIRECTLY***", "").strip()
         print(f"\n\n\n BEFORE CALLING HUMAN NODE  (2) \n\n\n\n")
         return HUMAN_NODE

# Router 3: After a Specialist Processor Node (`sql_processor_node`, `literature_search_node`)
def route_processor_output(state: GraphState) -> Literal["chatbot_router","human_node", "__end__"]:
    """
    Inspects the last message from a specialist processor node.
    Routes to 'tools' if a tool call was made (e.g., query_database, ground_search).
    Routes to 'human_node' if a final synthesized answer was provided.
    """
    print("\n--- ROUTING: route_processor_output ---")
    messages: List[BaseMessage] = state['messages']
    if not messages:
        #print("--- Routing Error: No messages found in route_processor_output ---")
        return END

    last_message = messages

    #if not isinstance(last_message, AIMessage):
    #    print(f"--- Routing Warning: Expected AIMessage from processor, got {type(last_message)}. Routing to Human. ---")
    #    return HUMAN_NODE
    # Otherwise, the processor provided its final synthesized answer
    #else:
    #    print("--- Routing: Processor to Human ---")
    return CHATBOT_NODE
        
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
workflow.add_node(PLOT_NODE, plot_node)


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
        PLOT_NODE: PLOT_NODE,              # Route to plot node
        END: END                           # Route to end (though usually handled via human)
    }
)

# 4. From SQL Processor Node
workflow.add_conditional_edges(
    SQL_NODE,
    route_processor_output, # Function to decide based on SQL processor output
    {
        CHATBOT_NODE: CHATBOT_NODE, # Route to main chatbot (if needed)
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


workflow.add_edge(PLOT_NODE, END)



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

