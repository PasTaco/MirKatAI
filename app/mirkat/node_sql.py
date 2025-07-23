from langchain_core.messages import ( 
    AIMessage
    )
from google.genai.types import GenerateContentResponse
from google.genai import types
from app.mirkat.plot_functions import PlotFunctons
from app.mirkat.global_variables import SQL_QUERIES
from app.mirkat.node_constructor import node


# save logs

class SQLNode(node):
    def __init__(self, llm=None, instructions=None, functions=None,  welcome=None):
        super().__init__(llm, instructions, functions, welcome, logging_key="SQL node.- ")
        self.set_model()

    def set_model(self):
        config_tools = types.GenerateContentConfig(
            system_instruction=self.instructions,
            tools=self.functions,
            temperature=0.0,
            max_output_tokens=600,
            )

        # Start a chat with automatic function calling enabled.
        self.chat = self.client.chats.create(
            model=self.llm,
            config=config_tools,
        )

    

    def run_model(self, messages):
        """Run the model with the given messages."""
        self.log_message(f"Message entering run model: {messages}")
        text = messages.content
        self.log_message(f"Message going to the sql model: {text}")
        new_try = True
        while new_try:
            try:
                response = self.chat.send_message(text)
            except Exception as e:
                new_try = False
                self.log_message(f"Error sending message to SQL model: {e}. Rennuning")
        return response

    def get_queries(self, callings):
        plotting_tools_instance = PlotFunctons(callings, '')
        queries = plotting_tools_instance.get_queries()
        return queries

    def get_node(self,state):
        """The sql llm that will check for the sql questions and get a json file in response."""

        self.log_message("Calling SQL Processor Node")
        self.log_message(f"State: {state}")
        history = state.get('history', [])
        # If history is empty, use the last message
        
        messages = state['request']
        if not messages:
            self.log_message("SQL processor called with no messages.")
            return state

        self.log_message(f"The type of the message is: {type(messages)}")
        # check if it is GenerateContentResponse
        if isinstance(messages, GenerateContentResponse):
            self.log_message("The message is GenerateContentResponse, changing to AIMessage")
            messages = AIMessage(content=messages.candidates[0].content)
        elif isinstance(messages, str):
            self.log_message("The message is str, changing to AIMessage")
            messages = AIMessage(content=messages)
        elif isinstance(messages, AIMessage):
            pass
        else:
            self.log_message("The message is not str or AIMessage, changing to AIMessage")
            self.log_message(f"The type of the message is: {type(messages)}")

        self.log_message(f"The message sent to the SQL node is: {messages}")
        response = self.run_model(messages)
        self.log_message(f"SQL Processor LLM Response: {response}")
        self.log_message("Run get_queries")
        callings = response.automatic_function_calling_history
        queries = self.get_queries(callings)
        SQL_QUERIES.update(queries)
        new_answer = AIMessage(content=response.text)
        history = state.get("history", [])

        return {
            #"messages": response.content,
            "original_query": state["original_query"], # Add the router's decision/response
            "messages": AIMessage(content=""), # Add the router's decision/response
            "request": AIMessage(content=response.text), # Add the router's decision/response
            "table": queries, # Use .get for safety
            "answer": new_answer, # Return the potentially updated answer
            "finished": state.get("finished", False), # Use .get for safety
            "answer_source": 'SQL_NODE',
            "trys": state.get("trys", 0) + 1, # Use .get for safety
            "history": history + [messages], # Update history with the new message
        }

