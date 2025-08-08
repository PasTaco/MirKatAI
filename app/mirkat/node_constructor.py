from altair import Chart
from langchain_core.messages import ( 
    AIMessage
    )
from google.genai.types import GenerateContentResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai import types
from google import genai 
from dotenv import load_dotenv
import os
from app.mirkat.plot_functions import PlotFunctons
import base64
import io
from app.mirkat.global_variables import SQL_QUERIES
from app.mirkat.instructions import Instructions
import re
import json
import sys
# save logs
import logging
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

load_dotenv()

# Get the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class node:
    def __init__(self, llm=None, instructions=None, functions=None, welcome=None, logging_key = None):
        self.llm = llm
        self.instructions = instructions
        self.functions = functions
        self.welcome = welcome
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.logging_key = logging_key
    def get_node(self, state):
        return None
    def log_message(self, message):
        """Log the message to the console."""
        logging.info(self.logging_key + message)
        
    


class HumanNode(node):
    def get_node(self, state):
        """Display the last model message to the user, and rec
        eive the user's input."""
        print("\n--- ENTERING: human_node ---")
        last_msg = state["messages"]
        answer = state["answer"]
        print(F"----- ANSWER: {answer} -------")
        if isinstance(last_msg, AIMessage) or isinstance(last_msg, GenerateContentResponse):
            if answer:
                print("Assistant:", answer)
                #display(Markdown(answer))
                state["answer"] = None
                #print()
            elif isinstance(last_msg, AIMessage):
                print("Assistant:", last_msg.content)
                #display(Markdown(last_msg.content))
            else:
                print("Assistant:", last_msg.text)
                #display(Markdown(last_msg.text))
        print("="*30)
        return state


class ChatbotNode(node):
    def __init__(self, llm=None, instructions=None, functions=None, welcome=None, complete_answer=None, limit_trys=5):
        super().__init__(llm, instructions, functions, welcome)
        self.llm_master=ChatGoogleGenerativeAI(model=self.llm)
        if complete_answer:
            self.complete_answer = complete_answer
        else:
            self.complete_answer = Instructions.format_answer.get_instruction()
        self.limit_trys = limit_trys

    def set_model(self, model):
        self.llm_master = ChatGoogleGenerativeAI(model=model)

    def run_model(self, messages):
        """Run the model with the given messages."""
        #print(f"--- Message going to the llm_master: {messages}---")
        response = self.llm_master.invoke(str(self.instructions) + messages)
        return response
    def run_model_for_compleatness(self, message_str):
        """Run the model to check if the answer is complete."""
        response = self.llm_master.invoke( str(self.complete_answer)+ message_str)
        return response
    
    def is_compleated(self, response, original_query, message,answer_source, trys, history = []):
        """Check if the response is complete."""
        logging.info("Checking if the response is complete")
        # Check if the response contains a SQL query
        if isinstance(original_query, str):
            original_query = original_query
        else:
            original_query = original_query.content
        message_eval = {'answer': response, 'original_query': original_query, "message": message.content, "answer_source": answer_source, "trys": trys, "try_limit": self.limit_trys, "history": history}
        message_str = str(message_eval)
        logging.info(f"Message for completeness check: {message_str}")
        is_compleate = self.run_model_for_compleatness(message_str)
        cleaned = re.sub(r"^```json\s*|\s*```$", "", is_compleate.content.strip())
        logging.info(f"Cleaned response: {cleaned}")
        return json.loads(cleaned)

        

    def get_node(self, state):
        """The chatbot with tools. A simple wrapper around the model's own chat interface."""
        #print("\n--- ENTERING: master_node ---")
        logging.info("Entering Master Node")
        messages = state['messages']
        logging.info(f"Messages received in Master Node: {messages}")
        #if isinstance(messages, list):
        #    messages = messages[-1]  # Get the last message if it's a list
        #if messages.content == "":
        #    messages = state['request']
        logging.info(f"Messages after processing: {messages}")
        answer = None
        orginal_query = state.get("original_query", None)
        compleate = False
        history = state.get("history", [])
        logging.info(f"History: {history}")
        if len(history) > 0:
            messages = history[-1]
        trys = state.get("trys", 0)
        finished = state.get("finished", False)
        if isinstance(messages, AIMessage) and trys > 0:
            logging.info(f"Cheking if the the response is complete: {messages.content}")
            response = state['request'].content
            answer_source = state.get("answer_source", None)
            dict_answer = self.is_compleated(response=response, 
                                        original_query=orginal_query,
                                        message=messages, answer_source=answer_source, trys=trys, history=history
                                        )

            answer = dict_answer.get("answer")
            returned_answer = dict_answer.get("return")
            logging.info(f"Returned answer: {returned_answer}")
            compleate = answer == "YES"
            if compleate:
                #returned_answer = "***FINISH***" + returned_answer
                finished = True

            messages =  AIMessage(content=returned_answer) 
            response = AIMessage(content=returned_answer)
        else:
            #print(f"--- Getting response from the human ---")
            logging.info(f"Getting question from the human")
            orginal_query = state['messages'][-1]
        logging.info(f"Compleate is {compleate}, trys is {trys}")
        if not compleate and trys < self.limit_trys:
            print(f"--- Original query: {orginal_query} ---")
            logging.info(f"Original query: {orginal_query}")
            # Normal operation: Invoke the master LLM for routing/response
            print("--- Calling Master Router LLM ---")
            logging.info("Calling Master Router LLM")
            # Always invoke with the system message + current history
            print(f"--- Message going to the llm_master: {messages}---")
            logging.info(f"Message going to the llm_master: {messages}")
            #messages_with_system = [{"type": "system", "content": MIRNA_ASSISTANT_SYSTEM_MESSAGE}] + state["messages"]
            response = self.run_model(str(messages))
            #if "***ANSWER_DIRECTLY***" in response.content.strip():
                #response = llm_master.invoke([ORIGINAL_MIRNA_SYSINT_CONTENT_MESSAGE] + messages)
            #    response.content = response.content.replace("***ANSWER_DIRECTLY***", "")
            #    answer = response.content
            messages = response
            print(f"--- Master Router Raw Response: {response.content} ---")
            logging.info(f"Master Router Raw Response: {response.content}")
            print(f"Exiting Master Router with response: {response.content}")
            logging.info(f"Exiting Master Router with response: {response.content}")
            print(f"--- Master Router Response: {response.content} ---")
            logging.info(f"Master Router Response: {response.content}")
            # Update state
        new_message = AIMessage(content="")
        if compleate or trys > self.limit_trys:
            #response.content = "***FINISH***" + response.content
            finished = True
            print(f"Final response is {response.content}")
            new_message = AIMessage(content=messages.content)
            
        logging.info(f"State before updating: {state}")
        logging.info(f"Response: {response.content}")
        logging.info(f"Answer: {answer}")
        logging.info(f"Messages: {new_message.content}")
        state = state | {
            #"messages": response.content , # Add the router's decision/response
            'messages':  new_message,
            "request": AIMessage(content=response.content), # Add the router's decision/response
            "answer": answer, # Update answer with the router's response
            "finished": finished, # Use .get for safety
            "original_query": orginal_query, # Add the original query
            "trys": trys + 1, # Increment the number of tries
            "answer_source": 'ChatbotNode', # Add the source of the answer
            "history": history + [messages], # Update history with the new message
        }
        return state



class SQLNode(node):
    def __init__(self, llm=None, instructions=None, functions=None,  welcome=None):
        super().__init__(llm, instructions, functions, welcome)
        self.set_model()

    def set_model(self):
        config_tools = types.GenerateContentConfig(
            system_instruction=self.instructions,
            tools=self.functions,
            temperature=0.0,
            )

        # Start a chat with automatic function calling enabled.
        self.chat = self.client.chats.create(
            model=self.llm,
            config=config_tools,
        )

    

    def run_model(self, messages):
        """Run the model with the given messages."""
        print(f"--- Message entering run model: {messages}---")
        logging.info(f"Message entering run model: {messages}")
        text = messages.content
        print (f"--- Message going to the sql model: {text}---")
        logging.info(f"Message going to the sql model: {text}")
        response = self. chat.send_message(text)
        return response

    def get_queries(self, callings):
        plotting_tools_instance = PlotFunctons(callings, '')
        queries = plotting_tools_instance.get_queries()
        return queries

    def get_node(self,state):
        """The sql llm that will check for the sql questions and get a json file in response."""

        print("--- Calling SQL Processor Node ---")
        logging.info("Calling SQL Processor Node")
        print("State: ", state)
        logging.info(f"State: {state}")
        history = state.get('history', [])
        # If history is empty, use the last message
        
        messages = state['request']
        if not messages:
            logging.warning("SQL processor called with no messages.")
            return state

        print("The type of the message is: ", type(messages))
        logging.info(f"The type of the message is: {type(messages)}")
        # check if it is GenerateContentResponse
        if isinstance(messages, GenerateContentResponse):
            print("The message is GenerateContentResponse, changing to AIMessage")
            logging.info("The message is GenerateContentResponse, changing to AIMessage")
            messages = AIMessage(content=messages.candidates[0].content)
        elif isinstance(messages, str):
            print("The message is str, changing to AIMessage")
            logging.info("The message is str, changing to AIMessage")
            messages = AIMessage(content=messages)
        elif isinstance(messages, AIMessage):
            pass
        else:
            print("The message is not str or AIMessage, changing to AIMessage")
            logging.info("The message is not str or AIMessage, changing to AIMessage")
            print("The type of the message is: ", type(messages))
            logging.info(f"The type of the message is: {type(messages)}")

        #print("The message sent to the SQL node is: ", messages)
        print("The message sent to the SQL node is: ", messages)
        logging.info(f"The message sent to the SQL node is: {messages}")
        response = self.run_model(messages)
        print(f"--- SQL Processor LLM Response: {response} ---")
        logging.info(f"SQL Processor LLM Response: {response}")
        print("Run get_queries")
        logging.info("Run get_queries")
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


class PlotNode(node):
    def __init__(self, llm=None, instructions=None, functions=None,  welcome=None):
        super().__init__(llm, instructions, functions, welcome)
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.set_model()

    

class LiteratureNode(node):
    def __init__(self, llm=None, instructions=None, functions=None,  welcome=None):
        super().__init__(llm, instructions, functions, welcome)
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.set_config()

    def set_config(self):
        self.config_with_search = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            )

    def run_model(self, user_query):
        """Run the model with the given messages."""
        logging.info(f"Running model with user query: {user_query}")
        response = self.client.models.generate_content(
            model=self.llm,
            contents=user_query,
            config=self.config_with_search,
            #system_instruction=LITERATURE_SYSTEM_INSTRUCTION_CONTENT, # Apply system instruction
        )
        return response
    
    def format_text(self, response):
        answer = response.text
        chunks = response.candidates[0].grounding_metadata.grounding_chunks
        supports = response.candidates[0].grounding_metadata.grounding_supports
        research_queries=response.candidates[0].grounding_metadata.web_search_queries
        lit_tools_instance = self.functions(chunks, supports, answer)

        answer=lit_tools_instance.process_references()
        bibliography=lit_tools_instance.create_bibliography()

        return answer, bibliography, research_queries
    
    def get_node(self, state):
        """Perform GroundSearch with UserQuery
        Returns:
        - Answer: Markdown formatted text answer from Ground Search iwth clickable references
        - Bibliography: References, link and website use to obtain the answer
        - ResearcQueries: Quesries used to perform GroundSearch"""
        #print("\n--- ENTERING: Literature node ---")
        logging.info("Entering Literature Node")
        user_query = state['request']
        #print(f"\n--- SEARCHING {user_query} with GroundSearch model ---")
        logging.info(f"Searching {user_query} with GroundSearch model")

        # --- Grounding Setup ---
        # Use the native Google Search tool for grounding
    

        # --- Model Selection ---
        ## gemini-2.0-flash is faster and return less issues compared to gemini-1.5-flash
        print("\n--- Performing: GroundSearch ---")
        logging.info("Performing GroundSearch")
        response = self.run_model(user_query.content)
        logging.info(f"GroundSearch Response: {response}")

        answer,bibliography, research_queries= self.format_text(response)

        
        #print(F"----- ANSWER: {answer} -------")

        history = state.get("history", [])
        message_text = answer + bibliography
        messageAI = AIMessage(content=message_text)
        return {**state,
        "messages":messageAI,
        "answer": AIMessage(content= answer+bibliography),
        "history": history + [messageAI]}