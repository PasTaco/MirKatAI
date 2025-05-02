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


# Load .env file
load_dotenv()

# Get the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class node:
    def __init__(self, llm=None, instructions=None, functions=None, welcome=None):
        self.llm = llm
        self.instructions = instructions
        self.functions = functions
        self.welcome = welcome
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
    def get_node():
        return None
    


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
    def __init__(self, llm=None, instructions=None, functions=None, welcome=None):
        super().__init__(llm, instructions, functions, welcome)
        self.llm_master=ChatGoogleGenerativeAI(model=self.llm)

    def set_model(self, model):
        self.llm_master = ChatGoogleGenerativeAI(model=model)

    def run_model(self, messages):
        """Run the model with the given messages."""
        #print(f"--- Message going to the llm_master: {messages}---")
        response = self.llm_master.invoke(messages)
        return response
    
    def get_node(self,state):
        """The chatbot with tools. A simple wrapper around the model's own chat interface."""
        print("\n--- ENTERING: master_node ---")
        # Get the model
        
        messages = state['messages']
        answer = None

        # Check if this is the very first turn (no messages yet)
        if not messages:
            # Generate the welcome message directly
            print("--- Generating Welcome Message ---")
            if not self.welcome:
                self.welcome = "Welcome to the chatbot! How can I assist you today?"
            response = AIMessage(content=self.welcome)
        else:
            # Normal operation: Invoke the master LLM for routing/response
            print("--- Calling Master Router LLM ---")
            # Always invoke with the system message + current history
            print(f"--- Message going to the llm_master: {messages}---")
            #messages_with_system = [{"type": "system", "content": MIRNA_ASSISTANT_SYSTEM_MESSAGE}] + state["messages"]
            response = self.run_model(str(self.instructions) + str(state["messages"]))
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
        #print(f"--- Message going to the llm_master: {messages}---")
        response = self.chat.send_message(messages)
        return response

    def get_queries(self, callings):
        plotting_tools_instance = PlotFunctons(callings, '')
        queries = plotting_tools_instance.get_queries()
        return queries

    def get_node(self,state):
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
        response = self.run_model(messages)
        print(f"--- SQL Processor LLM Response: {response} ---")
        print("Run get_queries")
        callings = response.automatic_function_calling_history
        queries = self.get_queries(callings)
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
        # new_messages = messages + [AIMessage(content=new_answer)]
        #print(f"--- Answer from SQL Processor LLM Response: {new_answer} ---")
        return {
            #"messages": response.content,
            "messages": AIMessage(content="This was the answer from SQL node, please format and give to the user: "+response.text), # Add the router's decision/response
            "table": queries, # Use .get for safety
            "answer": new_answer, # Return the potentially updated answer
            "finished": state.get("finished", False), # Use .get for safety
        }


class PlotNode(node):
    def __init__(self, llm=None, instructions=None, functions=None,  welcome=None):
        super().__init__(llm, instructions, functions, welcome)
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.set_model()

    def set_model(self):
        config_with_code = types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution())],
            temperature=0.0,
            )
        self.plotter_model = self.client.chats.create(model=self.llm, config=config_with_code)

    def run_model(self, messages):
        """Run the model with the given messages."""
        #print(f"--- Message going to the llm_master: {messages}---")
        response_plot = self.plotter_model.send_message(messages)
        return response_plot
    
    def handle_response(self, response_plot):
        plotting_tools_instance = PlotFunctons('', response_plot)
        plotting_tools_instance.handle_response()

    def get_node(self, state):
            messages = state['messages'][-1].content
            queries = state['table']

            response_plot = self.run_model(str(queries) + messages)
            self.handle_response(response_plot)
            answer = ''
            for part in response_plot.candidates[0].content.parts:
                if part.text is not None:
                    answer = answer + f"{part.text}\n"
            
            return {**state,
                "messages":state["messages"] + [AIMessage(content=answer)]
                }
    

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
        #print(f"--- Message going to the llm_master: {messages}---")
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
        user_query = state["messages"]
        #print(f"\n--- SEARCHING {user_query} with GroundSearch model ---")

        # --- Grounding Setup ---
        # Use the native Google Search tool for grounding
    

        # --- Model Selection ---
        ## gemini-2.0-flash is faster and return less issues compared to gemini-1.5-flash
        print("\n--- Performing: GroundSearch ---")

        response = self.run_model(user_query)


        answer,bibliography, research_queries= self.format_text(response)

        
        #print(F"----- ANSWER: {answer} -------")


        return {**state,
            "messages": answer,
            "answer": answer,
            "bibliography": bibliography,
            "research_queries": research_queries}