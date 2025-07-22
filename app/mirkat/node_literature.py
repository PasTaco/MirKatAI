from app.mirkat.node_constructor import node
from altair import Chart
from langchain_core.messages import ( 
    AIMessage
    )
from google.genai import types
from google import genai 
from dotenv import load_dotenv
import os
# save logs
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mirkat.log'),
        logging.StreamHandler()
    ]
)
load_dotenv()

# Get the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")




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
        logging.info("Entering Literature Node")
        user_query = state['request']
        logging.info(f"Searching {user_query} with GroundSearch model")

        # --- Grounding Setup ---
        # Use the native Google Search tool for grounding
    

        # --- Model Selection ---
        ## gemini-2.0-flash is faster and return less issues compared to gemini-1.5-flash
        logging.info("Performing GroundSearch")
        response = self.run_model(user_query.content)
        logging.info(f"GroundSearch Response: {response}")

        answer,bibliography, research_queries= self.format_text(response)

        
        history = state.get("history", [])
        message_text = answer + bibliography
        messageAI = AIMessage(content=message_text)
        return {**state,
        "messages":messageAI,
        "answer": AIMessage(content= answer+bibliography),
        "history": history + [messageAI]}