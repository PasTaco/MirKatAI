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
    def escape_newlines_in_json_string(self, json_str: str) -> str:
        # Replace literal newlines inside JSON string values only
        def replacer(match):
            s = match.group(0)  # The full quoted string
            s_inner = s[1:-1]  # Remove quotes
            s_inner = s_inner.replace('\n', '\\n').replace('\r', '\\r')
            return f'"{s_inner}"'

        pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"'
        return re.sub(pattern, replacer, json_str)

    def extract_json_from_markdown(self, content: str) -> str:

        # Extract fenced JSON block
        match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Fallback: extract raw JSON object
            match = re.search(r'\{(?:[^{}]|(?R))*\}', content, re.DOTALL)  # recursive pattern if supported
            if not match:
                # simple fallback without recursion (non-greedy)
                match = re.search(r'\{.*?\}', content, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                raise ValueError("No valid JSON block found in markdown.")

        # Escape literal newlines inside JSON string values
        json_str = self.escape_newlines_in_json_string(json_str)

        # Optionally try loading to check valid JSON
        try:
            _ = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Extracted JSON is invalid: {e}")

        return json_str
        
