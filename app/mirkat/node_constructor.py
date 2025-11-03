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
    def __init__(self, llm=None, instructions=None, functions=None, welcome=None, logging_key = None, user=None):
        self.llm = llm
        self.instructions = instructions
        self.functions = functions
        self.welcome = welcome
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.logging_key = logging_key
        if user is None:
            self.user = ""
        else:
            self.user = user
    def set_user(self, user):
        self.user = user
    def get_node(self, state):
        return None
    def log_message(self, message):
        """Log the message to the console."""
        if type(message) is not str:
            message = str(message)
        logging.info("User:" + self.user + " " + self.logging_key + " " + message)

    def cript_links(self, original_answer: str) -> tuple[str, dict]:
        """ This Function will take the text that contain the vertex link to the soruces, it is going to map the link to a short string like [source1], [source2], etc.
        It will save the original link in a dictionary and return the text with the mapped links.
        The text link will have a fromat

        [[4](https://vertexaisearch.cloud.google.com/grounding-api-redirect/longstring)]

        or accumulate multiople links like
         [[4](https://vertexaisearch.cloud.google.com/grounding-api-redirect/longstring),[5](https://vertexaisearch.cloud.google.com/grounding-api-redirect/other_longstring)]

        Args:
            original_answer (str): The original text with the links.
        Returns:
            tuple[str, dict]: The text with the mapped links and the dictionary with the original links
        """
        BASE_URL_RE = r"https://vertexaisearch\.cloud\.google\.com/grounding-api-redirect/[^\)]+"
        source_dict = {}
        modified_answer = original_answer

        # ---Match the entire [[[1](url),[2](url),...,[n](url)]] block ---
        multi_pattern = (
            r"(\[\[(?:\[\d+\]\(" + BASE_URL_RE + r"\))(?:,\s*\[\d+\]\(" + BASE_URL_RE + r"\))*\]\])"
        )

        for full_block in re.findall(multi_pattern, original_answer):
            # Extract each [n](url) inside the block
            inner_links = re.findall(r"\[(\d+)\]\((" + BASE_URL_RE + r")\)", full_block)
            if not inner_links:
                continue

            source_keys = []
            for match_num, full_link in inner_links:
                source_key = f"[source{match_num}]"
                if source_key not in source_dict:
                    source_dict[source_key] = full_link
                source_keys.append(source_key)

            replacement_string = "[" + ",".join(source_keys) + "]"
            modified_answer = modified_answer.replace(full_block, replacement_string, 1)

        # ---Handle remaining single [[n](url)]] links ---
        single_pattern = r"\[\[(\d+)\]\((" + BASE_URL_RE + r")\)\]"
        for match_num, full_link in re.findall(single_pattern, original_answer):
            source_key = f"[source{match_num}]"
            if source_key not in source_dict:
                source_dict[source_key] = full_link
            link_block = f"[[{match_num}]({full_link})]"
            modified_answer = modified_answer.replace(link_block, source_key, 1)

        return modified_answer, source_dict
    
    def decrypt_links(self, answer_with_links: str, source_dict: dict) -> str:
        """ This Function will take the text that contain the mapped links like [source1], [source2], etc.
        It will replace the mapped links with the original links from the source_dict.

        Args:
            answer_with_links (str): The text with the mapped links.
            source_dict (dict): The dictionary with the original links.
        Returns:
            str: The text with the original links.
        """
        modified_answer = answer_with_links
        for source_key, full_link in source_dict.items():
            # replace [source1[ with just [[1](full_link)]]]
            source_name = source_key.replace("source", "")
            modified_answer = modified_answer.replace(source_key, f"[{source_name}({full_link})]")
        return modified_answer

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
            match = re.search(r'\{.*?\}', content, re.DOTALL) # recursive pattern if supported
            if not match:
                # simple fallback without recursion (non-greedy)
                match = re.search(r'\{.*?\}', content, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                raise ValueError(f"No valid JSON block found in markdown. Input string: {content}")

        # Escape literal newlines inside JSON string values
        json_str = self.escape_newlines_in_json_string(json_str)
        # check for missing " and }
        if json_str.count('"') % 2 != 0:
            json_str += '"'
        if json_str.count('{') > json_str.count('}'):
            json_str += '}'


        # Optionally try loading to check valid JSON
        try:
            _ = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Extracted JSON is invalid: {e}, string: {json_str}")

        return json_str
        
