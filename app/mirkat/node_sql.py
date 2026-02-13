import json
import os
from datetime import datetime
from hashlib import md5
from typing import Any, Tuple, Optional

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
    # Maximum characters for table content before saving to file
    MAX_TABLE_SIZE_CHARS = 5000
    
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
        inner_tries = 1
        while inner_tries <= 2:
            try:
                response = self.chat.send_message(text)
                inner_tries = 2
            except Exception as e:
                if inner_tries < 2:
                    self.log_message(f"Error sending message to SQL model: {e}. Rennuning for {inner_tries} time")
                else:
                    self.log_message(f"Error sending message to SQL model: {e}. No more retries.")
                    raise e
            finally:
                inner_tries = inner_tries + 1
                
        return response

    def get_queries(self, callings):
        plotting_tools_instance = PlotFunctons(callings, '')
        queries = plotting_tools_instance.get_queries()
        return queries

    def _process_query_results(self, queries: Any) -> Tuple[str, Optional[str]]:
        """Process query results and determine if they should be saved to a file.
        
        Args:
            queries: The query results to process
            
        Returns:
            Tuple of (summary_string, file_path_or_none)
            - For small results: (full_results_string, None)
            - For large results: (summary_string, file_path)
        """
        queries_str = str(queries)
        queries_length = len(queries_str)
        
        # Check if results are small enough to include directly
        if queries_length < self.MAX_TABLE_SIZE_CHARS:
            self.log_message(f"Query results size ({queries_length} chars) is below threshold. Adding to history.")
            return queries_str, None
        
        # Results are large - save to file and create summary
        self.log_message(f"Query results size ({queries_length} chars) exceeds threshold ({self.MAX_TABLE_SIZE_CHARS}). Saving to file.")
        
        try:
            # Create temp directory if it doesn't exist
            temp_dir = "/tmp/sql_results"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Generate unique filename with microseconds for better uniqueness
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            content_hash = md5(queries_str.encode()).hexdigest()[:8]
            filename = f"sql_results_{timestamp}_{content_hash}.json"
            filepath = os.path.join(temp_dir, filename)
            
            # Save full results to file
            with open(filepath, 'w') as f:
                if isinstance(queries, dict):
                    json.dump(queries, f, indent=2)
                else:
                    json.dump({"results": queries_str}, f, indent=2)
            
            self.log_message(f"Saved query results to: {filepath}")
            
            # Create summary
            summary_parts = []
            
            if isinstance(queries, dict):
                # Extract information from dict structure
                total_rows = 0
                columns = []
                sample_data = []
                
                for key, value in queries.items():
                    if isinstance(value, list):
                        total_rows += len(value)
                        if len(value) > 0 and isinstance(value[0], (list, tuple)):
                            # Get first few rows as sample
                            sample_data.append(f"{key}: {value[:3]}")
                        else:
                            sample_data.append(f"{key}: {value[:5]}")
                    
                summary_parts.append(f"Query results contain {len(queries)} {'query' if len(queries) == 1 else 'queries'}")
                if total_rows > 0:
                    summary_parts.append(f"with approximately {total_rows} total rows")
                if sample_data:
                    summary_parts.append(f"Sample data: {'; '.join(sample_data)}")
            else:
                # For non-dict queries, provide basic summary
                summary_parts.append(f"Query results: {queries_str[:500]}...")
            
            summary_parts.append(f"Full results saved to: {filepath}")
            summary = ". ".join(summary_parts)
            
            return summary, filepath
            
        except Exception as e:
            self.log_message(f"Error saving query results to file: {e}")
            # Fallback: truncate the results
            truncated = queries_str[:self.MAX_TABLE_SIZE_CHARS] + "... [truncated]"
            return truncated, None

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
        
        # Process query results (may save to file if large)
        queries_summary, file_path = self._process_query_results(queries)
        # Note: file_path is currently unused but available for future enhancements
        # (e.g., passing to plot node for accessing full data)

        return {**state,
            #"messages": response.content,
            "original_query": state["original_query"], # Add the router's decision/response
            "messages": AIMessage(content=""), # Add the router's decision/response
            "request": AIMessage(content=response.text), # Add the router's decision/response
            "table": queries, # Use .get for safety
            "answer": new_answer, # Return the potentially updated answer
            "finished": state.get("finished", False), # Use .get for safety
            "answer_source": 'SQL_NODE',
            "trys": state.get("trys", 0) + 1, # Use .get for safety
            "history": history + [queries_summary + response.text], # Update history with the new message
        }

