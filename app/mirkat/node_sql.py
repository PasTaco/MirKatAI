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
    def __init__(self, llm=None, instructions=None, functions=None,  welcome=None,
                 max_result_rows=50, max_result_chars=5000):
        super().__init__(llm, instructions, functions, welcome, logging_key="SQL node.- ")
        
        # Store configuration
        self.max_result_rows = max_result_rows
        self.max_result_chars = max_result_chars
        
        # Store original functions for reference
        self.original_functions = functions
        
        # Wrap functions with truncation logic
        self.functions = self._wrap_functions_with_truncation(functions) if functions else None
        
        self.set_model()

    def _wrap_functions_with_truncation(self, functions):
        """
        Wrap database functions to truncate large results before they reach the chat context.
        
        This prevents token exhaustion by intercepting function results at the source,
        ensuring only manageable data sizes are added to the conversation history.
        
        Args:
            functions: List of database tool functions
            
        Returns:
            List of wrapped functions with same signatures but truncated outputs
        """
        wrapped = []
        
        for func in functions:
            # Only wrap execute_query - other functions return small metadata
            if hasattr(func, '__name__') and func.__name__ == 'execute_query':
                self.log_message("Wrapping execute_query with truncation logic")
                wrapped_func = self._create_truncating_wrapper(func)
                wrapped.append(wrapped_func)
            else:
                # Pass through other functions unchanged
                wrapped.append(func)
        
        return wrapped

    def _create_truncating_wrapper(self, original_func):
        """
        Create a wrapper function that truncates large query results.
        
        The wrapper:
        1. Calls the original execute_query function
        2. Checks if results exceed size thresholds
        3. If large: returns truncated version with metadata and instructions
        4. If small: returns original results unchanged
        
        Args:
            original_func: The original execute_query function to wrap
            
        Returns:
            Wrapped function with same signature but truncation logic
        """
        
        def truncating_execute_query(*args, **kwargs):
            """
            Wrapped version of execute_query that truncates large results.
            
            When results are truncated, adds:
            - total_count: Original number of rows
            - _truncated: Boolean flag indicating truncation occurred
            - _instruction: Guidance for model to use subqueries instead of literal values
            """
            # Call the original function
            result = original_func(*args, **kwargs)
            
            # Check if truncation is needed
            if not isinstance(result, dict) or 'result' not in result:
                return result
            
            result_list = result.get('result', [])
            result_str_len = len(str(result))
            row_count = len(result_list)
            
            # Determine if truncation is needed
            needs_truncation = (
                row_count > self.max_result_rows or 
                result_str_len > self.max_result_chars
            )
            
            if needs_truncation:
                self.log_message(
                    f"Truncating query result: {row_count} rows "
                    f"({result_str_len} chars) → {self.max_result_rows} rows"
                )
                
                # Create truncated response with metadata
                truncated_result = {
                    'result': result_list[:self.max_result_rows],
                    'columns': result.get('columns', []),
                    'total_count': row_count,
                    'showing': min(self.max_result_rows, row_count),
                    '_truncated': True,
                    '_instruction': (
                        f'Results truncated: showing {min(self.max_result_rows, row_count)} '
                        f'of {row_count} total rows. '
                        'IMPORTANT: Do not use these literal values in the next query. '
                        'Instead, use a subquery approach: '
                        'WHERE column IN (SELECT ... FROM ... WHERE ...) '
                        'This prevents token exhaustion and is more efficient.'
                    )
                }
                
                return truncated_result
            
            # No truncation needed, return original
            return result
        
        # Preserve function metadata for the API
        truncating_execute_query.__name__ = original_func.__name__
        truncating_execute_query.__doc__ = original_func.__doc__
        
        # Copy over any other attributes the SDK might need
        if hasattr(original_func, '__annotations__'):
            truncating_execute_query.__annotations__ = original_func.__annotations__
        
        return truncating_execute_query

    def _truncate_result_if_needed(self, result):
        """
        Public method to test truncation logic independently.
        Used by wrapper and can be called directly for testing.
        
        Args:
            result: Query result dictionary
            
        Returns:
            Truncated result if needed, original otherwise
        """
        if not isinstance(result, dict) or 'result' not in result:
            return result
        
        result_list = result.get('result', [])
        row_count = len(result_list)
        
        if row_count > self.max_result_rows:
            return {
                'result': result_list[:self.max_result_rows],
                'columns': result.get('columns', []),
                'total_count': row_count,
                'showing': min(self.max_result_rows, row_count),
                '_truncated': True,
                '_instruction': (
                    f'Results truncated: showing {min(self.max_result_rows, row_count)} '
                    f'of {row_count} rows. Use subqueries for efficient queries.'
                )
            }
        
        return result


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
            "history": history + [str(queries) + response.text], # Update history with the new message
        }

