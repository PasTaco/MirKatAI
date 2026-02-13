from google.genai import types
from google.genai.types import Content, FunctionResponse, GenerateContentResponse, Part
from langchain_core.messages import AIMessage

from app.mirkat.global_variables import SQL_QUERIES
from app.mirkat.node_constructor import node
from app.mirkat.plot_functions import PlotFunctons

# save logs

class SQLNode(node):
    def __init__(self, llm=None, instructions=None, functions=None,  welcome=None,
                 max_result_rows=50, max_result_chars=5000):
        super().__init__(llm, instructions, functions, welcome, logging_key="SQL node.- ")
        self.max_result_rows = max_result_rows
        self.max_result_chars = max_result_chars
        self.set_model()

    def set_model(self):
        config_tools = types.GenerateContentConfig(
            system_instruction=self.instructions,
            tools=self.functions,
            temperature=0.0,
            max_output_tokens=600,
            # Disable automatic function calling to have manual control
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            )
        )

        # Start a chat with manual function calling control
        self.chat = self.client.chats.create(
            model=self.llm,
            config=config_tools,
        )

    def _has_function_call(self, response):
        """Check if response contains a function call."""
        if not response.candidates:
            return False

        candidate = response.candidates[0]
        if not hasattr(candidate, 'content') or not candidate.content:
            return False

        if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
            return False

        # Check if any part is a function call
        for part in candidate.content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                return True

        return False

    def _extract_function_call(self, response):
        """Extract the function call from response."""
        if not response.candidates:
            return None

        candidate = response.candidates[0]
        if not hasattr(candidate, 'content') or not candidate.content:
            return None

        if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
            return None

        # Find and return the first function call
        for part in candidate.content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                return part.function_call

        return None

    def _execute_function(self, function_call):
        """Execute the function call and return results."""
        function_name = function_call.name
        function_args = function_call.args or {}

        self.log_message(f"Executing function: {function_name} with args: {function_args}")

        # Find and execute the appropriate function
        # self.functions is a list of Python callables (from DBTools)
        for func in self.functions:
            if callable(func) and func.__name__ == function_name:
                try:
                    # Execute the function with the provided arguments
                    result = func(**function_args)
                    # Wrap the result in the expected format
                    return {'result': result}
                except Exception as e:
                    self.log_message(f"Error executing function {function_name}: {e}")
                    return {'error': str(e)}

        self.log_message(f"Function {function_name} not found in available functions")
        return {'error': f'Function {function_name} not found'}

    def _truncate_result_if_needed(self, result):
        """
        Truncate query results to prevent token exhaustion while preserving utility.

        Returns modified result with:
        - First N rows as samples
        - Total count
        - Instruction to use subquery approach
        """
        if not isinstance(result, dict) or 'result' not in result:
            return result

        result_list = result['result']

        # Handle empty results
        if not result_list:
            return result

        result_str_len = len(str(result))

        if len(result_list) > self.max_result_rows or result_str_len > self.max_result_chars:
            # Note: Instruction is intentionally generic since we don't have access to
            # the actual query context at truncation time. The model should remember
            # the query it just executed and use that for the subquery.
            truncated = {
                'result': result_list[:self.max_result_rows],
                'total_count': len(result_list),
                'showing': min(self.max_result_rows, len(result_list)),
                '_truncated': True,
                '_instruction': (
                    f'Results truncated: showing {min(self.max_result_rows, len(result_list))} of {len(result_list)} total rows. '
                    'DO NOT use these literal values in the next query. '
                    'Instead, use a subquery that reproduces the logic of your previous query, '
                    'e.g., WHERE column IN (SELECT column FROM ... WHERE ...)'
                )
            }

            self.log_message(
                f"Truncated large query result: {len(result_list)} rows "
                f"({result_str_len} chars) -> {self.max_result_rows} rows shown"
            )

            return truncated

        return result

    def _format_function_response(self, function_call, result):
        """Format result as Content for next iteration."""
        # Create a FunctionResponse Part
        function_response = FunctionResponse(
            name=function_call.name,
            response=result
        )

        # Create a Part with the function response
        part = Part(function_response=function_response)

        # Create Content with the part
        content = Content(parts=[part], role="user")

        return content

    def run_model(self, messages):
        """Run the model with controlled function calling to prevent token exhaustion."""
        self.log_message(f"Message entering run model: {messages}")
        text = messages.content
        self.log_message(f"Message going to the sql model: {text}")

        # Limit iterations to prevent infinite loops while allowing complex multi-step queries.
        # 10 iterations should be sufficient for most SQL queries that require multiple
        # function calls (e.g., list_tables, get_table_schema, describe_columns, execute_query).
        # If a query legitimately needs more iterations, increase this value or make it configurable.
        max_iterations = 10
        # Store all function calls and responses for history
        all_function_calls = []

        for iteration in range(max_iterations):
            self.log_message(f"Function calling iteration {iteration + 1}/{max_iterations}")

            inner_tries = 1
            response = None
            while inner_tries <= 2:
                try:
                    response = self.chat.send_message(text)
                    inner_tries = 2
                except Exception as e:
                    if inner_tries < 2:
                        self.log_message(f"Error sending message to SQL model: {e}. Rerunning for {inner_tries} time")
                        inner_tries += 1
                    else:
                        self.log_message(f"Error sending message to SQL model: {e}. No more retries.")
                        raise e

            # Check if model wants to call a function
            if self._has_function_call(response):
                function_call = self._extract_function_call(response)

                if function_call:
                    self.log_message(f"Model requested function call: {function_call.name}")

                    # Execute the function
                    result = self._execute_function(function_call)

                    # Store the original call and response for history
                    all_function_calls.append({
                        'function_call': function_call,
                        'result': result
                    })

                    # CRITICAL: Truncate result if too large
                    truncated_result = self._truncate_result_if_needed(result)

                    # Format the (possibly truncated) result as content for next iteration
                    text = self._format_function_response(function_call, truncated_result)
                else:
                    # No valid function call found, break
                    break
            else:
                # No more function calls, we're done
                self.log_message(f"Model finished after {iteration + 1} iterations")
                break

        # Reconstruct automatic_function_calling_history for backward compatibility
        # This allows existing code to extract queries using get_queries()
        response.automatic_function_calling_history = self._reconstruct_history(all_function_calls)

        return response

    def _reconstruct_history(self, all_function_calls):
        """Reconstruct the function calling history in the expected format."""
        history = []

        for call_info in all_function_calls:
            function_call = call_info['function_call']
            result = call_info['result']

            # Create a Content object with function_call part
            call_part = Part(function_call=function_call)
            call_content = Content(parts=[call_part], role="model")
            history.append(call_content)

            # Create a Content object with function_response part
            function_response = FunctionResponse(
                name=function_call.name,
                response=result
            )
            response_part = Part(function_response=function_response)
            response_content = Content(parts=[response_part], role="user")
            history.append(response_content)

        return history

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
            "history": [*history, str(queries) + response.text], # Update history with the new message
        }

