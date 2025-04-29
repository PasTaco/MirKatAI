import collections
import streamlit as st

class PlotFunctons:
    def __init__(self, callings, msg):
        self.callings = callings
        self.msg = msg

    def get_queries(self):
        """ From the response.automatic_function_calling_history, 
        get the queries that were exceuted with good results and saving it on a dictionary
        query:result.
        """
        queries = {}
        query_queue = collections.deque()
        for call in self.callings:
            for part in call.parts:#.content.parts:
            #call = [0]
                #print (part)
                if hasattr(part, 'function_call') and part.function_call:
                    #print(part.function_call.name)
                    if part.function_call.name == 'execute_query':
                        query = part.function_call.args['sql']
                        query_queue.append(query)
                if hasattr(part, 'function_response') and part.function_response:
                    if part.function_response.name =='execute_query':
                        #print(part.function_response.response)
                        popped_query = query_queue.popleft()
                        if 'result' in part.function_response.response:
                            queries[popped_query] = part.function_response.response
                        query= ''
        return queries


    def handle_response(self, tool_impl=None):
        """Stream output and handle any tool calls during the session."""
        msg = self.msg.candidates[0].content
        for part in msg.parts:
            if result := part.code_execution_result:
                st.markdown(f'### Result: {result.outcome}\n'
                                f'```\n{pformat(result.output)}\n```')

            elif code := part.executable_code:
                #display(Markdown(
                #    f'### Code\n```\n{code.code}\n```'))
                exec(code.code) 
            
            elif img := part.inline_data:
                st.image(img.data) # TO  BE TESTED!


