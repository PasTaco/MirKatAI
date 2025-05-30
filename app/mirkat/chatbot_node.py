from langchain_core.messages import ( 
    AIMessage
    )
from langchain_google_genai import ChatGoogleGenerativeAI
from app.mirkat.instructions import Instructions
import re
import json
from app.mirkat.node_constructor import node
# save logs


class ChatbotNode(node):
    def __init__(self, llm=None, instructions=None, functions=None, welcome=None, complete_answer=None, limit_trys=5):
        super().__init__(llm=llm, instructions=instructions, functions=functions, welcome=welcome, logging_key="Master node.- ")
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
        self.log_message("Checking if the response is complete")
        # Check if the response contains a SQL query
        if isinstance(original_query, str):
            original_query = original_query
        else:
            original_query = original_query.content
        message_eval = {'answer': response, 'original_query': original_query, "message": message.content, "answer_source": answer_source, "trys": trys, "try_limit": self.limit_trys, "history": history}
        message_str = str(message_eval)
        self.log_message(f"Message for completeness check: {message_str}")
        is_compleate = self.run_model_for_compleatness(message_str)
        self.log_message(f"Response from completeness check: {is_compleate.content}")
        cleaned = re.sub(r"^```json\s*|\s*```$", "", is_compleate.content.strip())
        self.log_message(f"Cleaned response: {cleaned}")
        return json.loads(cleaned)

        

    def get_node(self, state):
        """The chatbot with tools. A simple wrapper around the model's own chat interface."""
        #print("\n--- ENTERING: master_node ---")
        self.log_message("Entering Master Node")
        messages = state['messages']
        self.log_message(f"Messages received in Master Node: {messages}")
        #if isinstance(messages, list):
        #    messages = messages[-1]  # Get the last message if it's a list
        #if messages.content == "":
        #    messages = state['request']
        self.log_message(f"Messages after processing: {messages}")
        answer = None
        orginal_query = state.get("original_query", None)
        compleate = False
        history = state.get("history", [])
        self.log_message(f"History: {history}")
        if len(history) > 0:
            messages = history[-1]
        trys = state.get("trys", 0)
        finished = state.get("finished", False)
        if isinstance(messages, AIMessage) and trys > 0:
            self.log_message(f"Cheking if the the response is complete: {messages.content}")
            response = state['request'].content
            answer_source = state.get("answer_source", None)
            dict_answer = self.is_compleated(response=response, 
                                        original_query=orginal_query,
                                        message=messages, answer_source=answer_source, trys=trys, history=history
                                        )

            answer = dict_answer.get("answer")
            returned_answer = dict_answer.get("return")
            self.log_message(f"Returned answer: {returned_answer}")
            compleate = answer == "YES"
            if compleate:
                #returned_answer = "***FINISH***" + returned_answer
                finished = True

            messages =  AIMessage(content=returned_answer) 
            response = AIMessage(content=returned_answer)
        else:
            #print(f"--- Getting response from the human ---")
            self.log_message(f"Getting question from the human")
            orginal_query = state['messages'][-1]
        self.log_message(f"Compleate is {compleate}, trys is {trys}")
        if not compleate and trys < self.limit_trys:
            print(f"--- Original query: {orginal_query} ---")
            self.log_message(f"Original query: {orginal_query}")
            # Normal operation: Invoke the master LLM for routing/response
            print("--- Calling Master Router LLM ---")
            self.log_message("Calling Master Router LLM")
            # Always invoke with the system message + current history
            print(f"--- Message going to the llm_master: {messages}---")
            self.log_message(f"Message going to the llm_master: {messages}")
            #messages_with_system = [{"type": "system", "content": MIRNA_ASSISTANT_SYSTEM_MESSAGE}] + state["messages"]
            response = self.run_model(str(messages))
            #if "***ANSWER_DIRECTLY***" in response.content.strip():
                #response = llm_master.invoke([ORIGINAL_MIRNA_SYSINT_CONTENT_MESSAGE] + messages)
            #    response.content = response.content.replace("***ANSWER_DIRECTLY***", "")
            #    answer = response.content
            messages = response
            print(f"--- Master Router Raw Response: {response.content} ---")
            self.log_message(f"Master Router Raw Response: {response.content}")
            print(f"Exiting Master Router with response: {response.content}")
            self.log_message(f"Exiting Master Router with response: {response.content}")
            print(f"--- Master Router Response: {response.content} ---")
            self.log_message(f"Master Router Response: {response.content}")
            # Update state
        new_message = AIMessage(content="")
        if compleate or trys > self.limit_trys:
            #response.content = "***FINISH***" + response.content
            finished = True
            print(f"Final response is {response.content}")
            new_message = AIMessage(content=messages.content)
            
        self.log_message(f"State before updating: {state}")
        self.log_message(f"Response: {response.content}")
        self.log_message(f"Answer: {answer}")
        self.log_message(f"Messages: {new_message.content}")
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

