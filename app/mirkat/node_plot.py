from altair import Chart
from langchain_core.messages import ( 
    AIMessage
    )
from google.genai import types
from google import genai 
from dotenv import load_dotenv
from app.mirkat.plot_functions import PlotFunctons
import base64
import io, os
from app.mirkat.global_variables import SQL_QUERIES
from app.mirkat.node_constructor import node

load_dotenv()

# Get the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class PlotNode(node):
    def __init__(self, llm=None, instructions=None, functions=None,  welcome=None):
        super().__init__(llm, instructions, functions, welcome, logging_key="Plot node.- ")
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.set_model()

    def set_model(self):
        config_with_code = types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution())],
            temperature=0.0,
            )
        self.model = self.client.chats.create(model=self.llm, config=config_with_code)

    def run_model(self, messages):
        """Run the model with the given messages."""
        #print(f"--- Message going to the llm_master: {messages}---")
        self.log_message(f"Message going to the node: {messages}")
        response_plot = self.model.send_message(messages)
        return response_plot
    
    def handle_response(self, response_plot):
        plotting_tools_instance = PlotFunctons('', response_plot)
        plotting_tools_instance.handle_response()

    def get_node(self, state):
        messages = state['request']
        self.log_message(f"Messages received in PlotNode: {messages}")
        queries = SQL_QUERIES # state['table']

        response_plot = self.run_model(str(queries) + self.instructions + messages.content)
        
        plotting_tools_instance = PlotFunctons('', response_plot)        
        plot = plotting_tools_instance.handle_response()
        
        answer = ''
        for part in response_plot.candidates[0].content.parts:
            if part.text is not None:
                answer = answer + f"{part.text}\n"
        answer_b = answer
        if plot:
            buf = io.BytesIO()
            if not isinstance(plot, Chart):
                plot.savefig(buf, format='png') # Or another format like 'jpeg'
            elif isinstance(plot, Chart):
                plot.save(buf, format='json')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            # response_plot.candidates[0].content.parts[0].text =  f"binary_image: {image_base64}"
            answer_b = answer + f"binary_image: {image_base64}"
        history = state.get("history", [])
        return {**state,
                "messages": AIMessage(content=""),
                "answer": answer,
                "request": AIMessage(content=answer_b),
                "answer_source": 'PlotNode',
                "history": history + [messages], # Update history with the new message
            }
    
    
