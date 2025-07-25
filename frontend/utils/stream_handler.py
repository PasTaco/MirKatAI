# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mypy: disable-error-code="unreachable"
import importlib
import json
import uuid
from collections.abc import Generator
from typing import Any
from urllib.parse import urljoin

import google.auth
import google.auth.transport.requests
import google.oauth2.id_token
import requests
import streamlit as st
import vertexai
from google.auth.exceptions import DefaultCredentialsError
from langchain_core.messages import AIMessage, ToolMessage
from vertexai import agent_engines

from frontend.utils.multimodal_utils import format_content
import logging
import os
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mirkat_handler.log'),
        logging.StreamHandler()
    ]
)
st.cache_resource.clear()


@st.cache_resource
def get_remote_agent(remote_agent_engine_id: str) -> Any:
    """Get cached remote agent instance."""
    # Extract location and engine ID from the full resource ID.
    parts = remote_agent_engine_id.split("/")
    project_id = parts[1]
    location = parts[3]
    vertexai.init(project=project_id, location=location)
    return agent_engines.AgentEngine(remote_agent_engine_id)


@st.cache_resource
def get_remote_url_config(url: str, authenticate_request: bool) -> dict[str, Any]:
    """Get cached remote URL agent configuration."""
    stream_url = urljoin(url, "stream_messages")
    creds, _ = google.auth.default()
    id_token = None
    if authenticate_request:
        auth_req = google.auth.transport.requests.Request()
        try:
            id_token = google.oauth2.id_token.fetch_id_token(auth_req, stream_url)
        except DefaultCredentialsError:
            creds.refresh(auth_req)
            id_token = creds.id_token
    return {
        "url": stream_url,
        "authenticate_request": authenticate_request,
        "creds": creds,
        "id_token": id_token,
    }


@st.cache_resource()
def get_local_agent(agent_callable_path: str) -> Any:
    """Get cached local agent instance."""
    module_path, class_name = agent_callable_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    agent = getattr(module, class_name)()
    agent.set_up()
    return agent


class Client:
    """A client for streaming events from a server."""

    def __init__(
        self,
        agent_callable_path: str | None = None,
        remote_agent_engine_id: str | None = None,
        url: str | None = None,
        authenticate_request: bool = False,
    ) -> None:
        """Initialize the Client with appropriate configuration.

        Args:
            agent_callable_path: Path to local agent class
            remote_agent_engine_id: ID of remote Agent engine
            url: URL for remote service
            authenticate_request: Whether to authenticate requests to remote URL
        """
        if url:
            remote_config = get_remote_url_config(url, authenticate_request)
            self.url = remote_config["url"]
            self.authenticate_request = remote_config["authenticate_request"]
            self.creds = remote_config["creds"]
            self.id_token = remote_config["id_token"]
            self.agent = None
        elif remote_agent_engine_id:
            self.agent = get_remote_agent(remote_agent_engine_id)
            self.url = None
        else:
            self.url = None
            if agent_callable_path is None:
                raise ValueError("agent_callable_path cannot be None")
            self.agent = get_local_agent(agent_callable_path)

    def log_feedback(self, feedback_dict: dict[str, Any], run_id: str) -> None:
        """Log user feedback for a specific run."""
        score = feedback_dict["score"]
        if score == "😞":
            score = 0.0
        elif score == "🙁":
            score = 0.25
        elif score == "😐":
            score = 0.5
        elif score == "🙂":
            score = 0.75
        elif score == "😀":
            score = 1.0
        feedback_dict["score"] = score
        feedback_dict["run_id"] = run_id
        feedback_dict["log_type"] = "feedback"
        feedback_dict.pop("type")
        url = urljoin(self.url, "feedback")
        headers = {
            "Content-Type": "application/json",
        }
        if self.url:
            url = urljoin(self.url, "feedback")
            headers = {
                "Content-Type": "application/json",
            }
            if self.authenticate_request:
                headers["Authorization"] = f"Bearer {self.id_token}"
            requests.post(
                url, data=json.dumps(feedback_dict), headers=headers, timeout=10
            )
        elif self.agent is not None:
            self.agent.register_feedback(feedback=feedback_dict)
        else:
            raise ValueError("No agent or URL configured for feedback logging")

    def stream_messages(
        self, data: dict[str, Any]
    ) -> Generator[dict[str, Any], None, None]:
        """Stream events from the server, yielding parsed event data."""
        if self.url:
            headers = {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }
            if self.authenticate_request:
                headers["Authorization"] = f"Bearer {self.id_token}"
            with requests.post(
                self.url, json=data, headers=headers, stream=True, timeout=60
            ) as response:
                for line in response.iter_lines():
                    if line:
                        try:
                            event = json.loads(line.decode("utf-8"))
                            yield event
                        except json.JSONDecodeError:
                            print(f"Failed to parse event: {line.decode('utf-8')}")
        elif self.agent is not None:
            yield from self.agent.stream_query(**data)


class StreamHandler:
    """Handles streaming updates to a Streamlit interface."""

    def __init__(self, st: Any, initial_text: str = "") -> None:
        """Initialize the StreamHandler with Streamlit context and initial text."""
        self.st = st
        self.tool_expander = st.expander("Tool Calls:", expanded=False)
        self.container = st.empty()
        self.text = initial_text
        self.tools_logs = initial_text

    def new_token(self, token: str) -> None:
        """Add a new token to the main text display."""
        self.text += token
        logging.info(f"STREAM HANDLER New Tockem self.text = {self.text}")
        self.container.markdown(format_content(self.text), unsafe_allow_html=True)
        

    def new_status(self, status_update: str) -> None:
        """Add a new status update to the tool calls expander."""
        self.tools_logs += status_update
        self.tool_expander.markdown(status_update)
        


class EventProcessor:
    """Processes events from the stream and updates the UI accordingly."""

    def __init__(self, st: Any, client: Client, stream_handler: StreamHandler) -> None:
        """Initialize the EventProcessor with Streamlit context, client, and stream handler."""
        self.st = st
        self.client = client
        self.stream_handler = stream_handler
        self.final_content = ""
        self.tool_calls: list[dict[str, Any]] = []
        self.current_run_id: str | None = None
        self.additional_kwargs: dict[str, Any] = {}

    def process_events(self) -> None:
        """Process events from the stream, handling each event type appropriately."""
        messages = self.st.session_state.user_chats[
            self.st.session_state["session_id"]
        ]["messages"]
        self.current_run_id = str(uuid.uuid4())
        # Set run_id in session state at start of processing
        self.st.session_state["run_id"] = self.current_run_id

        # --- Clear previous output before starting stream ---
        self.stream_handler.container.empty()
        self.stream_handler.tool_expander.empty()
        # ----------------------------------------------------

        stream = self.client.stream_messages(
            data={
                "input": {"messages": messages},
                "config": {
                    "run_id": self.current_run_id,
                    "metadata": {
                        "user_id": self.st.session_state["user_id"],
                        "session_id": self.st.session_state["session_id"],
                    },
                },
            }
        )
        # Each event is a tuple message, metadata. https://langchain-ai.github.io/langgraph/how-tos/streaming/#messages
        ignore_messages = True
        for message, _ in stream:
            if isinstance(message, dict):
                if message.get("type") == "constructor":
                    message = message["kwargs"]

                    # Handle tool calls - Accumulate info but DON'T display yet
                    if message.get("tool_calls"):
                        tool_calls = message["tool_calls"]
                        ai_message = AIMessage(content="", tool_calls=tool_calls)
                        self.tool_calls.append(ai_message.model_dump())
                    # Handle tool responses - Accumulate info but DON'T display yet
                    elif message.get("tool_call_id"):
                        content = message["content"]
                        tool_call_id = message["tool_call_id"]
                        tool_message = ToolMessage(
                            content=content, type="tool", tool_call_id=tool_call_id
                        ).model_dump()
                        self.tool_calls.append(tool_message)

                    # Handle AI responses - Accumulate content but DON'T display yet
                    elif content := message.get("content"):
                        #logging.info(f"Received list of content parts: {content}")
                        if '"answer": "YES"' in content:
                            ignore_messages = not ignore_messages
                            #logging.info("STREAM_HANDLER: Stop ignoring messages.")
                        if not ignore_messages or True:
                            if isinstance(content, list):
                                #print(f"STREAM_HANDLER: Received list of content parts: {content}")
                                self.final_content += "".join(str(part) for part in content)
                            else:
                                self.final_content += content
                        # (Streaming display commented out as per previous request)


        # --- Handle end of stream: Now display the final collected content ---
        if self.final_content:
            logging.info(f"STREAM_HANDLER: Final content collected: {self.final_content}")
            self.final_content = self.final_content.split("****FINAL_RESPONSE**** ")[-1]
            self.final_content = self.final_content.replace("image_save", "image")
            # ---- START: Added JSON parsing logic ----
            display_content = self.final_content # Default to the full content
            try:
                # Attempt to find and parse JSON within the final content
                # Handle potential markdown fences like ```json\n{...}\n```
                json_part = self.final_content
                if json_part.strip().startswith("```json"):
                    json_part = json_part.split("```json", 1)[1]
                elif json_part.strip().startswith("json\n"):
                     json_part = json_part.split("json\n", 1)[1]

                # Find the start and end of the JSON object
                start_index = json_part.find('{')
                end_index = json_part.rfind('}')
                if start_index != -1 and end_index != -1 and end_index > start_index:
                    json_str = json_part[start_index : end_index + 1]
                    parsed_json = json.loads(json_str)
                    if "return" in parsed_json:
                        display_content = parsed_json["return"]
                    else:
                         print("Parsed JSON, but 'return' key not found. Displaying full content.")
                else:
                     print("Could not find valid JSON object delimiters {}. Displaying full content.")

            except json.JSONDecodeError:
                print(f"Final content is not valid JSON or couldn't parse relevant part. Displaying full content.")
            except Exception as e:
                print(f"An unexpected error occurred during JSON parsing: {e}. Displaying full content.")
             # --- END: Added JSON parsing logic ---

            # Display the potentially extracted 'return' value or the full content
            # Use the 'display_content' variable here

            logging.info("STREAM_HANDLER: Getting image")
            image_path = self.get_picture()
            if image_path:
                display_content = display_content + image_path
                self.final_content = self.final_content + image_path
            self.stream_handler.container.markdown(format_content(display_content),
                                                   unsafe_allow_html=True)  # MODIFIED LINE
            logging.info(f"STREAM_HANDLER: Final content: {self.final_content}")
            # IMPORTANT: Store the ORIGINAL full final_content in the message history
            final_message = AIMessage(
                content=self.final_content, # Use original self.final_content here
                id=self.current_run_id,
                additional_kwargs={**self.additional_kwargs, "image_path": image_path
                                   }
            ).model_dump()
            session = self.st.session_state["session_id"]
            # Update messages in session state *after* processing
            self.st.session_state.user_chats[session]["messages"] = (
                self.st.session_state.user_chats[session]["messages"] + self.tool_calls
            )
            self.st.session_state.user_chats[session]["messages"].append(final_message)
            self.st.session_state.run_id = self.current_run_id # Ensure run_id is set

            # Display tool calls summary in the expander (optional)
            # (Code for displaying tool calls remains the same as before)
            if self.tool_calls:
                tool_log_str = ""
                for tool_call_msg in self.tool_calls:
                    if tc_list := tool_call_msg.get("tool_calls"):
                        for tc in tc_list:
                            name = tc.get('name', 'Unknown Tool')
                            args = tc.get('args', {})
                            tool_log_str += f"\n\nCalling tool: `{name}` with args: `{json.dumps(args)}`"
                    elif tool_call_msg.get("tool_call_id"):
                        tool_content = tool_call_msg.get("content", "No content in response")
                        tool_log_str += f"\n\nTool response: `{tool_content}`"
                if tool_log_str:
                    self.stream_handler.tool_expander.markdown(tool_log_str.strip())

        else:
            # Handle cases where there might be no final text content
            self.stream_handler.container.markdown("*No text content returned.*")
            if self.tool_calls:
                session = self.st.session_state["session_id"]
                self.st.session_state.user_chats[session]["messages"] = (
                    self.st.session_state.user_chats[session]["messages"] + self.tool_calls
                )
                # (Display tool calls summary as before)
                tool_log_str = ""
                for tool_call_msg in self.tool_calls:
                     if tc_list := tool_call_msg.get("tool_calls"):
                         for tc in tc_list:
                             name = tc.get('name', 'Unknown Tool')
                             args = tc.get('args', {})
                             tool_log_str += f"\n\nCalling tool: `{name}` with args: `{json.dumps(args)}`"
                     elif tool_call_msg.get("tool_call_id"):
                         tool_content = tool_call_msg.get("content", "No content in response")
                         tool_log_str += f"\n\nTool response: `{tool_content}`"
                if tool_log_str:
                    self.stream_handler.tool_expander.markdown(tool_log_str.strip())

            self.st.session_state.run_id = self.current_run_id # Ensure run_id is set

    def get_picture(self):

        if "<image>" in self.final_content and "</image>" in self.final_content:
            start_tag = self.final_content.find("<image>") + len("<image>")
            end_tag = self.final_content.find("</image>")
            image_path = self.final_content[start_tag:end_tag].strip()
            logging.info(f"STREAM_HANDLER: There is the image registered: {image_path}")
            if os.path.exists(image_path) and image_path.endswith(".svg"):
                logging.info(f"STREAMLIT HANDLER: Image exists")
            else:
                logging.warning(f"Image path does not exist: {image_path}")
            return image_path

        else:
            logging.info(f"STREAM_HANDLER: No image found, skipping")
        return None


def get_chain_response(st: Any, client: Client, stream_handler: StreamHandler) -> None:
    """Process the chain response update the Streamlit UI.

    This function initiates the event processing for a chain of operations,
    involving an AI model's response generation and potential tool calls.
    It creates an EventProcessor instance and starts the event processing loop.

    Args:
        st (Any): The Streamlit app instance, used for accessing session state
                 and updating the UI.
        client (Client): An instance of the Client class used to stream events
                        from the server.
        stream_handler (StreamHandler): An instance of the StreamHandler class
                                      used to update the Streamlit UI with
                                      streaming content.

    Returns:
        None

    Side effects:
        - Updates the Streamlit UI with streaming tokens and tool call information.
        - Modifies the session state to include the final AI message and run ID.
        - Handles various events like chain starts/ends, tool calls, and model outputs.
    """
    processor = EventProcessor(st, client, stream_handler)
    processor.process_events()
