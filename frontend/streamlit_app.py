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

# mypy: disable-error-code="arg-type"
import base64
import json
import uuid
from collections.abc import Sequence
from functools import partial
from typing import Any

import streamlit as st
from langchain_core.messages import HumanMessage
from google.genai.types import GenerateContentResponse

from streamlit_feedback import streamlit_feedback

from frontend.side_bar import SideBar
from frontend.style.app_markdown import MARKDOWN_STR
from frontend.utils.local_chat_history import LocalChatMessageHistory
from frontend.utils.message_editing import MessageEditing
from frontend.utils.multimodal_utils import format_content, get_parts_from_files
from frontend.utils.stream_handler import Client, StreamHandler, get_chain_response

from PIL import Image
USER = "my_user"
EMPTY_CHAT_NAME = "Empty chat"


def setup_page() -> None:
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="MirKatAI",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    # Load SVG logo
    logo = Image.open("frontend/assets/MiRkatAI.png")
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; align-items: center;">
        """,
        unsafe_allow_html=True
    )
    col1, col2 = st.columns([1, 10])
    with col1:
        st.image(logo, width=75)
    with col2:
        st.markdown("## MirKatAI")


def initialize_session_state() -> None:
    """Initialize the session state with default values."""
    if "user_chats" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state.uploader_key = 0
        st.session_state.run_id = None
        st.session_state.user_id = USER
        st.session_state["gcs_uris_to_be_sent"] = ""
        st.session_state.modified_prompt = None
        st.session_state.session_db = LocalChatMessageHistory(
            session_id=st.session_state["session_id"],
            user_id=st.session_state["user_id"],
        )
        st.session_state.user_chats = (
            st.session_state.session_db.get_all_conversations()
        )
        st.session_state.user_chats[st.session_state["session_id"]] = {
            "title": EMPTY_CHAT_NAME,
            "messages": [],
        }


def display_messages() -> None:
    """Display all messages in the current chat session."""
    messages = st.session_state.user_chats[st.session_state["session_id"]]["messages"]
    if "answer" in st.session_state.user_chats[st.session_state["session_id"]]:
        answer = st.session_state.user_chats[st.session_state["session_id"]]["answer"]
        print(f"---- Entering display_messages with answer: {answer}----")
    tool_calls_map = {}  # Map tool_call_id to tool call input
    #print(f"---- Entering display_messages with messages: {messages}----")
    for i, message in enumerate(messages):
        print(f"for enaumerate: {i} message: {message}")
        if message["type"] in ["ai", "human"] and message["content"]:
            print("Display chat message")
            display_chat_message(message, i)
        elif message.get("tool_calls"):
            # Store each tool call input mapped by its ID
            for tool_call in message["tool_calls"]:
                tool_calls_map[tool_call["id"]] = tool_call
        elif message["type"] == "tool":
            # Look up the corresponding tool call input by ID
            tool_call_id = message["tool_call_id"]
            if tool_call_id in tool_calls_map:
                display_tool_output(tool_calls_map[tool_call_id], message)
            else:
                st.error(f"Could not find tool call input for ID: {tool_call_id}")
        else:
            st.error(f"Unexpected message type: {message['type']}")
            st.write("Full messages list:", messages)
            raise ValueError(f"Unexpected message type: {message['type']}")


def display_chat_message(message: dict[str, Any], index: int) -> None:
    """Display a single chat message with edit, refresh, and delete options."""
    chat_message = st.chat_message(message["type"])
    print(f"---- Entering display_chat_message with message: {message}----")
    with chat_message:
        raw_content = message["content"]
        display_content = raw_content # Default display is the raw content
        plot = False

        # ---- START: Add JSON parsing logic for AI messages ----
        if message["type"] == "ai" and isinstance(raw_content, str): # Only parse AI string content
            try:
                # Attempt to find and parse JSON within the final content
                json_part = raw_content
                if json_part.strip().startswith("```json"):
                    json_part = json_part.split("```json", 1)[1]
                elif json_part.strip().startswith("json\n"):
                    json_part = json_part.split("json\n", 1)[1]

                start_index = json_part.find('{')
                end_index = json_part.rfind('}')
                if start_index != -1 and end_index != -1 and end_index > start_index:
                    json_str = json_part[start_index : end_index + 1]
                    parsed_json = json.loads(json_str)
                    if "return" in parsed_json:
                        display_content = parsed_json["return"] # Use the extracted value
                        print(f"Extracted 'return' value for display: {display_content}")
                    else:
                         print("Parsed JSON, but 'return' key not found. Displaying full content.")
                else:
                    print("Could not find valid JSON object delimiters {}. Displaying full content.")

            except json.JSONDecodeError:
                print(f"Content is not valid JSON or couldn't parse relevant part. Displaying full content.")
            except Exception as e:
                print(f"An unexpected error occurred during JSON parsing: {e}. Displaying full content.")
        # ---- END: Added JSON parsing logic ----

        # --- Handle potential image data (assuming it's separate from JSON) ---
        # Note: This assumes the image data isn't *inside* the JSON you want to parse.
        # If it is, the parsing logic needs adjustment.
        message_to_display = display_content
        if isinstance(display_content, str) and "binary_image" in display_content:
             plot = True
             image_binary = display_content.split("binary_image")[1].strip()
             message_to_display = display_content.split("binary_image")[0] # Text part
        elif not isinstance(display_content, str):
            # If parsing resulted in non-string (e.g. the dict itself if 'return' not found), handle appropriately
            message_to_display = format_content(display_content) # Use format_content for safety

        # --- Display the processed content ---
        st.markdown(format_content(message_to_display), unsafe_allow_html=True) # Use message_to_display

        if plot:
            image_binary_bytes = base64.b64decode(image_binary)
            print(f"---- Displaying image ----")
            st.image(
                image_binary_bytes,
                caption="Generated Image",
                use_column_width=True,
                output_format="auto",
            )
            plot = False

        # --- Display buttons (logic remains the same) ---
        col1, col2, col3 = st.columns([2, 2, 94])
        # Pass the RAW content to the edit buttons if needed
        display_message_buttons(message, index, col1, col2, col3, raw_content_for_editing=raw_content)


def display_message_buttons(
    message: dict[str, Any], index: int, col1: Any, col2: Any, col3: Any, raw_content_for_editing: Any
) -> None:
    """Display edit, refresh, and delete buttons for a chat message."""
    edit_button = f"{index}_edit"
    refresh_button = f"{index}_refresh"
    delete_button = f"{index}_delete"
    # Decide what content to use for editing - might need the raw content
    content_for_editing = (
        raw_content_for_editing
        if isinstance(raw_content_for_editing, str)
        else json.dumps(raw_content_for_editing) # Or handle non-string raw content appropriately
    )
    # Ensure edit uses the original raw content if needed
    if isinstance(message["content"], list): # Handle multimodal input for editing
         # Find the text part for editing, might need adjustment based on your structure
         text_parts = [part["text"] for part in message["content"] if part["type"] == "text"]
         content_for_editing = text_parts[0] if text_parts else ""


    with col1:
        st.button(label="✎", key=edit_button, type="primary")
    if message["type"] == "human":
        with col2:
            st.button(
                label="⟳",
                key=refresh_button,
                type="primary",
                on_click=partial(MessageEditing.refresh_message, st, index, content_for_editing), # Use content_for_editing
            )
        with col3:
            st.button(
                label="X",
                key=delete_button,
                type="primary",
                on_click=partial(MessageEditing.delete_message, st, index),
            )

    if st.session_state[edit_button]:
        st.text_area(
            "Edit your message:",
            value=content_for_editing, # Use content_for_editing
            key=f"edit_box_{index}",
            on_change=partial(MessageEditing.edit_message, st, index, message["type"]),
        )


def display_tool_output(
    tool_call_input: dict[str, Any], tool_call_output: dict[str, Any]
) -> None:
    """Display the input and output of a tool call in an expander."""
    # Use st.expander directly within display_messages if preferred,
    # or ensure this function is called correctly.
    # Let's assume it's called from display_messages:
    with st.expander(label=f"Tool Call: `{tool_call_input.get('name', 'Unknown Tool')}`", expanded=False):
         st.markdown("**Input:**")
         st.json(tool_call_input.get('args', {}))
         st.markdown("**Output:**")
         # The tool output content might be JSON string or actual dict/list
         output_content = tool_call_output.get('content', '')
         try:
             # Try to pretty-print if it's a JSON string
             parsed_output = json.loads(output_content)
             st.json(parsed_output)
         except (json.JSONDecodeError, TypeError):
             # Otherwise, display as text
             st.code(str(output_content), language=None)


def handle_user_input(side_bar: SideBar) -> None:
    """Process user input, generate AI response, and update chat history."""
    prompt = st.chat_input() or st.session_state.modified_prompt
    if prompt:
        st.session_state.modified_prompt = None
        parts = get_parts_from_files(
            upload_gcs_checkbox=st.session_state.checkbox_state,
            uploaded_files=side_bar.uploaded_files,
            gcs_uris=side_bar.gcs_uris,
        )
        st.session_state["gcs_uris_to_be_sent"] = ""
        parts.append({"type": "text", "text": prompt})
        st.session_state.user_chats[st.session_state["session_id"]]["messages"].append(
            HumanMessage(content=parts).model_dump()
        )

        display_user_input(parts)
        generate_ai_response(
            remote_agent_engine_id=side_bar.remote_agent_engine_id,
            agent_callable_path=side_bar.agent_callable_path,
            url=side_bar.url_input_field,
            authenticate_request=side_bar.should_authenticate_request,
        )
        update_chat_title()
        if len(parts) > 1:
            st.session_state.uploader_key += 1
        st.rerun()


def display_user_input(parts: Sequence[dict[str, Any]]) -> None:
    """Display the user's input in the chat interface."""
    human_message = st.chat_message("human")
    with human_message:
        existing_user_input = format_content(parts)
        st.markdown(existing_user_input, unsafe_allow_html=True)


def generate_ai_response(
    remote_agent_engine_id: str | None = None,
    agent_callable_path: str | None = None,
    url: str | None = None,
    authenticate_request: bool = False,
) -> None:
    """Generate and display the AI's response to the user's input."""
    ai_message = st.chat_message("ai")
    with ai_message:
        status = st.status("Generating answer🤖")
        stream_handler = StreamHandler(st=st)
        client = Client(
            remote_agent_engine_id=remote_agent_engine_id,
            agent_callable_path=agent_callable_path,
            url=url,
            authenticate_request=authenticate_request,
        )
        get_chain_response(st=st, client=client, stream_handler=stream_handler)
        status.update(label="Finished!", state="complete", expanded=False)


def update_chat_title() -> None:
    """Update the chat title if it's currently empty."""
    if (
        st.session_state.user_chats[st.session_state["session_id"]]["title"]
        == EMPTY_CHAT_NAME
    ):
        st.session_state.session_db.set_title(
            st.session_state.user_chats[st.session_state["session_id"]]
        )
    st.session_state.session_db.upsert_session(
        st.session_state.user_chats[st.session_state["session_id"]]
    )


def display_feedback(side_bar: SideBar) -> None:
    """Display a feedback component and log the feedback if provided."""
    if st.session_state.run_id is not None:
        feedback = streamlit_feedback(
            feedback_type="faces",
            optional_text_label="[Optional] Please provide an explanation",
            key=f"feedback-{st.session_state.run_id}",
        )
        if feedback is not None:
            client = Client(
                remote_agent_engine_id=side_bar.remote_agent_engine_id,
                agent_callable_path=side_bar.agent_callable_path,
                url=side_bar.url_input_field,
                authenticate_request=side_bar.should_authenticate_request,
            )
            client.log_feedback(
                feedback_dict=feedback,
                run_id=st.session_state.run_id,
            )


def main() -> None:
    """Main function to set up and run the Streamlit app."""
    setup_page()
    initialize_session_state()
    side_bar = SideBar(st=st)
    side_bar.init_side_bar()
    display_messages()
    handle_user_input(side_bar=side_bar)
    display_feedback(side_bar=side_bar)


if __name__ == "__main__":
    main()
