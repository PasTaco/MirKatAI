import os
import streamlit as st
import pandas as pd
import uuid


from app.agent import agent
#### create or load vector database:
from PIL import Image

from typing import TypedDict # For GraphState type hinting
from langchain_core.messages import HumanMessage

from app.mirkat.instructions import Instructions

from mirkat_sidebar import render_sidebar

EMPTY_CHAT_NAME = "Empty chat"
_, WELCOME_MSG = Instructions.router.get_instruction()





if "messages" not in st.session_state:
    st.session_state.messages = []

def setup_page() -> None:
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="MirKatAI",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items=None,
    )

    # Only assign a UUID if one doesn't already exist in this session
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())

    # Load SVG logo
    logo = Image.open("frontend/assets/MiRkatAI.png")
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px;">
        """,
        unsafe_allow_html=True
    )
    col1, col2 = st.columns([1, 10])
    with col1:
        st.image(logo, width=75)
    with col2:
        st.markdown("## MirKatAI")

    # Once user_id is set
    st.markdown(WELCOME_MSG.content)

    
    # Define a callback to inject a question when sidebar button clicked
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    def set_question(q):
        st.session_state.pending_question = q

    # Render the sidebar
    render_sidebar(set_question_callback=set_question)




    

def messages_stream():
    """Stream messages from the session state."""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            yield st.chat_message("user").markdown(msg["content"])
        else:
            yield st.chat_message("assistant").markdown(msg["content"], unsafe_allow_html=True)

def get_user_input():
    """Get user input from the chat input box."""
    return st.chat_input("Ask a question...")



def main():
    """Main function to run the Streamlit app."""
    setup_page()

    # Display chat messages
    for message in messages_stream():
        pass  # Messages are displayed in the generator
    
    
    # Get user input
    question = get_user_input()
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None



    # If a question is asked, process it
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").markdown(question)
        state = {
            "session_id": st.session_state.user_id,
            "messages": [st.session_state.messages[-1]['content']],
        }
        workflow = agent
        result = workflow.invoke(state)
        assistant_response = result.get("messages", "").content
        assistant_bibliography = result.get("bibliography", "").content
        # strip ****FINAL RESPONSE**** prefix if present
        if assistant_response.startswith("****FINAL_RESPONSE****"):
            assistant_response = assistant_response.replace("****FINAL_RESPONSE****", "").strip()
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        st.chat_message("assistant").markdown(assistant_response, unsafe_allow_html=True)
        if assistant_bibliography != "":
            st.session_state.messages.append({"role": "assistant", "content": assistant_bibliography})
            st.chat_message("assistant").markdown(assistant_bibliography, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
