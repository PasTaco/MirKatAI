# sidebar.py
import streamlit as st

def render_sidebar(set_question_callback=None):
    """Render the sidebar with quick question chips."""

    st.sidebar.markdown("### Example questions")



    chips = [
        "What is MirKatAI?",
        "What are the targets of mir1?",
        "Which values are stored in the tissue table?",
        "Find the 5 most specific miRNAs for muscle tissue with their Tissue Specific Index (TSI).",
    ]

    for i, chip in enumerate(chips):
        if st.sidebar.button(chip, key=f"chip_{i}"):
            if set_question_callback:
                set_question_callback(chip)
