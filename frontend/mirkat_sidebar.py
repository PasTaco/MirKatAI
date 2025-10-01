# sidebar.py
import streamlit as st

def render_sidebar(set_question_callback=None):
    """Render the sidebar with quick question chips."""

    st.sidebar.markdown("### Example questions")



    # Render chips (as HTML links that trigger JS -> Streamlit events)
    chips = [
        "What is MirKatAI?",
        "What are the targets of mir-1?",
        "What is the presence of mir1 in muscle tissue?",
        "Find the 5 most specific miRNAs for muscle tissue",
    ]

    for i, chip in enumerate(chips):
        if st.sidebar.button(chip, key=f"chip_{i}"):
            if set_question_callback:
                set_question_callback(chip)
