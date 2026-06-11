"""
session_state.py
Centralised init and access helpers for Streamlit session_state.
Prevents KeyError on first run and keeps app.py clean.
"""

import streamlit as st


def init_state():
    defaults = {
        "pipeline": None,         # RAGPipeline instance
        "chat_history": [],       # [{role, content, citations}]
        "docs_processed": False,
        "doc_names": [],
        "total_chunks": 0,
        "processing": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_state():
    for key in ["pipeline", "chat_history", "docs_processed", "doc_names", "total_chunks"]:
        if key in st.session_state:
            del st.session_state[key]
    init_state()
