"""
app.py  —  RAG-Powered Document QA System
==========================================
Run with:  streamlit run app.py

Architecture:
  Upload PDFs/TXT  →  DocumentProcessor  →  VectorStore (FAISS) + BM25
                   →  HybridRetriever (RRF fusion)
                   →  RAGPipeline (LLM answer + citations)
                   →  Streamlit chat UI
"""

import os
import sys
import tempfile

import streamlit as st
from dotenv import load_dotenv

# ── Path setup (so src/* imports work) ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from document_processor import DocumentProcessor
from vector_store import VectorStore
from bm25_retriever import BM25Retriever
from hybrid_retriever import HybridRetriever
from llm_client import build_llm_client
from rag_pipeline import RAGPipeline
from session_state import init_state, reset_state

load_dotenv()

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document QA",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
  .chat-box { border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.75rem; line-height: 1.7; }
  .user-box { background: #EEF2FF; color: #1e1e2e; }
  .assistant-box { background: #F0FDF4; color: #1e1e2e; }
  .citation-card { border-left: 3px solid #6366F1; background: #F8F8FF;
                   border-radius: 6px; padding: 0.6rem 0.9rem; margin-top: 0.4rem;
                   font-size: 0.82rem; color: #444; }
  .badge { display: inline-block; background: #E0E7FF; color: #3730A3;
           border-radius: 99px; padding: 2px 10px; font-size: 0.75rem;
           font-weight: 600; margin-right: 6px; }
  .score-badge { background: #DCFCE7; color: #166534; }
  .metric-card { text-align: center; padding: 1rem; border-radius: 10px;
                 background: #F1F5F9; border: 1px solid #E2E8F0; }
  .metric-num { font-size: 1.8rem; font-weight: 700; color: #4F46E5; }
  .metric-label { font-size: 0.78rem; color: #64748B; margin-top: 2px; }
  .stButton>button { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Init session state ───────────────────────────────────────────────────
init_state()


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 RAG Document QA")
    st.markdown("*Advanced Hybrid Retrieval System*")
    st.divider()

    # --- LLM Config ---
    st.markdown("### ⚙️ LLM Settings")
    provider = st.selectbox("Provider", ["groq", "openai"], index=0,
                            help="Groq is FREE — get key at console.groq.com")
    if provider == "groq":
        model = st.selectbox(
    "Model",
    [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "gemma2-9b-it"
    ]
)
        api_key_input = st.text_input("Groq API Key", type="password",
                                      value=os.getenv("GROQ_API_KEY", ""),
                                      help="Free at console.groq.com")
    else:
        model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"])
        api_key_input = st.text_input("OpenAI API Key", type="password",
                                      value=os.getenv("OPENAI_API_KEY", ""))

    st.divider()

    # --- Chunking Config ---
    st.markdown("### 📐 Chunking Settings")
    chunk_size = st.slider("Chunk size (words)", 200, 1000, 500, 50)
    chunk_overlap = st.slider("Overlap (words)", 0, 200, 100, 10)
    top_k = st.slider("Chunks to retrieve (top-k)", 2, 10, 5)

    st.divider()

    # --- File Upload ---
    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF or TXT files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload one or more documents to query"
    )

    build_btn = st.button("🔨 Build Knowledge Base", use_container_width=True,
                          type="primary", disabled=not uploaded_files)

    if st.session_state.docs_processed:
        st.success(f"✅ {len(st.session_state.doc_names)} doc(s) loaded")
        if st.button("🗑️ Reset", use_container_width=True):
            reset_state()
            st.rerun()

    st.divider()
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    - **Embeddings**: `all-MiniLM-L6-v2` (local)  
    - **Dense**: FAISS IndexFlatIP  
    - **Sparse**: BM25 (rank-bm25)  
    - **Fusion**: Reciprocal Rank Fusion  
    - **LLM**: Groq / OpenAI  
    """)


# ── Build Pipeline ────────────────────────────────────────────────────────
if build_btn and uploaded_files:
    with st.spinner("⚙️ Processing documents and building index..."):
        # Save uploads to temp dir
        tmp_dir = tempfile.mkdtemp()
        file_paths = []
        for uf in uploaded_files:
            dst = os.path.join(tmp_dir, uf.name)
            with open(dst, "wb") as f:
                f.write(uf.getbuffer())
            file_paths.append(dst)

        # Process
        processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = processor.process_files(file_paths)

        if not chunks:
            st.error("No text extracted from the uploaded files. Try different documents.")
            st.stop()

        # Build FAISS
        vs = VectorStore(model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        vs.build(chunks)

        # Build BM25
        bm25 = BM25Retriever()
        bm25.build(chunks)

        # Hybrid retriever
        hybrid = HybridRetriever(vs, bm25)

        # LLM client
        if provider == "groq":
            os.environ["GROQ_API_KEY"] = api_key_input
            os.environ["LLM_PROVIDER"] = "groq"
        else:
            os.environ["OPENAI_API_KEY"] = api_key_input
            os.environ["LLM_PROVIDER"] = "openai"
        os.environ["LLM_MODEL"] = model
        os.environ["MAX_TOKENS"] = "1024"

        llm = build_llm_client()

        # Pipeline
        pipeline = RAGPipeline(retriever=hybrid, llm=llm, top_k=top_k)

        # Store in session
        st.session_state.pipeline = pipeline
        st.session_state.docs_processed = True
        st.session_state.doc_names = [uf.name for uf in uploaded_files]
        st.session_state.total_chunks = len(chunks)

    st.success(f"✅ Knowledge base ready! {len(chunks)} chunks indexed from {len(uploaded_files)} file(s).")
    st.rerun()


# ── Main UI ───────────────────────────────────────────────────────────────
st.markdown("# 📄 RAG-Powered Document QA")
st.markdown("*Hybrid Search (BM25 + FAISS) · Reciprocal Rank Fusion · Citation Tracking*")

if not st.session_state.docs_processed:
    # Welcome screen
    st.info("👈 Upload documents in the sidebar and click **Build Knowledge Base** to begin.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="metric-card">
          <div class="metric-num">🔍</div>
          <div class="metric-label">Hybrid BM25 + FAISS</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
          <div class="metric-num">🔀</div>
          <div class="metric-label">Reciprocal Rank Fusion</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
          <div class="metric-num">📌</div>
          <div class="metric-label">Source Citations</div></div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📖 How it works")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Step 1 — Upload**  
        Add PDF, TXT, or MD files via the sidebar.

        **Step 2 — Index**  
        Click *Build Knowledge Base*. The system:
        - Extracts text page-by-page with PyMuPDF  
        - Splits into overlapping chunks  
        - Creates a FAISS dense vector index  
        - Builds a BM25 sparse index  
        """)
    with c2:
        st.markdown("""
        **Step 3 — Query**  
        Ask any question. The system:  
        - Runs both dense + sparse retrieval  
        - Fuses results via Reciprocal Rank Fusion  
        - Sends top-k chunks to LLM with citations  
        - Returns a sourced, grounded answer  

        **Step 4 — Verify**  
        Each answer shows exact source, page, and excerpt.
        """)

else:
    # ── Stats bar ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-num">{len(st.session_state.doc_names)}</div>
          <div class="metric-label">Documents</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-num">{st.session_state.total_chunks}</div>
          <div class="metric-label">Chunks indexed</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-num">{len(st.session_state.chat_history) // 2}</div>
          <div class="metric-label">Questions asked</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-num">RRF</div>
          <div class="metric-label">Fusion mode</div></div>""", unsafe_allow_html=True)

    st.divider()

    # ── Chat history ──
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-box user-box">🧑 <strong>You:</strong><br>{msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-box assistant-box">🤖 <strong>Assistant:</strong><br>{msg["content"]}</div>',
                        unsafe_allow_html=True)
            # Show citations
            if msg.get("citations"):
                with st.expander(f"📌 {len(msg['citations'])} source(s) used", expanded=False):
                    for c in msg["citations"]:
                        st.markdown(
                            f'<div class="citation-card">'
                            f'<span class="badge">{c["ref"]}</span>'
                            f'<span class="badge score-badge">score: {c["score"]}</span>'
                            f'<strong>{c["source"]}</strong> — Page {c["page"]}<br>'
                            f'<em>"{c["excerpt"]}"</em>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

    # ── Input ──
    with st.form("chat_form", clear_on_submit=True):
        query = st.text_input(
            "Ask a question about your documents",
            placeholder="e.g. What are the main findings? Summarise section 3...",
        )
        submitted = st.form_submit_button("Ask →", use_container_width=True, type="primary")

    if submitted and query.strip():
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.spinner("🔍 Retrieving and generating answer..."):
            try:
                response = st.session_state.pipeline.query(query)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.answer,
                    "citations": response.citations,
                })
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {str(e)}\n\nCheck your API key in the sidebar.",
                    "citations": [],
                })
        st.rerun()

    # Clear chat
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()
