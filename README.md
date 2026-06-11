# 📄 RAG-Powered Document QA System

> **Advanced Retrieval-Augmented Generation** with Hybrid Search (BM25 + FAISS), Reciprocal Rank Fusion, and Citation Tracking.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red) ![FAISS](https://img.shields.io/badge/FAISS-CPU-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🏗️ Architecture

```
PDF/TXT Upload
     │
     ▼
DocumentProcessor (PyMuPDF)
     │  sliding-window chunking
     ▼
┌────────────────────────────────┐
│  Dense Index    │  Sparse Index │
│  FAISS (cosine) │  BM25 Okapi  │
└────────────────────────────────┘
          │               │
          └──── RRF  ─────┘   ← Reciprocal Rank Fusion
                  │
                  ▼
         RAG Pipeline (LLM)
                  │
                  ▼
       Answer + Inline Citations
```

**Why Hybrid?** Dense retrieval excels at semantic similarity; sparse BM25 excels at exact keyword matches. Combining both with RRF maximises recall — this is what production RAG systems at enterprises use.

---

## ⚡ Quick Start (Windows — Step by Step)

### Prerequisites
- Python 3.10 or 3.11 (download from python.org — check *Add to PATH*)
- 8 GB RAM minimum (12 GB recommended)

### Step 1 — Clone / unzip the project
```
cd C:\Users\YourName\Documents
```
Unzip the project folder here.

### Step 2 — Create a virtual environment
```cmd
cd rag_doc_qa
python -m venv venv
venv\Scripts\activate
```
You should see `(venv)` in your terminal.

### Step 3 — Install dependencies
```cmd
pip install -r requirements.txt
```
This takes 3–5 minutes on first run (downloads PyTorch + sentence-transformers).

### Step 4 — Configure API key
```cmd
copy .env.example .env
```
Open `.env` in Notepad and paste your **free Groq API key**:
- Go to https://console.groq.com → sign up → API Keys → Create Key
- Paste it next to `GROQ_API_KEY=`

### Step 5 — Run the app
```cmd
streamlit run app.py
```
Browser opens at `http://localhost:8501` 🎉

---

## 🔑 API Keys

| Provider | Cost | Speed | Quality | Get Key |
|----------|------|-------|---------|---------|
| **Groq** (recommended) | FREE | Very fast | Llama3 8B/70B | console.groq.com |
| OpenAI | Paid | Fast | GPT-3.5/4 | platform.openai.com |

**Embeddings run 100% locally** — no API key needed for indexing.

---

## 🧠 Technical Deep-Dive

### 1. Document Processing
- **PyMuPDF (fitz)**: Extracts text from PDFs page-by-page. Chosen over PyPDF2/pdfplumber because it handles complex layouts better and is a pure Python wheel (no VC++ build tools needed on Windows).
- **Sliding window chunking**: 500-word chunks with 100-word overlap ensures context isn't lost at chunk boundaries.

### 2. Dense Retrieval (FAISS)
- **Model**: `all-MiniLM-L6-v2` — 80MB, runs on CPU, 384-dim embeddings.
- **Index**: `IndexFlatIP` with L2-normalised vectors = cosine similarity.
- **faiss-cpu**: Prebuilt wheel, no CUDA, no compilation.

### 3. Sparse Retrieval (BM25)
- **rank-bm25**: Pure Python, zero C extensions.
- **BM25Okapi** with standard k1=1.5, b=0.75.
- Handles exact keyword matches that dense retrieval misses.

### 4. Reciprocal Rank Fusion
```python
RRF(d) = Σ  weight_i / (k + rank_i(d))
```
- k=60 (smoothing, standard from literature)
- Dense weight=0.7, Sparse weight=0.3
- No training needed — works out of the box.

### 5. Citation Tracking
Every LLM response includes numbered references `[1][2]...` mapped back to exact source file, page number, chunk ID, and relevance score.

---

## 📁 Project Structure

```
rag_doc_qa/
├── app.py                    # Streamlit UI
├── requirements.txt
├── .env.example
├── src/
│   ├── document_processor.py # PDF/text ingestion & chunking
│   ├── vector_store.py       # FAISS dense retrieval
│   ├── bm25_retriever.py     # BM25 sparse retrieval
│   ├── hybrid_retriever.py   # RRF fusion
│   ├── llm_client.py         # OpenAI / Groq abstraction
│   ├── rag_pipeline.py       # Orchestration + citation
│   └── session_state.py      # Streamlit state helpers
└── data/
    └── uploads/              # Temp upload directory
```

---

## 🚀 Features

- ✅ Upload multiple PDFs + TXT files simultaneously
- ✅ Hybrid BM25 + FAISS retrieval with RRF fusion
- ✅ Inline citations with source, page, and relevance score
- ✅ Configurable chunk size and overlap via UI sliders
- ✅ Free LLM via Groq (Llama3 8B/70B/Mixtral)
- ✅ Persistent chat history within session
- ✅ No Docker, no GPU required

---

## 🧪 Sample Questions to Try

After uploading a research paper or report:
- "What is the main contribution of this paper?"
- "Summarise the methodology section"
- "What datasets were used in the experiments?"
- "What are the limitations mentioned by the authors?"
- "Compare the results in Table 2 vs Table 3"

---

## 🐛 Troubleshooting (Windows)

| Error | Fix |
|-------|-----|
| `pip` not found | Re-install Python, check "Add to PATH" |
| `torch` install fails | Use `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `fitz` not found | Run `pip install pymupdf` |
| Streamlit port busy | Use `streamlit run app.py --server.port 8502` |
| LLM error 401 | API key incorrect — check `.env` file |
| RAM error on large PDF | Reduce chunk size slider to 300 |

---

## 📊 Recruiter-Facing Talking Points

1. **Hybrid RAG** — most production systems use BM25+dense, not just dense alone
2. **RRF** — parameter-free fusion; no training data needed
3. **Citation tracking** — addresses LLM hallucination, critical for enterprise
4. **Provider abstraction** — swappable LLM backend (OpenAI/Groq) via env vars
5. **Windows-native** — zero Docker, zero WSL, zero compilation

---

## 📝 License
MIT — free to use, modify, and showcase in your portfolio.
