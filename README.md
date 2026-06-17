# 🤖 Rafif's AI CV Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about [Rafif Shafwan](https://linkedin.com/in/rafif-shafwan)'s CV, skills, experience, and projects. Built with LangChain, FAISS, Google Gemini, and Streamlit.

---

## ✨ Features

- **Conversational RAG** — Ask follow-up questions naturally; the bot remembers context
- **Multi-model fallback** — Primary model (`gemini-3-flash-preview`) automatically falls back to `gemini-2.5-flash` on rate limits
- **MMR retrieval** — Maximum Marginal Relevance ensures diverse, non-repetitive context
- **RAGAS evaluation** — Quantified performance on faithfulness, answer relevancy, context precision, and context recall
- **Streamlit UI** — Clean chat interface with sidebar profile, download CV button, and mobile-friendly layout

---

## 🚀 Quick Start

### 1. Clone & enter

```bash
git clone https://github.com/rafifshaf-fun/rag-cv-chatbot.git
cd rag-cv-chatbot
```

### 2. Set up environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

> Get a free API key from [Google AI Studio](https://aistudio.google.com/).

### 4. (Optional) Seed the FAISS index

The index is pre-built and included in the repo. If you edit the data files, rebuild it:

```bash
python -c "from src.vectorstore import build_vectorstore; build_vectorstore(force=True)"
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## 🏗️ Project Structure

```
rag-cv-chatbot/
├── app.py                 ← Streamlit UI entry point
├── config.py              ← All tunable constants (models, paths, chunk size, etc.)
├── rag_pipeline.py        ← Backward-compatible re-exports
├── evaluate.py            ← RAGAS evaluation script
├── requirements.txt
├── .env                   ← API keys (not committed)
├── data/
│   ├── my_cv.md           ← Curriculum vitae (structured)
│   └── about_me.md        ← Extended personal profile
├── faiss_index/           ← Cached vector store
├── assets/
│   └── profpic.jpeg       ← Profile picture for sidebar
└── src/                   ← Modular pipeline
    ├── embeddings.py      ← FastEmbed (ONNX-based, no API key)
    ├── ingestion.py       ← Load & chunk documents
    ├── vectorstore.py     ← FAISS build / load
    ├── retrieval.py       ← MMR retriever setup
    ├── llm.py             ← Gemini models + fallback + session history
    ├── prompts.py         ← Prompt templates
    └── chain.py           ← Wires everything → get_chain()
```

---

## 🧠 How It Works

```
User Question
      │
      ▼
┌─────────────────┐
│ Contextualize    │ ← Rephrases follow-ups with chat history
│ (if needed)      │
└────────┬────────┘
         ▼
┌─────────────────┐
│  MMR Retriever   │ ← Fetches 8 diverse chunks from FAISS
└────────┬────────┘
         ▼
┌─────────────────┐
│ QA Chain         │ ← Gemini answers with context
│ (w/ fallback)    │
└────────┬────────┘
         ▼
      Answer
```

- **Embeddings:** `BAAI/bge-small-en-v1.5` via FastEmbed (ONNX, runs locally)
- **Vector store:** FAISS with MMR search (`k=8`, `fetch_k=25`, `lambda_mult=0.6`)
- **Chunking:** Recursive split at 500 chars with 100 overlap
- **Memory:** LangChain `RunnableWithMessageHistory` (per-session in-memory)

---

## 📊 Evaluation

Run the RAGAS evaluation suite:

```bash
python evaluate.py
```

This tests the pipeline on 8 Q&A pairs and reports:
- **Faithfulness** — Is the answer grounded in the retrieved context?
- **Answer Relevancy** — Does the answer address the question?
- **Context Precision** — Are the retrieved chunks relevant?
- **Context Recall** — Does the context contain all necessary information?

Results are saved to `evaluation_results.csv`.

---

## 📝 Customizing the Data

Edit the markdown files in `data/`:

- **`my_cv.md`** — Structured CV (experience, projects, skills, education)
- **`about_me.md`** — Extended personal profile (personality, hobbies, achievements, languages)

After editing, rebuild the index:

```bash
python -c "from src.vectorstore import build_vectorstore; build_vectorstore(force=True)"
```

To change the bot's behaviour or persona, edit `src/prompts.py`.

---

## 🛠️ Tech Stack

| Area | Tools |
|---|---|
| **UI** | [Streamlit](https://streamlit.io/) |
| **LLM** | [Google Gemini](https://ai.google.dev/) (`gemini-3-flash-preview` → `gemini-2.5-flash`) |
| **Embeddings** | [FastEmbed](https://github.com/qdrant/fastembed) (`BAAI/bge-small-en-v1.5`) |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) |
| **Orchestration** | [LangChain](https://www.langchain.com/) (LCEL, `RunnableWithMessageHistory`) |
| **Evaluation** | [RAGAS](https://docs.ragas.io/) |
| **Deployment** | [Streamlit Community Cloud](https://streamlit.io/cloud) |

---

## 🔮 Roadmap Ideas

- [ ] Add source citation (show which chunk the answer came from)
- [ ] PDF ingestion (upload and query your own documents)
- [ ] Switchable LLM backends (Groq, OpenAI, local Ollama)
- [ ] Docker Compose for self-hosted deployment
- [ ] Persistent session history (SQLite instead of in-memory)

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

## 👤 About

Built by **Rafif Shafwan** — [GitHub](https://github.com/rafifshaf-fun) · [LinkedIn](https://linkedin.com/in/rafif-shafwan)

*This is one of several portfolio projects demonstrating applied ML engineering. The techniques shown here are the same ones used in production systems delivered to enterprise clients — just simplified and open for review.*
