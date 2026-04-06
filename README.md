# RAG-QnA-System
## 📌 Overview

This project is an end-to-end Retrieval-Augmented Generation (RAG) system designed to answer user queries using information from multiple data sources. It enables accurate, context-aware responses with source citations, reducing hallucinations and improving trust in generated answers. <br>

The system integrates document ingestion, semantic retrieval, and LLM-based response generation into a full-stack AI application.

---
## 🚀 Features

### 🔎 Multi-Source Data Ingestion
- Supports PDFs, websites, and CSV files
- Automated parsing and chunking for efficient retrieval

### 🧠 Advanced Retrieval Architecture
- Vector search using Qdrant
- Sentence-window retrieval for better context understanding
- Graph-based indexing to capture relationships between data
- Re-ranking to improve relevance of retrieved results

### 💬 Semantic Q&A with Citations
- Retrieves relevant context before generating answers
- Provides source references for transparency and verification

### 🌐 Interactive Web Application
- Built with Streamlit for real-time user interaction
- Simple UI for querying and viewing results

---

## 🏗️ System Architecture

### 1. Data Ingestion Layer
- Load and preprocess PDFs, web pages, and CSV data
- Chunk documents into smaller semantic units

### 2. Indexing Layer
- Store embeddings in Qdrant vector database
- Build graph-based and structured indices

### 3. Retrieval Layer
#### Multi-retriever pipeline:
- Vector similarity search
- Sentence-window retrieval
- Graph-based retrieval
- Apply re-ranking for optimal context selection

### 4. Generation Layer
- Pass retrieved context to LLM
- Generate grounded answers with citations

### 5. Frontend
- Streamlit app for user interaction

---

## 🛠️ Tech Stack
- **Framework:** LlamaIndex
- **Vector Database:** Qdrant
- **Frontend:** Streamlit
- **LLM Integration:** Gemini
- **Data Sources:** PDF, Web, CSV

---

## 📊 Key Highlights
- Combines multiple retrieval strategies for higher accuracy
- Reduces hallucination via grounded generation
- Demonstrates full-stack AI system design
- Scalable architecture for enterprise knowledge systems

---

## 🚀 How to Start the Application

Follow the steps below to run both the frontend and backend services locally.

### 1. Start the Docker engine for Qdrant vector database

### 2. Start the FastAPI backend

```bash
uv run uvicorn main:app
```

### 3. Run the Inngest development server

```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

### 4. Start the Frontend (Streamlit)

Run the following command to launch the Streamlit app:

```bash
uv run streamlit run .\streamlit_app.py
```

---

## 🔮 Future Improvements
- Add hybrid keyword + vector search
- Support real-time data updates
- Enhance UI/UX for better visualization of sources
- Integrate evaluation metrics for retrieval quality
