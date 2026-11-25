# 📚 AI Research Assistant

An AI-powered research assistant that allows users to upload multiple research papers in PDF format, ask questions in natural language, and receive context-aware answers based on the uploaded documents.

The system combines **Hybrid Search, Reciprocal Rank Fusion (RRF), Cross-Encoder Reranking, and Adaptive Top-K Retrieval** to improve the relevance of retrieved research content.

---

## 🚀 Features

- **Multi-PDF Upload:** Upload and process multiple research papers.
- **PDF Text Extraction:** Extract document content and preserve page-level metadata.
- **Vector Database:** Store document embeddings persistently using ChromaDB.
- **Semantic Search:** Retrieve contextually relevant passages using Sentence Transformers.
- **Keyword Search:** Apply BM25 to identify keyword-relevant passages.
- **Hybrid Retrieval:** Combine semantic and keyword search using Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking:** Improve retrieval relevance by reranking candidate passages.
- **Adaptive Top-K Retrieval:** Dynamically select relevant passages based on reranking scores.
- **Conversational Memory:** Maintain recent conversation history for follow-up questions.
- **Query Rewriting:** Convert follow-up questions into standalone queries for improved retrieval.
- **LLM Integration:** Generate answers using an LLM accessed through the OpenRouter API.
- **Streaming Responses:** Display generated answers progressively.
- **Research Library:** View processed PDFs and manage stored documents.

---

## 🏗️ System Architecture

```text
             User
              |
              v
       Streamlit Interface
              |
       Upload Research PDFs
              |
              v
        PDF Text Extraction
              |
              v
        Text Chunking
              |
              v
      Embedding Generation
              |
              v
          ChromaDB
              |
              v
        User Question
              |
              v
        Query Rewriting
              |
       +------+------+
       |             |
       v             v
  Semantic Search  BM25 Search
       |             |
       +------+------+
              |
              v
       Reciprocal Rank
        Fusion (RRF)
              |
              v
      Cross-Encoder Reranking
              |
              v
       Adaptive Top-K Selection
              |
              v
      Relevant Context Retrieval
              |
              v
        OpenRouter LLM
              |
              v
       Context-Aware Answer
              |
              v
          User Interface
```

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM API | OpenRouter |
| Vector Database | ChromaDB |
| Embedding Model | all-MiniLM-L6-v2 |
| Keyword Retrieval | BM25 |
| Reranking | Cross-Encoder |
| PDF Processing | PyMuPDF |
| Text Chunking | LangChain Text Splitters |
| Version Control | Git and GitHub |

---

## 📂 Project Structure

```text
AI-Research-Assistant/
│
├── app.py                 # Streamlit application and UI
├── data.py                # PDF text extraction
├── embedding.py           # Text chunking and embeddings
├── vector.py              # ChromaDB storage and operations
├── retrieval.py           # Hybrid retrieval and reranking
│
├── requirements.txt       # Project dependencies
├── .gitignore             # Ignored files and secrets
│
├── uploads/               # Uploaded PDF files (local)
├── chroma_db/             # Persistent vector database (local)
│
└── README.md
```

---

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-Research-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv new_env
```

Activate it on Windows:

```bash
new_env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Set your OpenRouter API key as an environment variable:

```powershell
$env:OPENROUTER_API_KEY="your_api_key"
```

For Streamlit Community Cloud, configure the key in the app's Secrets settings instead of hardcoding it.

### 5. Run the application

```bash
streamlit run app.py
```

---

## 🔍 Retrieval Pipeline

The application uses a multi-stage retrieval process:

1. **Query Rewriting:** Converts conversational follow-up questions into standalone search queries.
2. **Semantic Retrieval:** Finds passages based on embedding similarity.
3. **Keyword Retrieval:** Uses BM25 to identify keyword-based matches.
4. **RRF Fusion:** Combines results from both retrieval methods.
5. **Cross-Encoder Reranking:** Re-evaluates candidate passages for query relevance.
6. **Adaptive Top-K:** Selects relevant passages dynamically.
7. **Context Generation:** Supplies retrieved passages to the LLM to generate an answer.

---

## 🎯 Project Objective

The objective is to simplify research-paper analysis by allowing users to interact with multiple research documents through natural-language questions, reducing the need to manually search through lengthy academic PDFs.

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.
https://ai-research-assistant-gbj9vymqzktc39eeivusqt.streamlit.app/

## ⚠️ Important Note – LLM Token Limits

The application uses an external LLM through the OpenRouter API. Requests may occasionally fail if the combined prompt, conversation history, and retrieved document context exceed the selected model's token or context-window limit.

**If you encounter this issue:**
- Try asking shorter and more specific questions.
- Reduce the amount of PDF content being processed.
- Start a new chat to clear previous conversation history.
- Check the selected model's context-window limits and OpenRouter usage restrictions.

## 👨‍💻 Author

**Abdul Touheed Soudagar**

GitHub: [abdultouheed](https://github.com/abdultouheed)

---

*Built with Python, Retrieval-Augmented Generation, and modern information retrieval techniques.*
