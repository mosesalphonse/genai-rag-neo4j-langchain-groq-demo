# genai-rag-neo4j-langchain-groq-demo
POC for GenAI with RAG to ingest unstructured data into Vector embedding with semantic search

This repository demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline built using:
- **LangChain** for orchestration  
- **Neo4j** as a knowledge graph & vector store  
- **HuggingFace** embeddings for semantic search  
- **Groq’s ultra-fast LLMs** (e.g., Llama 3.3 70B) for reasoning  

It is a simple, end-to-end example designed for anyone learning **Enterprise GenAI with Knowledge Graphs + LLMs**.

---

## 🧩 Architecture Overview

1. **Data Preparation**
   - Takes any unstructured text (e.g., FAQ, PDF text, Word documents).
   - Splits into chunks using `CharacterTextSplitter`.

2. **Knowledge Graph + Vector Store**
   - Stores text chunks and embeddings inside **Neo4j Aura** (or local Neo4j).
   - Creates a vector index for semantic similarity search.

3. **RAG Pipeline**
   - Retrieves the most relevant chunks from Neo4j.
   - Uses **Groq Llama-3.3-70B** model to generate context-aware responses.

---

## 🧱 Prerequisites

### 1. Environment
You can run this on:
- **Google Colab (recommended)** — easiest way to try.
- **Local environment** with Python 3.10+ and pip.

### 2. Accounts Required
- 🧠 **[Groq Cloud](https://console.groq.com/)** — get a free API key.  
- 🕸 **[Neo4j Aura](https://console.neo4j.io/)** — free cloud graph database.

---

## 🧱 Prerequisites

### 1. Environment
You can run this project on:
- 🟢 **Google Colab (Recommended)** — easiest way to try.  
- 💻 **Local Python 3.10+ environment** with pip.

### 2. Accounts Required
Before running, create:
- **Groq Cloud account:** [https://console.groq.com/](https://console.groq.com/) → get your free API key.  
- **Neo4j AuraDB account:** [https://console.neo4j.io/](https://console.neo4j.io/) → create a free cloud graph database.

---

## ☁️ Run This Project on Google Colab

### 🧑‍💻 1. Open Google Colab
Go to [https://colab.research.google.com/](https://colab.research.google.com/).

### 📦 2. Clone the Repository
In a new Colab cell, run:
```python
!git clone https://github.com/<your-username>/genai-rag-neo4j-langchain-groq-demo.git
%cd genai-rag-neo4j-langchain-groq-demo


### Enter Required Credentials

When prompted, provide:

GROQ_API_KEY → <<Groq_API_Key>>

NEO4J_URI → e.g. neo4j+s://<hash>.databases.neo4j.io

NEO4J_USERNAME → usually neo4j

NEO4J_PASSWORD → from your Neo4j Aura dashboar


### Expected Output

Text ingested into Neo4j with embeddings successfully!
