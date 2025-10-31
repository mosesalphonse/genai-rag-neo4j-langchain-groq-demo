# Hybrid Graph RAG – File Upload + Knowledge Graph + Vector Search

**Upload any document (PDF, DOCX, TXT, MD) → Build a knowledge graph → Ask questions in plain English.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/genai-rag-neo4j-langchain-groq-demo/blob/main/rag_demo.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-v0.1%2B-orange)](https://python.langchain.com)
[![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-green)](https://neo4j.com/cloud/aura/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-yellow)](https://console.groq.com)

> **Guarantees**  
> - Plain text **is stored** (not just embeddings)  
> - **Relationships always created** (LLM + fallback)  
> - Works with **PDF, DOCX, TXT, MD**  
> - Answers in **clean, plain English**

---

## Features
- File upload: `.pdf`, `.docx`, `.txt`, `.md`
- Stores **text + embeddings** in Neo4j
- Extracts **entities & relationships** (LLM + fallback)
- Semantic search via vector index
- Fast answers with **Groq Llama 3.3 70B**
- Runs in **Colab** or **locally**

---

## Prerequisites

| Requirement       | How to Get |
|-------------------|------------|
| **Groq API Key**  | [console.groq.com](https://console.groq.com) |
| **Neo4j AuraDB**  | [console.neo4j.io](https://console.neo4j.io) |
| **Python 3.10+**  | Local or Colab |

> No GPU needed.

---

## Run in Google Colab

1. Click:  
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/genai-rag-neo4j-langchain-groq-demo/blob/main/rag_demo.ipynb)

2. Run all cells → enter:
   - `Groq API key`
   - `Neo4j URI` (`neo4j+s://<id>.databases.neo4j.io`)
   - `Username`: `neo4j`
   - `Password`
3. Upload file → **Choose Files**
4. Wait for: `Vectors stored`, `X relationships stored`
5. Start asking questions!

> **Tip:** The graph persists in Neo4j — close and reopen later.

---

## Architecture

### Data Ingestion Flow
```mermaid
graph TD
    A["Upload File\nPDF, DOCX, TXT, MD"] --> B[Extract Text]
    B --> C["Chunk Text\n500 chars, 100 overlap"]
    C --> D[Neo4j DB]
    D --> E["Vector Index\nTextChunk + Embedding"]
    C --> F["LLM Extract\nEntities & Relationships"]
    F --> G[Rule-based Fallback]
    F --> H["Store Entities\n:Entity {name}"]
    F --> I["Store Relationships\n-[:REL]->"]
    E & H & I --> J[Ready for Inference]
