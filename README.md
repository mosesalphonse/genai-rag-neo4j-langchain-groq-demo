# Hybrid Graph RAG – File Upload + Knowledge Graph + Vector Search

**Upload any document (PDF, DOCX, TXT, MD) → Build a knowledge graph → Ask questions in plain English.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/genai-rag-neo4j-langchain-groq-demo/blob/main/rag_demo.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-v0.1%2B-orange)](https://python.langchain.com)
[![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-green)](https://neo4j.com/cloud/aura/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-yellow)](https://console.groq.com)

> **Guarantees**:
> - Plain text **is stored** (not just embeddings)
> - **Relationships are always created** (LLM + fallback)
> - Works with **PDF, DOCX, TXT, MD**
> - Answers in **clean, plain English**

---

## Features

- File upload support: `.pdf`, `.docx`, `.txt`, `.md`
- Stores **text + embeddings** in Neo4j (`TextChunk` nodes)
- Extracts **entities & relationships** (LLM + rule-based fallback)
- Semantic search via vector index
- Fast reasoning with **Groq Llama 3.3 70B**
- Runs in **Google Colab** or **locally**

---

## Prerequisites

| Requirement         | How to Get                                      |
|---------------------|-------------------------------------------------|
| Groq API Key        | [https://console.groq.com](https://console.groq.com) → Free tier available |
| Neo4j AuraDB        | [https://console.neo4j.io](https://console.neo4j.io) → Free cloud instance |
| Python 3.10+        | Local install or use Google Colab               |

**Note:** No GPU needed — runs on CPU in Colab.

## Run in Google Colab (Recommended)

1. **Open in Colab**  
   Click the badge:  
   ![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)

2. **Run All Cells**  
   - Run the setup cell → installs required packages  
   - Enter credentials when prompted:  
     - Groq API key  
     - Neo4j URI: `neo4j+s://<your-id>.databases.neo4j.io`  
     - Neo4j Username: `neo4j`  
     - Neo4j Password  
   - Upload your file → Click “Choose Files”  
   - Wait until you see messages: “Vectors stored”, “X relationships stored”  
   - Ask questions!

```
git clone https://github.com/mosesalphonse/genai-rag-neo4j-langchain-groq-demo.git
cd genai-rag-neo4j-langchain-groq-demo
python rag_demo.py
```

## Architecture

```mermaid
graph TD
    A[Upload File] --> B[Extract Text]
    B --> C[Chunk Text]
    C --> D[Store in Neo4j]
    D --> E[Vector Index]
    D --> F[Entity + Relationship Extraction]
    E --> G[Semantic Search]
    F --> H[Knowledge Graph]
    G & H --> I[Hybrid RAG]
    I --> J[Plain English Answer]

