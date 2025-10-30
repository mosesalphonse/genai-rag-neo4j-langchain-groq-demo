# HYBRID GRAPH RAG – File Upload + Knowledge Graph + Vector Search

**Ingest any document (PDF, DOCX, TXT, MD) → Build a knowledge graph → Ask questions in plain English.**

This repo shows a **complete RAG pipeline** using:
- **LangChain** (orchestration)
- **Neo4j** (vector + graph store)
- **HuggingFace** (embeddings)
- **Groq Llama 3.3 70B** (fast reasoning)

> **Guarantees**:  
> - Plain text **is stored** (not just embeddings)  
> - **Relationships are always created** (LLM + fallback)  
> - Works with **PDF, DOCX, TXT, MD**  
> - Answers in **clean, plain English**

---

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



