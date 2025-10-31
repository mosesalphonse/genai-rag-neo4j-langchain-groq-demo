# ================================================================================
# HYBRID GRAPH RAG – DATA INGESTION
# ================================================================================

# 0. Install & imports
!pip install -q langchain langchain-community langchain-groq neo4j \
                sentence-transformers langchain-text-splitters langchain-neo4j \
                PyPDF2 python-docx

import os, re, io
from getpass import getpass
from google.colab import files
import PyPDF2, docx
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import Neo4jGraph
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ------------------------------------------------------------------
# 1. Credentials & models
# ------------------------------------------------------------------
os.environ["GROQ_API_KEY"] = getpass("Groq API key: ")
os.environ["NEO4J_URI"] = getpass("Neo4j URI: ")
os.environ["NEO4J_USERNAME"] = getpass("Neo4j Username: ")
os.environ["NEO4J_PASSWORD"] = getpass("Neo4j Password: ")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Setup ready!\n")

# ------------------------------------------------------------------
# 2. FILE UPLOAD & TEXT EXTRACTION
# ------------------------------------------------------------------
def read_file(uploaded_dict) -> str:
    file_name = list(uploaded_dict.keys())[0]
    file_bytes = uploaded_dict[file_name]
    file_stream = io.BytesIO(file_bytes)

    if file_name.lower().endswith(('.txt', '.md')):
        return file_stream.read().decode('utf-8', errors='ignore')
    elif file_name.lower().endswith('.pdf'):
        reader = PyPDF2.PdfReader(file_stream)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif file_name.lower().endswith(('.docx', '.doc')):
        doc = docx.Document(file_stream)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        raise ValueError(f"Unsupported file: {file_name}")

print("Upload your document (PDF, DOCX, TXT, MD)...")
uploaded = files.upload()
plain_text = read_file(uploaded)
print(f"Loaded ~{len(plain_text.split())} words\n")

# ------------------------------------------------------------------
# 3. INGEST – CHUNKS + VECTORS + ENTITIES & RELATIONSHIPS
# ------------------------------------------------------------------
def ingest(text: str):
    g = Neo4jGraph(url=os.environ["NEO4J_URI"],
                   username=os.environ["NEO4J_USERNAME"],
                   password=os.environ["NEO4J_PASSWORD"])
    g.query("MATCH (n) DETACH DELETE n")
    print("Neo4j cleared.")

    # Chunking
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    print(f"{len(chunks)} chunk(s) created.")

    # Vector store
    vec = Neo4jVector.from_texts(
        chunks, embedding=emb,
        url=os.environ["NEO4J_URI"], username=os.environ["NEO4J_USERNAME"], password=os.environ["NEO4J_PASSWORD"],
        index_name="text_embeddings", node_label="TextChunk",
        embedding_node_property="embedding", text_node_property="text"
    )
    print("Vectors stored.")

    # Entity/relationship extraction
    extract_prompt = PromptTemplate.from_template(
        "Extract **entities** (programming languages, frameworks, concepts) and **relationships**.\n"
        "Return **only** valid JSON. Use this format:\n\n"
        "{{\n"
        ' "entities": ["Java", "Spring", "Quarkus", "Functional Programming"],\n'
        ' "relationships": [\n'
        ' {{"source": "Java", "relation": "SUPPORTS", "target": "Functional Programming"}},\n'
        ' {{"source": "Java", "relation": "USED_WITH", "target": "Spring"}}\n'
        " ]\n"
        "}}\n\n"
        "Text: {text}\n\n"
        "JSON:"
    )
    extract_chain = extract_prompt | llm | JsonOutputParser()

    entities, relationships = set(), []
    print("Extracting KG (entities + relationships)...")
    for i, c in enumerate(chunks):
        try:
            data = extract_chain.invoke({"text": c})
            ents = [e.strip() for e in data.get("entities", []) if e.strip()]
            rels = data.get("relationships", [])
            entities.update(ents)
            relationships.extend(rels)
            print(f" chunk {i}: {len(ents)} ent, {len(rels)} rel")
        except Exception as e:
            print(f" chunk {i} error: {e}")

    # Fallback if LLM gave no relationships
    if not relationships:
        print("No relationships from LLM → using rule-based fallback...")
        for chunk in chunks:
            chunk_lower = chunk.lower()
            if "java" in chunk_lower and "spring" in chunk_lower:
                relationships.append({"source": "Java", "relation": "USED_WITH", "target": "Spring"})
            if "java" in chunk_lower and "quarkus" in chunk_lower:
                relationships.append({"source": "Java", "relation": "USED_WITH", "target": "Quarkus"})
            if "java" in chunk_lower and ("functional" in chunk_lower or "object oriented" in chunk_lower):
                target = "Functional Programming" if "functional" in chunk_lower else "Object Oriented Programming"
                relationships.append({"source": "Java", "relation": "SUPPORTS", "target": target})

    # Store entities
    if entities:
        g.query("UNWIND $list AS e MERGE (n:Entity {name: e.name})",
                {"list": [{"name": e} for e in entities]})
    print(f"{len(entities)} Entity nodes stored.")

    # Store relationships
    def safe_rel_type(s): return re.sub(r'[^A-Z0-9_]', '_', s.strip().upper())
    stored = 0
    for r in relationships:
        src = r.get("source", "").strip()
        tgt = r.get("target", "").strip()
        typ = safe_rel_type(r.get("relation", "RELATED_TO"))
        if src and tgt and typ:
            cypher = f"""
            MATCH (a:Entity {{name: $src}})
            MATCH (b:Entity {{name: $tgt}})
            MERGE (a)-[r:`{typ}`]->(b)
            """
            g.query(cypher, {"src": src, "tgt": tgt})
            print(f" REL: {src} -[{typ}]→ {tgt}")
            stored += 1
    print(f"{stored} relationship(s) stored.\n")
    return vec, g

vector_store, graph = ingest(plain_text)

# ------------------------------------------------------------------
# 4. VERIFICATION
# ------------------------------------------------------------------
print("=== NEO4J CONTENT ===")
print("TextChunk :", graph.query("MATCH (c:TextChunk) RETURN count(c)")[0]["count(c)"])
print("Entity :", graph.query("MATCH (e:Entity) RETURN count(e)")[0]["count(e)"])
print("Rels :", graph.query("MATCH ()-[r]->() RETURN count(r)")[0]["count(r)"])
print("\nEntities:", [r["e.name"] for r in graph.query("MATCH (e:Entity) RETURN e.name ORDER BY e.name")])
print("\nRelationships:")
for r in graph.query("MATCH (a)-[rel]->(b) RETURN a.name, type(rel), b.name"):
    print(f" {r['a.name']} -[{r['type(rel)']}]→ {r['b.name']}")
print("\nIngestion complete! You can now run the **inference** notebook.")
