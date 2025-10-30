# ================================================================================
#  HYBRID GRAPH RAG – DATA INGESTION (FILE UPLOAD) + KNOWLEDGE GRAPH + INFERENCES
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
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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
# 3. INGEST – CHUNKS + VECTORS + **GUARANTEED** ENTITIES & RELS
# ------------------------------------------------------------------
def ingest(text: str):
    g = Neo4jGraph(url=os.environ["NEO4J_URI"],
                   username=os.environ["NEO4J_USERNAME"],
                   password=os.environ["NEO4J_PASSWORD"])

    g.query("MATCH (n) DETACH DELETE n")
    print("Neo4j cleared.")

    # LARGER CHUNKS → more context for relationships
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    print(f"{len(chunks)} chunk(s) created.")

    vec = Neo4jVector.from_texts(
        chunks, embedding=emb,
        url=os.environ["NEO4J_URI"], username=os.environ["NEO4J_USERNAME"], password=os.environ["NEO4J_PASSWORD"],
        index_name="text_embeddings", node_label="TextChunk",
        embedding_node_property="embedding", text_node_property="text"
    )
    print("Vectors stored.")

    # BETTER PROMPT – with real examples
    extract_prompt = PromptTemplate.from_template(
        "Extract **entities** (programming languages, frameworks, concepts) and **relationships**.\n"
        "Return **only** valid JSON. Use this format:\n\n"
        "{{\n"
        '  "entities": ["Java", "Spring", "Quarkus", "Functional Programming"],\n'
        '  "relationships": [\n'
        '    {{"source": "Java", "relation": "SUPPORTS", "target": "Functional Programming"}},\n'
        '    {{"source": "Java", "relation": "USED_WITH", "target": "Spring"}}\n'
        "  ]\n"
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
            print(f"  chunk {i}: {len(ents)} ent, {len(rels)} rel")
        except Exception as e:
            print(f"  chunk {i} error: {e}")

    # FALLBACK: If LLM gave 0 rels, extract simple ones
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

    # STORE ENTITIES
    if entities:
        g.query("UNWIND $list AS e MERGE (n:Entity {name: e.name})",
                {"list": [{"name": e} for e in entities]})
    print(f"{len(entities)} Entity nodes stored.")

    # STORE RELATIONSHIPS (with debug print)
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
            print(f"  REL: {src} -[{typ}]→ {tgt}")
            stored += 1
    print(f"{stored} relationship(s) stored.\n")
    return vec, g

vector_store, graph = ingest(plain_text)

# ------------------------------------------------------------------
# 4. VERIFICATION – PROOF RELS EXIST
# ------------------------------------------------------------------
print("=== NEO4J CONTENT ===")
print("TextChunk :", graph.query("MATCH (c:TextChunk) RETURN count(c)")[0]["count(c)"])
print("Entity    :", graph.query("MATCH (e:Entity) RETURN count(e)")[0]["count(e)"])
print("Rels      :", graph.query("MATCH ()-[r]->() RETURN count(r)")[0]["count(r)"])
print("\nEntities:", [r["e.name"] for r in graph.query("MATCH (e:Entity) RETURN e.name ORDER BY e.name")])
print("\nRelationships:")
for r in graph.query("MATCH (a)-[rel]->(b) RETURN a.name, type(rel), b.name"):
    print(f"  {r['a.name']} -[{r['type(rel)']}]→ {r['b.name']}")
print("\n")

# ------------------------------------------------------------------
# 5. SAFE GRAPH RETRIEVAL
# ------------------------------------------------------------------
cypher_prompt = PromptTemplate.from_template(
    "Write ONE MATCH query using label `Entity`. Return ONLY Cypher.\n"
    "Question: {question}\nEntity hint: {entity}\nCypher:"
)

def extract_entity(q: str) -> str:
    words = re.findall(r'\b[A-Z][a-z]+\b', q)
    stop = {"the","a","an","and","or","but","in","on","at","to","for","of","with","is","was","are","were"}
    for w in words:
        if w.lower() not in stop:
            return w
    return q.split()[0] if q else ""

def graph_context(q: str) -> str:
    try:
        entity = extract_entity(q)
        cy = cypher_prompt | llm | StrOutputParser()
        query = cy.invoke({"question": q, "entity": entity}).strip()
        if not query.upper().startswith("MATCH"):
            query = f"""
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower('{entity}')
            OPTIONAL MATCH (e)-[r]-(other)
            RETURN e.name AS name,
                   collect({{rel:type(r), target:other.name}}) AS rels
            LIMIT 3
            """
        if any(k in query.upper() for k in ("CREATE","DROP","DELETE","SET","DETACH")):
            return ""
        rows = graph.query(query)
        if not rows: return ""
        lines = []
        for row in rows:
            name = row.get("name") or "?"
            rels = row.get("rels") or []
            lines.append(f"{name}")
            for r in rels[:3]:
                lines.append(f"  -[{r.get('rel','?')}]→ {r.get('target','?')}")
        return "\n".join(lines)
    except:
        return ""

# ------------------------------------------------------------------
# 6. HYBRID RAG CHAIN – PLAIN ENGLISH
# ------------------------------------------------------------------
def hybrid_retrieve(q):
    vec = vector_store.similarity_search(q, k=2)
    vctx = "\n\n".join(d.page_content for d in vec)
    gctx = graph_context(q)
    return vctx, gctx

final_prompt = ChatPromptTemplate.from_template(
    "Answer in **plain English** using this info:\n\n"
    "Document:\n{vector_context}\n\n"
    "Knowledge graph:\n{graph_context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

chain = (
    {"vector_context": lambda q: hybrid_retrieve(q)[0],
     "graph_context": lambda q: hybrid_retrieve(q)[1],
     "question": RunnablePassthrough()}
    | final_prompt | llm | StrOutputParser()
)

# ------------------------------------------------------------------
# 7. INTERACTIVE(INFERENCES) LOOP
# ------------------------------------------------------------------
print("READY! Ask a question (type 'quit' to exit)\n")
while True:
    q = input("Question: ").strip()
    if q.lower() in {"quit", "exit", "q"}:
        print("Goodbye!")
        break
    if not q: continue
    ans = chain.invoke(q)
    print(f"\nAnswer: {ans}\n")
    print("-" * 60)
