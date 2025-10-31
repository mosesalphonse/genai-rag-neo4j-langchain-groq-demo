# ================================================================================
# HYBRID GRAPH RAG – INFERENCE ONLY
# ================================================================================

# 0. Install & imports
!pip install -q langchain langchain-community langchain-groq neo4j \
                sentence-transformers langchain-neo4j

import os, re
from getpass import getpass
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ------------------------------------------------------------------
# 1. Credentials & models (reuse the same Neo4j DB created by ingestion)
# ------------------------------------------------------------------
os.environ["GROQ_API_KEY"] = getpass("Groq API key: ")
os.environ["NEO4J_URI"] = getpass("Neo4j URI: ")
os.environ["NEO4J_USERNAME"] = getpass("Neo4j Username: ")
os.environ["NEO4J_PASSWORD"] = getpass("Neo4j Password: ")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to existing vector index & graph
vector_store = Neo4jVector(
    embedding=emb,
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
    index_name="text_embeddings",
    embedding_node_property="embedding",
    text_node_property="text"
)
graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"]
)
print("Connected to existing knowledge graph & vector store.\n")

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
                lines.append(f" -[{r.get('rel','?')}]→ {r.get('target','?')}")
        return "\n".join(lines)
    except:
        return ""

# ------------------------------------------------------------------
# 6. HYBRID RAG CHAIN
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
# 7. INTERACTIVE LOOP
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
