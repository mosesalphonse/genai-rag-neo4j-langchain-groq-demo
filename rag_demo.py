# Install required packages
!pip install -q langchain langchain-community langchain-groq neo4j sentence-transformers langchain-text-splitters

# Import necessary modules
import os
from getpass import getpass
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Set up API keys and Neo4j credentials
# Replace with your actual values or use Colab secrets
os.environ["GROQ_API_KEY"] = getpass("Enter your Groq API key: ")
os.environ["NEO4J_URI"] = getpass("Enter your Neo4j URI (e.g., neo4j+s://<hash>.databases.neo4j.io or bolt://localhost:7687): ")
os.environ["NEO4J_USERNAME"] = getpass("Enter your Neo4j username: ")
os.environ["NEO4J_PASSWORD"] = getpass("Enter your Neo4j password: ")

# Initialize embeddings (using open-source HuggingFace model since Groq does not provide native embedding models)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Sample plain text (unstructured data) - replace with your own text
plain_text = """
Quarkus is a modern Java framework built from the ground up to support cloud-native, containerised and serverless applications. It emphasises fast startup times, low memory footprint, native-image compilation (via GraalVM) support, and both imperative and reactive programming models. On the other hand, Spring Boot has been a go-to framework for enterprise Java applications for many years: it offers a mature and broad ecosystem, vast community support, robust integration with databases, messaging systems and the Spring family of projects, and a familiar productive developer experience.

When comparing the two frameworks:

In terms of startup time and memory usage, Quarkus has a clear advantage thanks to build-time processing, tree-shaking, ahead-of-time (AOT) native compilation and container-first optimisation. Spring Boot, while highly capable, generally runs with higher memory overhead and slightly slower startup.

On the ecosystem and maturity front, Spring Boot stands out: its ecosystem is extensive (Spring Data, Spring Cloud, Spring Security etc.), which means fewer surprises when integrating with various third-party technologies. Quarkus’s ecosystem is growing rapidly, but still somewhat smaller in breadth compared to Spring.

For reactive programming, microservices and containerised/cloud environments (especially Kubernetes/serverless), Quarkus excels due to its design for modern architectures. Spring Boot also supports reactive programming (via Spring WebFlux) and microservices (via Spring Cloud), and if your team is already experienced with Spring you benefit from that familiarity.

With regards to developer experience, Spring Boot offers a smoother learning curve if you already know the Spring ecosystem, plus wide support in IDEs, tooling and communities. Quarkus offers novel features like live-coding, dev-mode reload and native binary option, which are attractive for iterative development and resource-constrained deployments.

In short: Choose Quarkus when your priorities are fast startup, low resource usage, container-first or serverless microservices and you’re building green-field services optimised for the cloud. Choose Spring Boot when your priority is a rich ecosystem, team familiarity, enterprise-grade integrations, and you already have investment in the Spring platform. Each has its strengths, and the decision really depends on your project’s specific constraints, team expertise and deployment environment
"""

# Split the text into chunks for better embedding and retrieval
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(plain_text)

# Connect to Neo4j and ingest the text chunks with embeddings
# This creates a vector store in Neo4j with a vector index for similarity search
vector_store = Neo4jVector.from_texts(
    chunks,
    embedding=embeddings,
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
    index_name="text_embeddings",  # Name of the vector index in Neo4j
    node_label="TextChunk",        # Label for the nodes
    embedding_node_property="embedding",  # Property to store the vector
    text_node_property="text"      # Property to store the original text
)

print("Text ingested into Neo4j with embeddings successfully!")

# Optional: To demonstrate usage with Groq AI models, set up a simple RAG (Retrieval-Augmented Generation) chain
# This retrieves similar chunks from Neo4j and uses a Groq model to generate a response

# Initialize Groq LLM (using an open-weight model like Llama3 hosted on Groq for fast inference)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# Define a prompt template for RAG
prompt_template = ChatPromptTemplate.from_template("""
Answer the question based on the following context:
{context}

Question: {question}
""")

# Function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Set up the RAG chain
rag_chain = (
    {
        "context": vector_store.as_retriever() | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

# Example query to test the setup
query = "What is quarkus?"
response = rag_chain.invoke(query)
print("\nRAG Response using Groq model:")
print(response)
