import os
import warnings

# Suppress deprecation and runtime warnings for a clean console output
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load environment variables
load_dotenv()

# Support both GROK_API_KEY and GROQ_API_KEY from .env
groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
if not groq_api_key:
    raise ValueError("Neither GROQ_API_KEY nor GROK_API_KEY found in .env file!")

# Set in os.environ for langchain-groq compatibility
os.environ["GROQ_API_KEY"] = groq_api_key

persistent_directory = "db/chroma_db"

# 2. Load the HuggingFace embedding model (same model used during ingestion)
print("Loading Embedding Model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Load the Chroma vector database from disk
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

# 4. Define the user query
query = "what are the main differenece between artificial intelligence and machine learning"

# 5. Create the retriever with cosine similarity score threshold
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,               # Retrieve top 3 most relevant chunks
        "score_threshold": 0.3  # Only return chunks with similarity >= 0.3
    }
)

# 6. Retrieve relevant documents from ChromaDB
print("Retrieving matching documents from vector database...")
relevant_docs = retriever.invoke(query)

print("\n" + "=" * 60)
print(f"USER QUERY: {query}")
print("=" * 60)
print(f"Total Relevant Documents Retrieved: {len(relevant_docs)}\n")

if not relevant_docs:
    print("No relevant documents found meeting the score threshold.")
    exit()

# Display the retrieved chunks and their sources
for i, doc in enumerate(relevant_docs, 1):
    source = doc.metadata.get("source", "Unknown")
    print(f"--- [Document {i}] Source: {source} ---")
    print(doc.page_content.strip())
    print("-" * 60 + "\n")

# 7. Prepare the context text by combining all retrieved chunks
context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

# 8. Create the prompt template instructing the LLM to use the context
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert AI assistant. Answer the user's question using ONLY the "
        "provided context below. If the context does not contain the answer, say that "
        "the provided documents do not have that information. Be clear, accurate, and concise.\n\n"
        "Context:\n{context}"
    ),
    ("human", "{question}")
])

# 9. Initialize the Groq LLM (Grok/Groq high-speed inference)
print("Sending context and query to Groq LLM...")
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0.2,
    api_key=groq_api_key
)

# 10. Create the RAG chain and generate the answer
rag_chain = prompt | llm
response = rag_chain.invoke({
    "context": context,
    "question": query
})

# 11. Display the final generated response
print("\n" + "=" * 60)
print("FINAL LLM GENERATED ANSWER")
print("=" * 60)
print(response.content)
print("=" * 60 + "\n")