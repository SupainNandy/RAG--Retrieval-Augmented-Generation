from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import warnings

# Suppress deprecation warnings from langchain
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv

load_dotenv()

persistent_directry = "db/chroma_db"

# Loading the embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the vector store from disk
db = Chroma (
    persist_directory=persistent_directry,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space":"cosine"}# which algorith we use mention here
)

#search relevent document
query="what are the main differenece between artificial intelligence and machine learning"
retriever=db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":3,#retirive only top most 3 chunk as per similarity
        "score_threshold":0.3 #Only return chunks with cosine similarity above 0.3
    } 
) 
relevent_docs = retriever.invoke(query)

print("=" * 60)
print(f"User Query: {query}")
print("=" * 60)
print(f"Total Relevant Documents Retrieved: {len(relevent_docs)}\n")

for i, doc in enumerate(relevent_docs, 1):
    source = doc.metadata.get("source", "Unknown")
    print(f"--- Document {i} [Source: {source}] ---")
    print(doc.page_content.strip())
    print("-" * 60 + "\n")