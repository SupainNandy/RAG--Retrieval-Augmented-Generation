import os
import warnings

# Suppress deprecation warnings from langchain
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()


# Step 1: Loading the text documents
def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    # Check if the docs directory is present
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")

    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No .txt files found in {docs_path}. Please add your documents in the docs directory."
        )

    print(f"Successfully loaded {len(documents)} document(s).\n")

    for i, doc in enumerate(documents[:2]):
        print(f"--- Document {i+1} ---")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Content Length: {len(doc.page_content)} characters")
        print(f"Content Preview: {doc.page_content[:80]}...")
        print(f"Metadata: {doc.metadata}\n")

    return documents


# Step 2: Chunking the documents
def split_docs(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into small chunks with overlap"""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"---- Chunk {i+1} ----")
            print(f"Content Length: {len(chunk.page_content)} characters")
            print(f"Content Preview: {chunk.page_content[:80]}...")
            print(f"Metadata: {chunk.metadata}\n")

        if len(chunks) > 5:
            print(f"...and {len(chunks) - 5} more chunks\n")

    return chunks


# Step 3: Embedding and Creating Vector DB
def create_vector_store(chunks, persist_dir="db/chroma_db"):
    """Create and persist ChromaDB vector store using local free embeddings"""
    print("Creating vector store...")

    # Free, local embedding model (no API key or credits needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Creating ChromaDB vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    print(f"Vector store successfully created at {persist_dir}")
    print(f"Total chunks indexed in Vector DB: {vectorstore._collection.count()}\n")

    return vectorstore


def main():
    print("Main function started\n")

    # 1. Load documents
    docs = load_documents(docs_path="docs")

    # 2. Create chunks
    chunks = split_docs(docs)

    # 3. Create embeddings and store in Chroma Vector DB
    create_vector_store(chunks)

    print("Ingestion pipeline completed successfully!")


if __name__ == "__main__":
    main()
