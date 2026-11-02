from langchain_text_splitters import CharacterTextSplitter
# Agar puraana version use kar rahe ho toh: 
# from langchain.text_splitter import CharacterTextSplitter

# 1. Sample Text (Lamba text jisko todna hai)
text = """
Retrieval-Augmented Generation (RAG) ek aisi technique hai jo LLM ko external data provide karti hai.
Isse LLM ke hallucination reduce hote hain aur accurate answers milte hain.
CharacterTextSplitter text ko characters count ke basis par chunks me todta hai.
"""

# 2. Splitter initialize karo
splitter = CharacterTextSplitter(
    separator="\n",       # Kaha se todna hai (e.g. newline ya space)
    chunk_size=100,       # Ek chunk me maximum kitne characters honge
    chunk_overlap=20      # Agle chunk me pichle chunk ke kitne characters repeat honge taaki context miss na ho
)

# 3. Text ko split karo
chunks = splitter.split_text(text)

# 4. Result print karo
print(f"Total Chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} (Length: {len(chunk)}) ---")
    print(chunk.strip())
