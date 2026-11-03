# Recursive, Structure-Aware Chunking

This guide explains [`8_recursive_structure_aware_chunking.py`](./8_recursive_structure_aware_chunking.py)
step by step. It uses LangChain's `RecursiveCharacterTextSplitter` to split
Markdown while trying to keep headings, paragraphs, and sentences together.

## 1. What are we trying to do?

A RAG system searches small pieces of documents, called **chunks**. If a chunk
starts or ends in the wrong place, it may lose useful context. For example, a
paragraph that says “They are rocky worlds” is unclear if the previous sentence
about the inner planets is in another chunk.

A recursive splitter tries a prioritized list of boundaries:

1. Larger structure, such as headings.
2. Paragraph breaks.
3. Line breaks.
4. Spaces between words.
5. Individual characters, as a last resort.

It uses a smaller boundary only when the current piece is still too large. It
does not understand meaning like an LLM; it follows the separators you provide.

## 2. Install and run

The project already lists `langchain-text-splitters` in `requirements.txt`.
From the project directory, run:

```powershell
pip install -r requirements.txt
python 8_recursive_structure_aware_chunking.py
```

The example is self-contained and does not need an API key or a network call.
It prints each chunk's text, character count, source, and starting character
position.

## 3. The sample Markdown

The Python example contains a small Markdown document about the solar system.
It has:

- A top-level `# Solar System` heading.
- `##` headings for the Sun and planets.
- `###` headings for topics within the planets section.
- Paragraphs that explain each topic.

This makes it easy to observe how heading and paragraph boundaries affect
splitting. For your own data, replace `markdown_text` with the content of your
document, or load files with the project's document loader and call
`split_documents()` on the resulting documents.

## 4. How the separator list works

```python
SEPARATORS = [
    "\n# ",
    "\n## ",
    "\n### ",
    "\n\n",
    "\n",
    " ",
    "",
]
```

The splitter tries separators from the first item to the last. The list says:

- First, try Markdown heading boundaries.
- If a section still needs to be divided, try blank lines between paragraphs.
- If needed, try individual lines and then spaces.
- As a final fallback, split at character boundaries.

The final empty string is the character-level fallback. The `#` in the
separator is a literal Markdown heading character; it is not a regular
expression here.

## 5. The splitter settings

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=420,
    chunk_overlap=60,
    separators=SEPARATORS,
    keep_separator="start",
    add_start_index=True,
)
```

- **`chunk_size=420`**: target maximum chunk size, measured in characters with
  the default length function. It is not 420 tokens.
- **`chunk_overlap=60`**: neighboring chunks can repeat up to 60 characters to
  retain some context across a split. Overlap increases stored text, so tune it
  using retrieval tests.
- **`separators=SEPARATORS`**: controls which boundaries are preferred and in
  what order.
- **`keep_separator="start"`**: keeps a matched boundary with the text after
  it. This helps the next chunk retain its heading marker.
- **`add_start_index=True`**: asks the splitter to include each chunk's start
  position in its metadata.

The size is a target, not a way to promise that every arbitrary input is a
complete semantic section. A long section may still need splitting to fit the
configured size. A very long word or unbroken string is also an edge case to
check when you use a custom length function or separators.

## 6. Creating chunks and keeping metadata

```python
documents = splitter.create_documents(
    [markdown_text],
    metadatas=[{"source": "solar_system_demo.md"}],
)
```

`create_documents()` returns LangChain `Document` objects. Each object has:

- `page_content`: the chunk's text.
- `metadata`: extra information about the chunk, such as its source and
  starting index.

When splitting documents that already have metadata, use
`splitter.split_documents(documents)` so the source metadata is carried into
the resulting chunks.

## 7. Why is it “structure-aware”?

The splitter is **structure-aware because we gave it structure-specific
separators**. It prefers to split at headings and paragraphs instead of
arbitrarily cutting after a fixed number of characters.

It does **not** automatically parse every Markdown feature or understand the
meaning of headings. For tables, code blocks, HTML, PDFs, or complex Markdown,
inspect the output and consider a format-specific loader or parser.

## 8. Why use this as a practical starting point?

- It is fast and does not need an LLM API call.
- It usually preserves natural text boundaries better than one fixed
  character separator.
- You can use it for plain text and customize the separator list for Markdown.
- You can pass its chunks to an embedding model and vector database afterward.

It is a good starting point, not a guarantee of perfect retrieval. Test with
real questions, inspect the retrieved chunks, and adjust size, overlap, and
separators based on the results.

## Hinglish summary

Simple baat: splitter pehle heading par todne ki koshish karta hai. Agar
heading ke neeche ka section bahut bada ho, toh paragraph, line, word, aur
aakhir mein character level tak jaata hai. `chunk_size` characters mein hai;
`chunk_overlap` pichhle chunk ka thoda text agle chunk mein repeat karta hai.
Yeh LLM-based semantic splitting nahi hai—boundaries hum separator list mein
define karte hain.
