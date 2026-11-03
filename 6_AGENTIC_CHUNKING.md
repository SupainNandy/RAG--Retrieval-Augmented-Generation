# Agentic Chunking: A Simple Example

This guide explains the companion script [`6_agentic_chunking.py`](./6_agentic_chunking.py)
in simple English.

## What is agentic chunking?

Chunking breaks a long document into smaller pieces so a RAG system can search
them. In agentic chunking, a language model looks at the content and decides
which nearby paragraphs belong together.

For example, the model may keep a definition and its explanation in one chunk.
It returns paragraph numbers rather than rewriting the text. Python then uses
the original paragraphs to build the chunks.

## What the example does

1. Reads `docs/artificial_intelligence.txt`.
2. Splits it into paragraphs using blank lines.
3. Sends numbered paragraphs to a Groq chat model.
4. Asks the model to group related, adjacent paragraphs and return JSON.
5. Checks that every paragraph appears exactly once, in order, and that no
   chunk exceeds 2,000 characters.
6. Prints the resulting chunks.

These checks matter: an LLM can return malformed JSON, skip text, repeat text,
or make a chunk too large. The script reports those problems instead of
silently accepting a bad result.

## Setup and run

Install the project's dependencies:

```powershell
pip install -r requirements.txt
```

Set `GROQ_API_KEY` in your environment or add it to a `.env` file in the
project folder. Then run:

```powershell
python 6_agentic_chunking.py
```

The script uses the Groq model configured in the code, so it needs an internet
connection and a valid API key. The key should stay in your environment or
`.env` file; do not put it in the Python script.

## Important details

- The model chooses paragraph groupings; it does not generate or edit the
  document text.
- Paragraphs remain in their original order, and this example uses no overlap.
- The size limit is measured in **characters**, not LLM tokens.
- If one source paragraph is already longer than the limit, the script stops
  with an error so that paragraph can be split first.
- Agentic chunking can improve grouping for some documents, but it costs more
  time and API usage than a basic text splitter. Check the resulting chunks
  against real retrieval questions before using them in a production index.

This is a focused chunking example. It prints chunks but does not embed them or
save them to Chroma; those steps belong to the separate ingestion pipeline.
