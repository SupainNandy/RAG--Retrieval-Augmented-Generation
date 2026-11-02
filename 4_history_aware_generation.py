# Step 1: Import the tools this chat needs.
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


# Step 2: Find this project folder, the saved document database, and the memory limit.
PROJECT_DIRECTORY = Path(__file__).resolve().parent
DATABASE_DIRECTORY = PROJECT_DIRECTORY / "db" / "chroma_db"
MAX_HISTORY_MESSAGES = 20


# Step 3: Start the chat and load the secret Groq API key from the .env file.
def main() -> None:
    load_dotenv(PROJECT_DIRECTORY / ".env")

    groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
    if not groq_api_key:
        raise ValueError("Neither GROQ_API_KEY nor GROK_API_KEY was found in the .env file.")

    os.environ["GROQ_API_KEY"] = groq_api_key

    # Step 4: Turn text into meaning-vectors and open the saved Chroma document database.
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma(
        persist_directory=str(DATABASE_DIRECTORY),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # Step 5: Prepare a librarian to find up to three useful document pieces.
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.3},
    )

    # Step 6: Connect to the Groq language model that will rewrite and answer questions.
    llm = ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0.2,
        api_key=groq_api_key,
    )

    # Step 7: Tell the model how to rewrite follow-up questions for document search.
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the latest user message as a concise, standalone search query "
                "using the conversation history to resolve references. Do not answer "
                "the message. For greetings or casual conversation, keep the user's "
                "message as-is. Return only the rewritten query.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    # Step 8: Tell the model how to answer using documents, chat history, or general knowledge.
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant. Use the retrieved context when it is "
                "relevant to the user's question. If it is empty or does not contain "
                "the answer, answer using your general knowledge. Respond naturally "
                "to greetings and follow-up questions. Be clear and honest when unsure.\n\n"
                "Retrieved context:\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    # Step 9: Connect each set of instructions to the model and start with empty chat memory.
    rewrite_chain = rewrite_prompt | llm
    answer_chain = answer_prompt | llm
    chat_history: list[BaseMessage] = []

    # Step 10: Keep asking for questions until the user types "exit" or "quit".
    print("Chat is ready. Type 'exit' or 'quit' to end the session.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Chat ended.")
            break
        if not question:
            print("Please enter a question.")
            continue

        # Step 11: If there is earlier chat, rewrite this question so the librarian understands it.
        standalone_query = question
        if chat_history:
            rewritten_message = rewrite_chain.invoke(
                {"chat_history": chat_history, "question": question}
            )
            standalone_query = str(rewritten_message.content).strip()
            if not standalone_query:
                raise ValueError("The LLM returned an empty search query.")

        # Step 12: Search for matching document pieces and join them into one context.
        documents = retriever.invoke(standalone_query)
        context = "\n\n---\n\n".join(document.page_content for document in documents)

        # Step 13: Ask the model to answer the original question using history and found context.
        response = answer_chain.invoke(
            {
                "chat_history": chat_history,
                "context": context,
                "question": question,
            }
        )
        answer = response.content
        print(f"\nAssistant: {answer}")

        # Step 14: Remember this question and answer, keeping only the newest 20 messages.
        chat_history.extend(
            [HumanMessage(content=question), AIMessage(content=answer)]
        )
        chat_history = chat_history[-MAX_HISTORY_MESSAGES:]


# Step 15: Run the chat only when this file is started directly.
if __name__ == "__main__":
    main()