from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role

SYSTEM_PROMPT = """
You are a RAG-powered assistant that helps users with questions about microwave usage.

## User message structure:
- RAG CONTEXT: Retrieved relevant document chunks.
- USER QUESTION: The user's actual question.

## Instructions:
- Use only the information from the RAG CONTEXT and conversation history to answer the USER QUESTION.
- If the RAG CONTEXT does not contain relevant information, say you cannot answer.
- Do NOT answer questions unrelated to microwave usage or not covered by the context/history.
"""

USER_PROMPT = """
## RAG CONTEXT:
{context}

## USER QUESTION:
{query}
"""

def main():
    embeddings_client = DialEmbeddingsClient(
        deployment="text-embedding-3-small-1",
        api_key=API_KEY
    )
    chat_client = DialChatCompletionClient(
        deployment_name="gpt-4o",
        api_key=API_KEY
    )
    db_config = {
        'host': 'localhost',
        'port': 5433,
        'database': 'vectordb',
        'user': 'postgres',
        'password': 'postgres'
    }
    text_processor = TextProcessor(embeddings_client, db_config)
    conversation = Conversation()

    # --- Index the manual every time you start the app ---
    text_processor.process_text_file(
        file_path="task/embeddings/microwave_manual.txt",
        chunk_size=300,
        overlap=40,
        dimensions=1536,
        truncate_table=True
    )
    print("Manual indexed and ready!")

    print("🎯 Microwave RAG Assistant (type 'exit' to quit)")
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ['exit', 'quit']:
            break

        # Step 1: Retrieval
        results = text_processor.search(
            query=user_input,
            search_mode=SearchMode.COSINE_DISTANCE,
            top_k=5,
            dimensions=1536
        )
        print("Retrieved context chunks:")
        for text, score in results:
            print(f"Score: {score:.3f} | {text[:100]}...")

        context = "\n\n".join([text for text, score in results])

        # Step 2: Augmentation
        user_prompt = USER_PROMPT.format(context=context, query=user_input)

        # Step 3: Generation
        messages = [
            Message(role=Role.SYSTEM, content=SYSTEM_PROMPT),
            Message(role=Role.USER, content=user_prompt)
        ]
        conversation.add_message(messages[0])
        conversation.add_message(messages[1])
        response = chat_client.get_completion(conversation.get_messages())
        print(f"\n💡 Answer:\n{response}")

if __name__ == "__main__":
    main()