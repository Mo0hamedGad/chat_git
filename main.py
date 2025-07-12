
from app2 import ask_bot, load_index_and_chunks, SUPPORTED_TOPICS

# Then pass `databases` to `ask_bot()` or modify ask_bot to use it

if __name__ == "__main__":
    chat_history = []

    print("Welcome to the Medical Assistant Chatbot (type 'exit' to quit)\n")

    import os

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Take care! 👋")
            break

        answer = ask_bot(user_input, chat_history)
        print(f"Bot: {answer}\n")