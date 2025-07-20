import os
import pickle
import faiss
import numpy as np
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from textblob import TextBlob

# === Load environment variables ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Load embedding model ===
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

SUPPORTED_TOPICS = ["infection","anemia","injury","autoimmnune","cbc"]
databases: dict | None = None 

# === Load FAISS index and chunks ===
def load_index_and_chunks(supported_topics):
    databases = {}
    os.makedirs("indices", exist_ok=True)

    for topic in supported_topics:
        
        index_path = f"embeddings/{topic}.faiss"
        chunks_path = f"embeddings/{topic}_chunks.pkl"

        if os.path.exists(index_path) and os.path.exists(chunks_path):
            index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
        else:
            return topic + "not found"

        databases[topic] = {"index": index, "chunks": chunks}


    return databases

# === find the topic for the question ====
def find_best_topic(query, model, databases, top_k=3):
    query_embedding = model.encode([query])
    best_score = float("inf")
    best_topic = None

    for topic, data in databases.items():
        D, _ = data["index"].search(query_embedding, top_k)
        if D[0][0] < best_score:
            best_score = D[0][0]
            best_topic = topic
    return best_topic

def correct_spelling(text):
    blob = TextBlob(text)
    return str(blob.correct())

# === Get top matching chunks ===
# def get_top_chunks(query, model, index, text_chunks, top_k=5, threshold=2.1):
#     query_embedding = model.encode([query])
#     distances, indices = index.search(np.array(query_embedding), top_k)
    
#     # Normalize distances for cosine-like scores (0 = perfect match)
#     max_dist = max(distances[0])
#     if max_dist > threshold:
#         return None  # No good match found
    
#     return [text_chunks[i] for i in indices[0]]

def get_top_chunks(query, model, index, text_chunks, top_k=5, threshold=2.1):
    print("Entered get_top_chunks")
    try:
        import numpy as np
        print("NumPy version:", np.__version__)
        query_embedding = model.encode([query])
        print("Query embedding:", query_embedding)
        distances, indices = index.search(np.array(query_embedding), top_k)
        print("Distances:", distances)
        print("Indices:", indices)
    except Exception as e:
        print("Error in get_top_chunks:", str(e))
        return None

# === Generate answer with OpenAI ===
def generate_answer(query, top_chunks, chat_history):
    context = "\n\n".join(top_chunks)

    system_prompt = """
You are a helpful medical assistant trained to answer patient questions using only the provided documents.
- You focus on anemia, infection, autoimmune diseases, and injury.
- Only use the info from the documents.
- If unsure, reply: "I'm sorry, I couldn't find that in the available medical information."
- Use clear, non-technical language.
- Recommend seeing a doctor for personal health concerns.
"""

    # Build new user prompt with current context
    user_message = f"""
Answer the following question using only the information below.

{context}

Question: {query}
"""

    # Build the messages list from scratch
    messages = [{"role": "system", "content": system_prompt}] + chat_history
    messages.append({"role": "user", "content": user_message})

    # Call GPT
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3
    )

    answer = response["choices"][0]["message"]["content"]

    # Update chat history correctly
    chat_history.append({"role": "user", "content": query})      # JUST the question
    chat_history.append({"role": "assistant", "content": answer})

    # Limit to last 10 messages (keep system prompt separately)
    if len(chat_history) > 10:
        chat_history[:] = chat_history[-10:]

    return answer

# === small talk handeling ===
def is_small_talk(query):
    query = query.lower().strip()

    small_talk_patterns = {
        "greeting": {
            "keywords": ["hi", "hello", "hey", "good morning", "good evening"],
            "response": "Hello! How can I assist you today?"
        },
        "gratitude": {
            "keywords": ["thank you", "thanks", "thx", "appreciate it"],
            "response": "You're welcome! Let me know if you need anything else."
        },
        "farewell": {
            "keywords": ["bye", "goodbye", "see you", "later", "cya"],
            "response": "Goodbye! Stay healthy and take care."
        },
        "who_are_you": {
            "keywords": ["who are you", "what are you", "what is your name"],
            "response": "I'm a health assistant chatbot designed to help with medical questions."
        },
        "what_can_you_do": {
            "keywords": ["what can you do"],
            "response": "I can answer questions about conditions like anemia, infections, injuries, and autoimmune diseases."
        },
        "affirmation": {
            "keywords": ["yes", "sure", "of course", "yeah", "yep"],
            "response": "Okay! Just let me know what you need."
        },
        "negation": {
            "keywords": ["no", "not now", "maybe later"],
            "response": "No worries! I'm here whenever you need help."
        }
    }

    for intent in small_talk_patterns.values():
        for keyword in intent["keywords"]:
            if keyword in query:
                return intent["response"]

    return None

# === making the datbase ====
def make_database():
    global databases
    if databases is None:
        databases = load_index_and_chunks(SUPPORTED_TOPICS)
    
# === Main chatbot interface ===
def ask_bot(query, chat_history):
    make_database()
    small_talk_reply = is_small_talk(query)
    if small_talk_reply:
        return small_talk_reply
    
    topic = find_best_topic(query, embedding_model, databases)
    if not topic:
        return "Sorry, I couldn't determine the topic from your question."

    corrected_query = correct_spelling(query)
    top_chunks = get_top_chunks(query, embedding_model, databases[topic]["index"], databases[topic]["chunks"])
    if not top_chunks:
        return "Sorry, I couldn't find a relevant answer in the documents."

    answer = generate_answer(corrected_query, top_chunks,chat_history)

    return answer

