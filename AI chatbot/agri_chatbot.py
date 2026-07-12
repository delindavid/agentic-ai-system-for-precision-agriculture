import requests
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# load FAISS index
index = faiss.read_index("agri_vector/agri_index.faiss")

with open("agri_vector/answers.pkl", "rb") as f:
    answers = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

SIMILARITY_THRESHOLD = 1.2


def retrieve(query):

    query_embedding = model.encode([query])

    D, I = index.search(query_embedding, 1)

    distance = D[0][0]

    if distance > SIMILARITY_THRESHOLD:
        return None

    return answers[I[0][0]]


def ask_mistral(prompt):

    url = "http://localhost:11434/api/generate"

    response = requests.post(url, json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    })

    return response.json()["response"]


while True:

    user = input("Farmer: ")

    context = retrieve(user)

    if context is None:
        print("AgriBot: I can only answer agriculture related questions.")
        continue

    prompt = f"""
You are an AGRICULTURE ASSISTANT helping farmers.

Rules:
- Answer only using the provided context
- Give short and clear answers
- Focus on crops, soil, fertilizer, irrigation and farming

Context:
{context}

Question:
{user}

Answer:
"""

    answer = ask_mistral(prompt)

    print("AgriBot:", answer)