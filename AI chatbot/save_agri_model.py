import json
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os

os.makedirs("agri_vector", exist_ok=True)

# load dataset
with open("agriculture_dataset.json") as f:
    data = json.load(f)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(questions)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, "agri_vector/agri_index.faiss")

with open("agri_vector/answers.pkl", "wb") as f:
    pickle.dump(answers, f)

print("Agriculture vector database saved")