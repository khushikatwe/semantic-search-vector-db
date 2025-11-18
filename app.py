import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load the CSV database
df = pd.read_csv("semantic_db.csv")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text to embeddings
embeddings = model.encode(df["text"].tolist())

# Build FAISS vector search index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# Semantic search function
def semantic_search(query, top_k=3):
    q_vec = model.encode([query])
    distances, indices = index.search(q_vec, top_k)
    
    print("\nQuery:", query)
    print("Results:")
    for i in indices[0]:
        print("-", df.iloc[i]["text"])

# Test the search
if __name__ == "__main__":
    semantic_search("medical person")
