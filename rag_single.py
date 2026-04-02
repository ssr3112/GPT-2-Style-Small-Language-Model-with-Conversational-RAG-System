import os
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from google import genai


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(" GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

# ==========================
# LOAD EMBEDDING MODEL

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================
# FILE LOADER

def load_document(file_path):
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    else:
        raise ValueError("Unsupported file type")


# ==========================
# TEXT SPLITTER

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


# ==========================
# CREATE VECTOR STORE

def create_vector_store(chunks):
    embeddings = embed_model.encode(chunks)
    embeddings = np.array(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, chunks


# ==========================
# RETRIEVER

def retrieve(query, index, chunks, top_k=3):
    query_vec = embed_model.encode([query])
    query_vec = np.array(query_vec)

    distances, indices = index.search(query_vec, top_k)
    results = [chunks[i] for i in indices[0]]

    return results


# ==========================
# RAG PIPELINE

def rag_query(file_path, query):
    
    text = load_document(file_path)

    
    chunks = split_text(text)

    
    index, chunks = create_vector_store(chunks)

    
    context_list = retrieve(query, index, chunks)
    context = "\n".join(context_list)

    
    prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

    
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

    return response.text


# ==========================
# TESTING

if __name__ == "__main__":

    file_path = "data/sample.txt"

    query = "What is this document about?"

    answer = rag_query(file_path, query)

    print("\n--- RAG OUTPUT ---\n")
    print(answer)