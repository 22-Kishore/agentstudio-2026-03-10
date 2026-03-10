import os
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import numpy as np
import pickle

# Module: document_loader.py
def load_documents_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def load_documents_from_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Module: chunking.py
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Module: embeddings.py
def generate_embeddings(chunks):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(chunks)
    return embeddings, vectorizer

# Module: vector_storage.py
def store_embeddings(embeddings, file_path='embeddings.pkl'):
    with open(file_path, 'wb') as file:
        pickle.dump(embeddings, file)

def load_embeddings(file_path='embeddings.pkl'):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Module: retrieval.py
def retrieve_similar_chunks(query, vectorizer, embeddings, top_k=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, embeddings).flatten()
    indices = np.argsort(similarities)[-top_k:][::-1]
    return indices, similarities[indices]

# Module: rag_pipeline.py
class RAGDocumentSearch:
    def __init__(self, llm_api_key):
        self.llm_api_key = llm_api_key
        openai.api_key = self.llm_api_key

    def generate_response(self, query, chunks, indices):
        relevant_chunks = [chunks[i] for i in indices]
        context = " ".join(relevant_chunks)
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Context: {context}\n\nQuestion: {query}\nAnswer:",
            max_tokens=150
        )
        return response.choices[0].text.strip()

# Main script
if __name__ == "__main__":
    # Load documents
    text = load_documents_from_pdf('sample.pdf')
    # text = load_documents_from_text('sample.txt')

    # Chunk documents
    chunks = chunk_text(text)

    # Generate embeddings
    embeddings, vectorizer = generate_embeddings(chunks)

    # Store embeddings
    store_embeddings((embeddings, vectorizer))

    # Load embeddings
    embeddings, vectorizer = load_embeddings()

    # Initialize RAG pipeline
    rag = RAGDocumentSearch(llm_api_key='your-openai-api-key')

    # Retrieve similar chunks
    query = "What is the main topic of the document?"
    indices, _ = retrieve_similar_chunks(query, vectorizer, embeddings)

    # Generate response
    response = rag.generate_response(query, chunks, indices)
    print("Response:", response)

# GitHub setup (to be executed in terminal)
"""
git init
git add .
git commit -m "Initial commit of RAG based document search system"
git branch -M main
git remote add origin https://github.com/yourusername/rag-document-search.git
git push -u origin main
"""