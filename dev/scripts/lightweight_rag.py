#!/usr/bin/env python3
"""
Lightweight RAG system that works on MacBook Air
Uses smaller models and batch processing
"""

import numpy as np
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

class LightweightRAG:
    def __init__(self, model_name="tfidf"):
        self.model_name = model_name
        self.vectorizer = None
        self.tfidf_matrix = None
        self.chunks = []
        self.model = None
        
    def load_document(self, pdf_path):
        """Load and chunk the PDF document"""
        print("Loading PDF document...")
        document_text = load_pdf(pdf_path)
        print(f"Document length (chars): {len(document_text)}")
        
        # Chunk the text
        print("Chunking text...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )
        
        self.chunks = splitter.split_text(document_text)
        print(f"Total chunks created: {len(self.chunks)}")
        return self.chunks
    
    def build_index(self):
        """Build the search index"""
        print("Building search index...")
        
        if self.model_name == "tfidf":
            # Use TF-IDF for lightweight search
            self.vectorizer = TfidfVectorizer(
                max_features=1000, 
                stop_words='english',
                ngram_range=(1, 2)  # Use unigrams and bigrams
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
            print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        print("Index built successfully!")
    
    def search(self, query, top_k=3):
        """Search for relevant chunks"""
        if self.model_name == "tfidf":
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for i, idx in enumerate(top_indices):
                results.append({
                    'chunk': self.chunks[idx],
                    'similarity': similarities[idx],
                    'rank': i + 1
                })
            return results
    
    def answer_question(self, question, top_k=3):
        """Answer a question using retrieved context"""
        # Search for relevant chunks
        results = self.search(question, top_k)
        
        # Combine context
        context = "\n\n".join([r['chunk'] for r in results])
        
        # Simple answer generation (you can enhance this)
        answer = f"""
Based on the Bank of America Online Banking Service Agreement, here's what I found:

{context[:1000]}...

[Note: This is a basic retrieval system. For full answer generation, you would need a language model like the one in the production version.]
"""
        return answer, results
    
    def save_index(self, filepath="rag_index.pkl"):
        """Save the built index for future use"""
        index_data = {
            'chunks': self.chunks,
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'model_name': self.model_name
        }
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath="rag_index.pkl"):
        """Load a previously built index"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                index_data = pickle.load(f)
            self.chunks = index_data['chunks']
            self.vectorizer = index_data['vectorizer']
            self.tfidf_matrix = index_data['tfidf_matrix']
            self.model_name = index_data['model_name']
            print(f"Index loaded from {filepath}")
            return True
        return False

def main():
    print("=== Lightweight RAG System ===")
    
    # Initialize RAG system
    rag = LightweightRAG()
    
    # Check if index already exists
    if not rag.load_index():
        # Build new index
        pdf_path = "dataset/Bank_of_America_Online Banking_Service Agreement.pdf"
        rag.load_document(pdf_path)
        rag.build_index()
        rag.save_index()
    
    # Test queries
    test_questions = [
        "What is the Zelle transfer limit for new users?",
        "How do I cancel a scheduled bill payment?",
        "What is the cut-off time for domestic wire transfers?",
        "What are the fees for international wire transfers?",
        "How do I enroll in online banking?"
    ]
    
    print("\n=== Testing RAG System ===")
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)
        
        answer, results = rag.answer_question(question)
        print(f"Answer: {answer}")
        
        print(f"\nRetrieved {len(results)} relevant chunks:")
        for i, result in enumerate(results):
            print(f"\nChunk {i+1} (similarity: {result['similarity']:.3f}):")
            print(f"{result['chunk'][:200]}...")

if __name__ == "__main__":
    main()
