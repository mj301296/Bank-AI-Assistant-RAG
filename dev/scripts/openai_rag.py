#!/usr/bin/env python3
"""
RAG system using OpenAI embeddings and API
Much more reliable for MacBook Air - offloads heavy computation to OpenAI
"""

import numpy as np
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
import pickle
import time
from dotenv import load_dotenv
load_dotenv()


def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

class OpenAIRAG:
    def __init__(self, api_key=None, model="text-embedding-3-small"):
        self.model = model
        self.embeddings = []
        self.chunks = []
        
        # Set up OpenAI client
        if api_key:
            openai.api_key = api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter")
    
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
    
    def get_embeddings(self, texts, batch_size=100):
        """Get embeddings from OpenAI API with batching"""
        print(f"Getting embeddings for {len(texts)} chunks...")
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            try:
                response = openai.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error getting embeddings for batch {i//batch_size + 1}: {e}")
                # Retry with smaller batch
                if batch_size > 10:
                    print("Retrying with smaller batch size...")
                    return self.get_embeddings(texts, batch_size // 2)
                else:
                    raise e
        
        return all_embeddings
    
    def build_index(self):
        """Build the search index using OpenAI embeddings"""
        print("Building search index with OpenAI embeddings...")
        
        # Get embeddings for all chunks
        self.embeddings = self.get_embeddings(self.chunks)
        
        # Convert to numpy array for similarity search
        self.embeddings = np.array(self.embeddings)
        print(f"Embeddings shape: {self.embeddings.shape}")
        print("Index built successfully!")
    
    def search(self, query, top_k=3):
        """Search for relevant chunks using cosine similarity"""
        # Get embedding for the query
        query_embedding = self.get_embeddings([query])[0]
        query_vector = np.array(query_embedding).reshape(1, -1)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, self.embeddings).flatten()
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
        
        # Generate answer using OpenAI
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions about Bank of America's Online Banking Service Agreement. Use only the provided context to answer questions. If the answer is not in the context, say 'I don't have enough information to answer that question.'"},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
        except Exception as e:
            print(f"Error generating answer: {e}")
            answer = f"Based on the Bank of America Online Banking Service Agreement:\n\n{context[:1000]}..."
        
        return answer, results
    
    def save_index(self, filepath="openai_rag_index.pkl"):
        """Save the built index for future use"""
        index_data = {
            'chunks': self.chunks,
            'embeddings': self.embeddings,
            'model': self.model
        }
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath="openai_rag_index.pkl"):
        """Load a previously built index"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                index_data = pickle.load(f)
            self.chunks = index_data['chunks']
            self.embeddings = index_data['embeddings']
            self.model = index_data['model']
            print(f"Index loaded from {filepath}")
            return True
        return False

def main():
    print("=== OpenAI RAG System ===")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API key not found in environment variables.")
        api_key = input("Please enter your OpenAI API key: ").strip()
        if not api_key:
            print("No API key provided. Exiting.")
            return
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Initialize RAG system
    rag = OpenAIRAG()
    
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
