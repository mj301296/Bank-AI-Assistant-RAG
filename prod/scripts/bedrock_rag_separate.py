#!/usr/bin/env python3
"""
AWS Bedrock Production RAG System - Separate Retrieval and Generation
Production-ready implementation using AWS Bedrock Knowledge Base with Titan models
"""

import boto3
import json
import logging
import os
from typing import Dict, List, Optional
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bedrock_rag.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BedrockRAGSystem:
    """Production-ready Bedrock RAG System with separate retrieval and generation"""
    
    def __init__(self, knowledge_base_id: str, region: str = 'us-east-1'):
        """
        Initialize Bedrock RAG System
        
        Args:
            knowledge_base_id: AWS Bedrock Knowledge Base ID
            region: AWS region
        """
        self.knowledge_base_id = knowledge_base_id
        self.region = region
        
        # Initialize Bedrock client
        try:
            self.bedrock = boto3.client('bedrock-runtime', region_name=region)
            self.bedrock_agent = boto3.client('bedrock-agent-runtime', region_name=region)
            logger.info(f"Initialized Bedrock client for region: {region}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {str(e)}")
            raise
    
    def retrieve_documents(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents from Knowledge Base
        
        Args:
            question: User question
            top_k: Number of results to retrieve
            
        Returns:
            List of retrieved documents
        """
        try:
            logger.info(f"Retrieving documents for: {question[:50]}...")
            
            response = self.bedrock_agent.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={'text': question},
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': top_k
                    }
                }
            )
            
            return response.get('retrievalResults', [])
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def generate_response(self, question: str, context: str) -> str:
        """
        Generate response using Titan Text Lite
        
        Args:
            question: User question
            context: Retrieved context from Knowledge Base
            
        Returns:
            Generated response
        """
        try:
            # Truncate context to fit within token limits (roughly 3000 chars for safety)
            if len(context) > 3000:
                context = context[:3000] + "..."
            
            prompt = f"""Answer this banking question using the provided context:

Context: {context}

Question: {question}

Answer:"""

            response = self.bedrock.invoke_model(
                modelId='amazon.titan-text-lite-v1',
                body=json.dumps({
                    'inputText': prompt,
                    'textGenerationConfig': {
                        'maxTokenCount': 300,
                        'temperature': 0.1,
                        'topP': 0.9
                    }
                })
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['results'][0]['outputText']
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def query_knowledge_base(self, 
                           question: str, 
                           top_k: int = 5) -> Dict:
        """
        Query the Knowledge Base using separate retrieval and generation
        
        Args:
            question: User question
            top_k: Number of results to retrieve
            
        Returns:
            Response from Knowledge Base
        """
        try:
            # Step 1: Retrieve relevant documents
            documents = self.retrieve_documents(question, top_k)
            
            if not documents:
                return {
                    'answer': "I couldn't find relevant information to answer your question. Please try rephrasing or ask about a different banking topic.",
                    'citations': [],
                    'session_id': None,
                    'status': 'no_results'
                }
            
            # Step 2: Prepare context from retrieved documents
            context = "\n\n".join([
                f"Document {i+1}: {doc.get('content', '')}"
                for i, doc in enumerate(documents)
            ])
            
            # Step 3: Generate response using Titan Text Lite
            answer = self.generate_response(question, context)
            
            # Step 4: Prepare citations
            citations = []
            for doc in documents:
                if 'location' in doc:
                    citations.append({
                        'generated_text_chunk': doc.get('content', ''),
                        'location': doc['location']
                    })
            
            return {
                'answer': answer,
                'citations': citations,
                'session_id': None,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {str(e)}")
            return {
                'answer': f"I apologize, but I encountered an error while processing your question: {str(e)}",
                'citations': [],
                'session_id': None,
                'status': 'error',
                'error': str(e)
            }
    
    def health_check(self) -> Dict:
        """
        Perform health check on the RAG system
        
        Returns:
            Health status dictionary
        """
        try:
            logger.info("Performing health check...")
            
            # Test retrieval
            test_question = "What is online banking?"
            result = self.query_knowledge_base(test_question)
            
            if result['status'] == 'success':
                return {
                    'status': 'healthy',
                    'message': 'RAG system is working correctly',
                    'test_response': result['answer'][:100] + '...' if len(result['answer']) > 100 else result['answer']
                }
            else:
                return {
                    'status': 'unhealthy',
                    'message': 'RAG system has issues',
                    'error': result.get('error', 'Unknown error')
                }
                
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'message': 'Health check failed',
                'error': str(e)
            }
    
    def batch_query(self, questions: List[str]) -> List[Dict]:
        """
        Process multiple questions in batch
        
        Args:
            questions: List of questions to process
            
        Returns:
            List of responses
        """
        results = []
        for question in questions:
            result = self.query_knowledge_base(question)
            results.append({
                'question': question,
                'response': result
            })
        return results

def main():
    """Main function for testing the RAG system"""
    # Get configuration from environment variables
    knowledge_base_id = os.getenv('BEDROCK_KB_ID')
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    if not knowledge_base_id:
        print("❌ Error: BEDROCK_KB_ID environment variable not set")
        print("Please set it with: export BEDROCK_KB_ID='your-knowledge-base-id'")
        sys.exit(1)
    
    print(f"🏦 Bank AI Assistant - Production RAG System")
    print(f"Knowledge Base ID: {knowledge_base_id}")
    print(f"Region: {region}")
    print()
    
    # Initialize RAG system
    try:
        rag_system = BedrockRAGSystem(knowledge_base_id, region)
        print("✅ System initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize system: {str(e)}")
        sys.exit(1)
    
    # Perform health check
    health = rag_system.health_check()
    print(f"\n🔍 Health Check: {health['status']}")
    if health['status'] == 'healthy':
        print(f"✅ {health['message']}")
        print(f"Test response: {health['test_response']}")
    else:
        print(f"❌ {health['message']}")
        print(f"Error: {health['error']}")
        return
    
    # Interactive mode
    print(f"\n💬 Interactive Mode - Ask questions about banking services")
    print("Type 'quit' to exit")
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print("🤔 Thinking...")
            result = rag_system.query_knowledge_base(question)
            
            if result['status'] == 'success':
                print(f"\n🤖 Answer: {result['answer']}")
                
                if result['citations']:
                    print(f"\n📚 Sources:")
                    for i, citation in enumerate(result['citations'][:3], 1):
                        print(f"  {i}. {citation.get('location', {}).get('s3Location', {}).get('uri', 'Unknown source')}")
            else:
                print(f"❌ Error: {result['answer']}")
            
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
