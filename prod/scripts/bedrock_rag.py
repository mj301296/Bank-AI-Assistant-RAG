#!/usr/bin/env python3
"""
AWS Bedrock Production RAG System
Production-ready implementation using AWS Bedrock Knowledge Base and Claude 3 Haiku
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
    """Production-ready Bedrock RAG System"""
    
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
    
    def query_knowledge_base(self, 
                           question: str, 
                           max_tokens: int = 500,
                           temperature: float = 0.1,
                           top_k: int = 5) -> Dict:
        """
        Query the Bedrock Knowledge Base
        
        Args:
            question: User question
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0.0-1.0)
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            logger.info(f"Querying knowledge base: {question[:50]}...")
            
            response = self.bedrock_agent.retrieve_and_generate(
                input={'text': question},
                retrieveAndGenerateConfiguration={
                    'type': 'KNOWLEDGE_BASE',
                    'knowledgeBaseConfiguration': {
                        'knowledgeBaseId': self.knowledge_base_id,
                        'modelArn': f'arn:aws:bedrock:{self.region}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0',
                        'retrievalConfiguration': {
                            'vectorSearchConfiguration': {
                                'numberOfResults': top_k
                            }
                        },
                        'generationConfiguration': {
                            'inferenceConfig': {
                                'textInferenceConfig': {
                                    'maxTokens': max_tokens,
                                    'temperature': temperature,
                                    'topP': 0.9,
                                    'stopSequences': []
                                }
                            }
                        }
                    }
                }
            )
            
            result = {
                'answer': response['output']['text'],
                'citations': response.get('citations', []),
                'session_id': response.get('sessionId', ''),
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'knowledge_base_id': self.knowledge_base_id,
                    'region': self.region,
                    'max_tokens': max_tokens,
                    'temperature': temperature,
                    'top_k': top_k
                }
            }
            
            logger.info(f"Successfully generated answer for question")
            return result
            
        except Exception as e:
            error_msg = f"Error querying knowledge base: {str(e)}"
            logger.error(error_msg)
            return {
                'answer': f"I apologize, but I encountered an error while processing your question: {str(e)}",
                'citations': [],
                'session_id': '',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def batch_query(self, questions: List[str]) -> List[Dict]:
        """
        Process multiple questions in batch
        
        Args:
            questions: List of questions
            
        Returns:
            List of response dictionaries
        """
        logger.info(f"Processing batch of {len(questions)} questions")
        results = []
        
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}")
            result = self.query_knowledge_base(question)
            results.append({
                'question': question,
                'question_id': i,
                **result
            })
        
        return results
    
    def health_check(self) -> Dict:
        """
        Perform health check on the system
        
        Returns:
            Health status dictionary
        """
        try:
            # Test with a simple question
            test_question = "What is online banking?"
            result = self.query_knowledge_base(test_question, max_tokens=50)
            
            return {
                'status': 'healthy' if result['status'] == 'success' else 'unhealthy',
                'knowledge_base_id': self.knowledge_base_id,
                'region': self.region,
                'timestamp': datetime.now().isoformat(),
                'test_response': result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer']
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

def main():
    """Main function for testing the Bedrock RAG System"""
    
    # Configuration
    KB_ID = os.getenv('BEDROCK_KB_ID', 'YOUR_KNOWLEDGE_BASE_ID_HERE')
    REGION = os.getenv('AWS_REGION', 'us-east-1')
    
    if KB_ID == 'YOUR_KNOWLEDGE_BASE_ID_HERE':
        print("⚠️  Please set your Knowledge Base ID:")
        print("   export BEDROCK_KB_ID='your-actual-kb-id'")
        print("   or update the KB_ID variable in this script")
        return
    
    # Initialize system
    try:
        rag_system = BedrockRAGSystem(KB_ID, REGION)
        logger.info("Bedrock RAG System initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        return
    
    # Health check
    print("🔍 Performing health check...")
    health = rag_system.health_check()
    print(f"Status: {health['status']}")
    if health['status'] == 'unhealthy':
        print(f"Error: {health.get('error', 'Unknown error')}")
        return
    
    # Test questions
    test_questions = [
        "What are the Zelle transfer limits for new users?",
        "How do I cancel a scheduled bill payment?",
        "What is the cut-off time for domestic wire transfers?",
        "What are the fees for international wire transfers?",
        "How do I enroll in online banking?",
        "What are the security requirements for online banking?"
    ]
    
    print(f"\n🚀 Testing with {len(test_questions)} questions...")
    
    # Process questions
    results = rag_system.batch_query(test_questions)
    
    # Display results
    print("\n" + "="*80)
    print("BEDROCK RAG SYSTEM RESULTS")
    print("="*80)
    
    for result in results:
        print(f"\n📝 Question {result['question_id']}: {result['question']}")
        print("-" * 60)
        
        if result['status'] == 'success':
            print(f"✅ Answer: {result['answer']}")
            print(f"📚 Citations: {len(result['citations'])} sources")
            print(f"🕒 Timestamp: {result['timestamp']}")
        else:
            print(f"❌ Error: {result['answer']}")
    
    # Save results
    output_file = f"../tests/bedrock_rag_results/bedrock_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("🎉 Bedrock RAG System test completed!")

if __name__ == "__main__":
    main()
