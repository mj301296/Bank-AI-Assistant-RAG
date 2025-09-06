#!/usr/bin/env python3
"""
RAG System Evaluation Script
Measures accuracy, precision, recall, and other metrics for the Bank AI Assistant
"""

import numpy as np
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
import pickle
import time
import json
from typing import List, Dict, Tuple
import re

class RAGEvaluator:
    def __init__(self, rag_system):
        self.rag = rag_system
        self.evaluation_results = {}
    
    def create_ground_truth_dataset(self):
        """Create a ground truth dataset with expected answers"""
        ground_truth = {
            "What is the Zelle transfer limit for new users?": {
                "answer": "For new Consumer users with Zelle enrollment <= 15 days: $500 daily, $1,000 weekly, $2,000 monthly. For new Small Business users: $2,000 daily, $45,000 weekly, $60,000 monthly.",
                "keywords": ["zelle", "transfer", "limit", "new", "users", "500", "1000", "2000", "consumer", "business"],
                "category": "transfer_limits"
            },
            "How do I cancel a scheduled bill payment?": {
                "answer": "You can cancel through the payment activity section on the website or call 800.432.1000 for consumer accounts or 866.758.5972 for small business accounts.",
                "keywords": ["cancel", "bill", "payment", "scheduled", "800.432.1000", "activity", "section"],
                "category": "payment_management"
            },
            "What is the cut-off time for domestic wire transfers?": {
                "answer": "The cut-off time for Same Business Day domestic wire transfers is 5:00 PM Eastern Time.",
                "keywords": ["cut-off", "time", "domestic", "wire", "transfer", "5:00", "pm", "eastern"],
                "category": "wire_transfers"
            },
            "What are the fees for international wire transfers?": {
                "answer": "International wire transfers sent in US Dollars cost $45.00. There is no fee for transfers sent in foreign currency, but exchange rate markups apply.",
                "keywords": ["fees", "international", "wire", "transfer", "45.00", "dollar", "foreign", "currency"],
                "category": "fees"
            },
            "How do I enroll in online banking?": {
                "answer": "You need to access the service using your User ID and password, along with any other security methods required by the bank.",
                "keywords": ["enroll", "online", "banking", "user", "id", "password", "security"],
                "category": "enrollment"
            },
            "What is the minimum Zelle transfer amount?": {
                "answer": "The minimum transfer amount for any single Zelle transfer is $1.00.",
                "keywords": ["minimum", "zelle", "transfer", "amount", "1.00"],
                "category": "transfer_limits"
            },
            "What are the daily limits for Zelle after 60 days?": {
                "answer": "For Consumer users after 60+ days: $3,500 daily. For Small Business users after 60+ days: $8,000 daily.",
                "keywords": ["daily", "limits", "zelle", "60", "days", "3500", "8000", "consumer", "business"],
                "category": "transfer_limits"
            },
            "Can I cancel a one-time immediate payment?": {
                "answer": "A one-time immediate payment cannot be canceled after it has been submitted.",
                "keywords": ["cancel", "one-time", "immediate", "payment", "cannot", "submitted"],
                "category": "payment_management"
            },
            "What is the fee for domestic wire transfers?": {
                "answer": "Domestic wire transfers cost $30.00 for both Consumer and Small Business accounts.",
                "keywords": ["fee", "domestic", "wire", "transfer", "30.00", "consumer", "business"],
                "category": "fees"
            },
            "What are the security requirements for online banking?": {
                "answer": "You need a User ID, password, and any other security methods required by the bank such as security questions or one-time passcodes.",
                "keywords": ["security", "requirements", "online", "banking", "user", "id", "password", "questions", "passcodes"],
                "category": "security"
            }
        }
        return ground_truth
    
    def evaluate_retrieval_accuracy(self, question: str, top_k: int = 3) -> Dict:
        """Evaluate how well the system retrieves relevant chunks"""
        results = self.rag.search(question, top_k)
        
        # Calculate similarity scores
        similarities = [result['similarity'] for result in results]
        avg_similarity = np.mean(similarities)
        max_similarity = np.max(similarities)
        
        # Check if any chunk contains key information
        ground_truth = self.create_ground_truth_dataset()
        if question in ground_truth:
            expected_keywords = ground_truth[question]['keywords']
            retrieved_text = ' '.join([result['chunk'].lower() for result in results])
            
            keyword_matches = sum(1 for keyword in expected_keywords 
                                if keyword.lower() in retrieved_text)
            keyword_coverage = keyword_matches / len(expected_keywords)
        else:
            keyword_coverage = None
        
        return {
            'avg_similarity': avg_similarity,
            'max_similarity': max_similarity,
            'keyword_coverage': keyword_coverage,
            'top_similarities': similarities
        }
    
    def evaluate_answer_quality(self, question: str, expected_answer: str) -> Dict:
        """Evaluate the quality of generated answers"""
        answer, results = self.rag.answer_question(question)
        
        # Basic metrics
        answer_length = len(answer.split())
        expected_length = len(expected_answer.split())
        length_ratio = answer_length / expected_length if expected_length > 0 else 0
        
        # Keyword matching
        answer_lower = answer.lower()
        expected_lower = expected_answer.lower()
        
        # Extract keywords from expected answer
        expected_keywords = re.findall(r'\b\w+\b', expected_lower)
        answer_keywords = re.findall(r'\b\w+\b', answer_lower)
        
        # Calculate keyword precision and recall
        expected_set = set(expected_keywords)
        answer_set = set(answer_keywords)
        
        if len(expected_set) > 0:
            precision = len(expected_set.intersection(answer_set)) / len(answer_set) if len(answer_set) > 0 else 0
            recall = len(expected_set.intersection(answer_set)) / len(expected_set)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1_score = 0
        
        # Check for specific information
        ground_truth = self.create_ground_truth_dataset()
        if question in ground_truth:
            expected_keywords = ground_truth[question]['keywords']
            keyword_presence = sum(1 for keyword in expected_keywords 
                                 if keyword.lower() in answer_lower)
            keyword_accuracy = keyword_presence / len(expected_keywords)
        else:
            keyword_accuracy = None
        
        return {
            'answer': answer,
            'expected_answer': expected_answer,
            'length_ratio': length_ratio,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'keyword_accuracy': keyword_accuracy,
            'answer_length': answer_length,
            'expected_length': expected_length
        }
    
    def run_comprehensive_evaluation(self) -> Dict:
        """Run comprehensive evaluation on all test questions"""
        ground_truth = self.create_ground_truth_dataset()
        evaluation_results = {
            'questions': {},
            'overall_metrics': {},
            'category_metrics': {}
        }
        
        print("=== Running Comprehensive RAG Evaluation ===\n")
        
        retrieval_scores = []
        answer_scores = []
        category_scores = {}
        
        for question, truth_data in ground_truth.items():
            print(f"Evaluating: {question}")
            
            # Evaluate retrieval
            retrieval_metrics = self.evaluate_retrieval_accuracy(question)
            
            # Evaluate answer quality
            answer_metrics = self.evaluate_answer_quality(question, truth_data['answer'])
            
            # Store results
            evaluation_results['questions'][question] = {
                'retrieval': retrieval_metrics,
                'answer_quality': answer_metrics,
                'category': truth_data['category']
            }
            
            # Aggregate scores
            retrieval_scores.append(retrieval_metrics['avg_similarity'])
            if answer_metrics['f1_score'] > 0:
                answer_scores.append(answer_metrics['f1_score'])
            
            # Category-based scoring
            category = truth_data['category']
            if category not in category_scores:
                category_scores[category] = {'retrieval': [], 'answer': []}
            category_scores[category]['retrieval'].append(retrieval_metrics['avg_similarity'])
            if answer_metrics['f1_score'] > 0:
                category_scores[category]['answer'].append(answer_metrics['f1_score'])
            
            print(f"  Retrieval Similarity: {retrieval_metrics['avg_similarity']:.3f}")
            print(f"  Answer F1 Score: {answer_metrics['f1_score']:.3f}")
            keyword_acc = answer_metrics['keyword_accuracy']
            if keyword_acc is not None:
                print(f"  Keyword Accuracy: {keyword_acc:.3f}")
            else:
                print(f"  Keyword Accuracy: N/A")
            print()
        
        # Calculate overall metrics
        evaluation_results['overall_metrics'] = {
            'avg_retrieval_similarity': np.mean(retrieval_scores),
            'avg_answer_f1': np.mean(answer_scores) if answer_scores else 0,
            'total_questions': len(ground_truth),
            'successful_answers': len(answer_scores)
        }
        
        # Calculate category metrics
        for category, scores in category_scores.items():
            evaluation_results['category_metrics'][category] = {
                'avg_retrieval_similarity': np.mean(scores['retrieval']),
                'avg_answer_f1': np.mean(scores['answer']) if scores['answer'] else 0,
                'question_count': len(scores['retrieval'])
            }
        
        return evaluation_results
    
    def print_evaluation_summary(self, results: Dict):
        """Print a summary of evaluation results"""
        print("=" * 60)
        print("RAG SYSTEM EVALUATION SUMMARY")
        print("=" * 60)
        
        overall = results['overall_metrics']
        print(f"Total Questions Evaluated: {overall['total_questions']}")
        print(f"Successful Answers: {overall['successful_answers']}")
        print(f"Average Retrieval Similarity: {overall['avg_retrieval_similarity']:.3f}")
        print(f"Average Answer F1 Score: {overall['avg_answer_f1']:.3f}")
        print()
        
        print("CATEGORY BREAKDOWN:")
        print("-" * 40)
        for category, metrics in results['category_metrics'].items():
            print(f"{category.upper()}:")
            print(f"  Retrieval Similarity: {metrics['avg_retrieval_similarity']:.3f}")
            print(f"  Answer F1 Score: {metrics['avg_answer_f1']:.3f}")
            print(f"  Questions: {metrics['question_count']}")
            print()
        
        # Performance interpretation
        print("PERFORMANCE INTERPRETATION:")
        print("-" * 40)
        retrieval_score = overall['avg_retrieval_similarity']
        answer_score = overall['avg_answer_f1']
        
        if retrieval_score >= 0.7:
            print("✅ Retrieval: EXCELLENT - Very relevant chunks found")
        elif retrieval_score >= 0.5:
            print("✅ Retrieval: GOOD - Mostly relevant chunks found")
        elif retrieval_score >= 0.3:
            print("⚠️  Retrieval: FAIR - Some relevant chunks found")
        else:
            print("❌ Retrieval: POOR - Few relevant chunks found")
        
        if answer_score >= 0.7:
            print("✅ Answer Quality: EXCELLENT - Very accurate answers")
        elif answer_score >= 0.5:
            print("✅ Answer Quality: GOOD - Mostly accurate answers")
        elif answer_score >= 0.3:
            print("⚠️  Answer Quality: FAIR - Somewhat accurate answers")
        else:
            print("❌ Answer Quality: POOR - Inaccurate answers")
    
    def save_evaluation_results(self, results: Dict, filename: str = "rag_evaluation_results.json"):
        """Save evaluation results to file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation results saved to {filename}")

def main():
    # Import the RAG system
    from openai_rag import OpenAIRAG
    
    print("Loading RAG system...")
    rag = OpenAIRAG()
    
    # Load existing index
    if not rag.load_index():
        print("No existing index found. Please run openai_rag.py first to build the index.")
        return
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag)
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    # Print summary
    evaluator.print_evaluation_summary(results)
    
    # Save results
    evaluator.save_evaluation_results(results)

if __name__ == "__main__":
    main()
