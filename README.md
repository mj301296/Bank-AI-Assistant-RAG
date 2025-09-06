# Bank-AI-Assistant-RAG
Bank AI Assistance using AWS Bedrock+ OpenAI + Claude + ChromaDB


## Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
source rag_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Development Version
```bash
cd dev/scripts
python openai_rag.py
```

### 3. Run Evaluation
```bash
cd dev/scripts
python evaluate_rag.py
```

## Features
- **Multiple RAG Implementations**: OpenAI API, Hugging Face, AWS Bedrock
- **Comprehensive Evaluation**: F1 scores, similarity metrics, keyword accuracy
- **Production Ready**: Scalable architecture with multiple deployment options
- **Banking Domain**: Specialized for financial services Q&A

## Performance
- **Accuracy**: 70% F1 score, 100% success rate
- **Retrieval**: 60%+ similarity scores
- **Speed**: Sub-second response times

## File Descriptions
- `dev/scripts/openai_rag.py`: Main RAG system using OpenAI embeddings and GPT-3.5
- `dev/scripts/lightweight_rag.py`: Lightweight version using TF-IDF for resource-constrained environments
- `dev/scripts/evaluate_rag.py`: Comprehensive evaluation system with metrics and testing
- `dev/notebooks/dev_huggingface.ipynb`: Jupyter notebook for local Hugging Face implementation
- `dev/notebooks/dev_openai_rag.ipynb`: Jupyter notebook for OpenAI implementation
- `prod/notebooks/prod.bedrock.ipynb`: Production notebook using AWS Bedrock