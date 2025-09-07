# Development Environment - Bank AI Assistant RAG

This directory contains the development and experimental components of the Bank AI Assistant RAG system.

## 📁 Directory Structure

```
dev/
├── data/                    # Development data and indices
│   ├── openai_rag_index.pkl
│   └── rag_index_index.pkl
├── notebook/               # Jupyter notebooks for development
│   ├── dev_huggingface.ipynb
│   └── dev_openai_rag.ipynb
├── scripts/                # Development scripts
│   ├── lightweight_rag.py
│   └── openai_rag.py
├── tests/                  # Testing and evaluation
│   ├── evaluate_rag.py
│   └── results/
│       └── rag_evaluation_results.json
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Virtual environment activated
- Required dependencies installed

### Setup Development Environment

1. **Activate virtual environment:**
   ```bash
   source ../rag_env/bin/activate
   ```

2. **Install development dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

## 📓 Jupyter Notebooks

### `dev_huggingface.ipynb`
- **Purpose**: Local RAG implementation using Hugging Face models
- **Models**: SentenceTransformer + Flan-T5
- **Vector Store**: ChromaDB
- **Use Case**: Development and testing without external API costs

### `dev_openai_rag.ipynb`
- **Purpose**: OpenAI-based RAG implementation
- **Models**: OpenAI Embeddings + GPT-3.5-turbo
- **Vector Store**: FAISS
- **Use Case**: High-quality development with OpenAI models

## 🔧 Development Scripts

### `openai_rag.py`
- **Purpose**: Standalone OpenAI RAG implementation
- **Features**:
  - Interactive command-line interface
  - Batch processing capabilities
  - Performance metrics
  - Error handling

**Usage:**
```bash
cd scripts
python openai_rag.py
```

### `lightweight_rag.py`
- **Purpose**: Minimal RAG implementation for testing
- **Features**:
  - TF-IDF based retrieval
  - Simple text generation
  - Low resource requirements

## 🧪 Testing and Evaluation

### `evaluate_rag.py`
- **Purpose**: Comprehensive RAG system evaluation
- **Metrics**:
  - F1 Score
  - Retrieval Similarity
  - Keyword Accuracy
  - Response Quality

**Usage:**
```bash
cd tests
python evaluate_rag.py
```

### Evaluation Results
- **Location**: `tests/results/rag_evaluation_results.json`
- **Contains**: Performance metrics, test cases, and analysis

## 📊 Development Data

### Indices
- **`openai_rag_index.pkl`**: FAISS index for OpenAI embeddings
- **`rag_index_index.pkl`**: Alternative index for testing

### Source Documents
- **Location**: `../dataset/`
- **Format**: PDF documents (Bank of America service agreements)

## 🔄 Development Workflow

1. **Experiment** with new models in Jupyter notebooks
2. **Implement** features in development scripts
3. **Test** using evaluation framework
4. **Validate** performance metrics
5. **Migrate** successful implementations to production

## 🛠️ Development Tools

### Recommended IDE Setup
- **Jupyter Lab**: For notebook development
- **VS Code**: For script development
- **Python Debugger**: For troubleshooting

### Debugging Tips
- Use `logging` for detailed output
- Test with small datasets first
- Monitor memory usage with large models
- Use `gc.collect()` for memory cleanup

## 📈 Performance Considerations

### Memory Management
- **Local Models**: Monitor RAM usage (8GB+ recommended)
- **API Models**: Monitor token usage and costs
- **Vector Stores**: Consider index size and search speed

### Optimization Strategies
- **Chunking**: Optimize chunk size and overlap
- **Embeddings**: Choose appropriate embedding models
- **Retrieval**: Tune top-k parameters
- **Generation**: Adjust temperature and max_tokens

## 🚨 Common Issues

### Memory Issues
- **Problem**: Kernel crashes with large models
- **Solution**: Use smaller models or cloud APIs

### API Limits
- **Problem**: Rate limiting with OpenAI
- **Solution**: Implement retry logic and rate limiting

### Index Corruption
- **Problem**: Corrupted vector indices
- **Solution**: Rebuild indices from source documents

## 📚 Resources

### Documentation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FAISS Documentation](https://faiss.ai/)

### Model Resources
- [SentenceTransformers Models](https://www.sbert.net/docs/pretrained_models.html)
- [Flan-T5 Models](https://huggingface.co/google/flan-t5-base)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)

## 🔗 Related Files

- **Production**: `../prod/` - Production-ready implementations
- **Root**: `../README.md` - Project overview
- **Requirements**: `../requirements.txt` - Dependencies

---

**Note**: This is a development environment. For production deployment, see the `../prod/` directory.
