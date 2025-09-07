#!/usr/bin/env python3
"""
Simple Streamlit Web Interface for Bedrock RAG System
Focused on chat experience with suggested questions and clear context display
"""

import streamlit as st
import json
import os
import time
import uuid
from datetime import datetime
from bedrock_rag import BedrockRAGSystem

# Page configuration
st.set_page_config(
    page_title="Bank AI Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, focused design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2d5aa0 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .user-message {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .context-section {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        border-left: 4px solid #ff9800;
    }
    
    .sample-question {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .sample-question:hover {
        border-color: #1f4e79;
        background: #f8f9fa;
        transform: translateY(-2px);
    }
    
    .status-online {
        color: #4caf50;
        font-weight: bold;
    }
    
    .status-offline {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_rag_system():
    """Initialize the RAG system"""
    kb_id = os.getenv('BEDROCK_KB_ID')
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    if not kb_id:
        st.error("❌ Please set BEDROCK_KB_ID environment variable")
        return None
    
    try:
        return BedrockRAGSystem(kb_id, region)
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {str(e)}")
        return None

def display_sample_questions():
    """Display sample questions in an attractive format"""
    st.subheader("💡 Suggested Questions")
    
    sample_questions = [
        "What are the Zelle transfer limits for new users?",
        "How do I cancel a scheduled bill payment?",
        "What is the cut-off time for domestic wire transfers?",
        "What are the fees for international wire transfers?",
        "How do I enroll in online banking?",
        "What are the security requirements for online banking?",
        "How do I set up account alerts?",
        "What is the daily withdrawal limit for ATMs?"
    ]
    
    # Create columns for better layout
    cols = st.columns(2)
    for i, question in enumerate(sample_questions):
        with cols[i % 2]:
            if st.button(f"❓ {question}", key=f"sample_{i}", use_container_width=True):
                st.session_state.sample_question = question
                st.rerun()

def display_context_sources(citations):
    """Display the exact context sources used for the answer"""
    if not citations:
        return
    
    # Debug: Show citation count
    st.markdown(f"### 📚 Context Sources ({len(citations)} citation{'s' if len(citations) != 1 else ''})")
    
    source_counter = 1
    
    for citation in citations:
        if isinstance(citation, dict):
            # Check for retrievedReferences (Bedrock format)
            if 'retrievedReferences' in citation:
                # Debug: Show number of retrieved references
                ref_count = len(citation['retrievedReferences'])
                st.markdown(f"*Citation has {ref_count} retrieved reference{'s' if ref_count != 1 else ''}*")
                
                for ref in citation['retrievedReferences']:
                    if isinstance(ref, dict) and 'content' in ref:
                        content = ref['content']
                        if isinstance(content, dict) and 'text' in content:
                            source_text = content['text']
                            
                            with st.expander(f"Source {source_counter}", expanded=False):
                                # Display source content
                                st.markdown("**Content:**")
                                st.text_area("Source Content", source_text, height=200, key=f"context_{source_counter}_{uuid.uuid4().hex[:8]}", disabled=True, label_visibility="collapsed")
                                
                                # Display source location if available
                                if 'location' in ref and 's3Location' in ref['location']:
                                    s3_uri = ref['location']['s3Location'].get('uri', 'Unknown source')
                                    st.markdown(f"**Source:** `{s3_uri}`")
                                
                                # Display metadata if available
                                if 'metadata' in ref:
                                    metadata = ref['metadata']
                                    if 'x-amz-bedrock-kb-document-page-number' in metadata:
                                        page_num = metadata['x-amz-bedrock-kb-document-page-number']
                                        st.markdown(f"**Page:** {page_num}")
                            
                            source_counter += 1
            
            # Fallback for other citation formats
            elif 'generated_text_chunk' in citation:
                source_text = citation.get('generated_text_chunk', '')
                location = citation.get('location', {})
                
                with st.expander(f"Source {source_counter}", expanded=False):
                    # Display source content
                    if source_text:
                        st.markdown("**Content:**")
                        st.text_area("Source Content", source_text, height=150, key=f"context_{source_counter}_{uuid.uuid4().hex[:8]}", disabled=True, label_visibility="collapsed")
                    
                    # Display source location if available
                    if location and 's3Location' in location:
                        s3_uri = location['s3Location'].get('uri', 'Unknown source')
                        st.markdown(f"**Source:** `{s3_uri}`")
                
                source_counter += 1
        else:
            # Fallback for string citations
            with st.expander(f"Source {source_counter}", expanded=False):
                st.text_area("Source Content", str(citation), height=150, key=f"context_{source_counter}_{uuid.uuid4().hex[:8]}", disabled=True, label_visibility="collapsed")
            
            source_counter += 1

def main():
    """Main Streamlit application"""
    
    # Enhanced Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Bank AI Assistant</h1>
        <p>Ask questions about banking services and get accurate answers with source references</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Minimal and focused
    with st.sidebar:
        st.header("🔧 System Control")
        
        # Initialize RAG system
        if st.button("🚀 Initialize System", use_container_width=True):
            with st.spinner("Initializing RAG system..."):
                st.session_state.rag_system = initialize_rag_system()
                if st.session_state.rag_system:
                    st.success("✅ System ready!")
                else:
                    st.error("❌ Failed to initialize")
        
        # System status
        if st.session_state.rag_system:
            st.markdown('<p class="status-online">🟢 System Online</p>', unsafe_allow_html=True)
            
            # Quick health check
            if st.button("🔍 Quick Check", use_container_width=True):
                with st.spinner("Checking..."):
                    health = st.session_state.rag_system.health_check()
                    if health['status'] == 'healthy':
                        st.success("✅ All systems operational")
                    else:
                        st.error("❌ System issue detected")
        else:
            st.markdown('<p class="status-offline">🔴 System Offline</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Chat controls
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        
        # System info
        st.caption(f"**KB ID:** {os.getenv('BEDROCK_KB_ID', 'Not Set')}")
        st.caption(f"**Region:** {os.getenv('AWS_REGION', 'us-east-1')}")
    
    # Main content area
    if not st.session_state.rag_system:
        st.warning("⚠️ Please initialize the system from the sidebar first")
        
        # Show sample questions even when offline
        display_sample_questions()
        return
    
    # Sample questions section
    display_sample_questions()
    
    st.markdown("---")
    
    # Chat interface
    st.subheader("💬 Chat with Bank AI Assistant")
    
    # Handle sample question selection
    if hasattr(st.session_state, 'sample_question'):
        prompt = st.session_state.sample_question
        delattr(st.session_state, 'sample_question')
    else:
        # Chat input
        prompt = st.chat_input("Ask a question about banking services...")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Assistant message with context
            st.markdown(f"""
            <div class="assistant-message">
                <strong>Assistant:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Display context if available
            if "context" in message and message["context"]:
                st.markdown("""
                <div class="context-section">
                    <strong>📚 Context Sources:</strong>
                </div>
                """, unsafe_allow_html=True)
                display_context_sources(message["context"])
    
    # Process new prompt
    if prompt:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        st.markdown(f"""
        <div class="user-message">
            <strong>You:</strong> {prompt}
        </div>
        """, unsafe_allow_html=True)
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            result = st.session_state.rag_system.query_knowledge_base(prompt)
        
        if result['status'] == 'success':
            # Display assistant response
            st.markdown(f"""
            <div class="assistant-message">
                <strong>Assistant:</strong> {result['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            # Display context sources
            if result['citations']:
                st.markdown("""
                <div class="context-section">
                    <strong>📚 Context Sources:</strong>
                </div>
                """, unsafe_allow_html=True)
                display_context_sources(result['citations'])
            
            # Add to chat history with context
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": result['answer'],
                "context": result['citations']
            })
            
        else:
            st.error(f"❌ Error: {result['answer']}")
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": f"Error: {result['answer']}",
                "context": []
            })
        
        # Auto-refresh to show new message
        st.rerun()
    
    # Simple footer
    st.markdown("---")
    st.markdown("**🏦 Bank AI Assistant** | Powered by AWS Bedrock | Built with Streamlit")

if __name__ == "__main__":
    main()
