import os
import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
from typing import List, Dict
import re

class StudyMate:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
        self.index = None
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better search"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def create_embeddings(self, texts: List[str]):
        """Create embeddings for text chunks"""
        self.documents = texts
        self.embeddings = self.model.encode(texts)
        
        # Create FAISS index for fast similarity search
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
    
    def search_similar_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """Find most relevant chunks for the query"""
        if self.index is None:
            return []
        
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx],
                    'score': float(score),
                    'index': int(idx)
                })
        
        return results
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using the context (placeholder for AI integration)"""
        # This is a simple approach - in production, you'd use OpenAI API or similar
        # For hackathon purposes, we'll do basic keyword matching and context extraction
        
        sentences = context.split('.')
        relevant_sentences = []
        
        query_words = set(query.lower().split())
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            if query_words.intersection(sentence_words):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return ". ".join(relevant_sentences[:3]) + "."
        else:
            return "I couldn't find a specific answer to your question in the uploaded document. Try rephrasing your question or check if the information exists in the PDF."

def main():
    st.set_page_config(
        page_title="StudyMate - AI PDF Search",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 StudyMate - AI PDF Search Tool")
    st.markdown("Upload a PDF and ask questions to get instant answers!")
    
    # Initialize StudyMate
    if 'studymate' not in st.session_state:
        st.session_state.studymate = StudyMate()
        st.session_state.pdf_processed = False
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf",
            help="Upload your study material, research paper, or any PDF document"
        )
        
        if uploaded_file is not None and not st.session_state.pdf_processed:
            with st.spinner("Processing PDF..."):
                # Extract text
                text = st.session_state.studymate.extract_text_from_pdf(uploaded_file)
                
                if text:
                    # Chunk text
                    chunks = st.session_state.studymate.chunk_text(text)
                    
                    # Create embeddings
                    st.session_state.studymate.create_embeddings(chunks)
                    st.session_state.pdf_processed = True
                    
                    st.success(f"✅ PDF processed successfully!")
                    st.info(f"📄 Document contains {len(chunks)} text chunks")
    
    # Main interface
    if st.session_state.pdf_processed:
        st.markdown("### Ask Questions About Your PDF")
        
        # Question input
        question = st.text_input(
            "What would you like to know?",
            placeholder="e.g., What is the main conclusion of this research?",
            help="Ask any question about the content in your uploaded PDF"
        )
        
        if question:
            with st.spinner("Searching for answers..."):
                # Search for relevant chunks
                similar_chunks = st.session_state.studymate.search_similar_chunks(question, top_k=3)
                
                if similar_chunks:
                    # Combine context from top chunks
                    context = " ".join([chunk['text'] for chunk in similar_chunks])
                    
                    # Generate answer
                    answer = st.session_state.studymate.generate_answer(question, context)
                    
                    # Display answer
                    st.markdown("### 🤖 Answer")
                    st.write(answer)
                    
                    # Show relevant passages
                    st.markdown("### 📋 Relevant Passages")
                    for i, chunk in enumerate(similar_chunks):
                        with st.expander(f"Passage {i+1} (Relevance: {chunk['score']:.3f})"):
                            st.write(chunk['text'])
                else:
                    st.warning("No relevant information found. Try a different question.")
        
        # Sample questions
        st.markdown("### 💡 Sample Questions")
        sample_questions = [
            "What is the main topic of this document?",
            "What are the key findings?",
            "Can you summarize the conclusion?",
            "What methodology was used?",
            "What are the limitations mentioned?"
        ]
        
        cols = st.columns(len(sample_questions))
        for i, q in enumerate(sample_questions):
            if cols[i].button(f"❓ {q}", key=f"sample_{i}"):
                st.rerun()
    
    else:
        st.info("👈 Please upload a PDF file to get started!")
        
        # Show features
        st.markdown("### ✨ Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🔍 Smart Search**
            - Semantic search through PDF content
            - Find relevant information quickly
            """)
        
        with col2:
            st.markdown("""
            **🤖 AI-Powered Answers**
            - Get contextual answers
            - Natural language processing
            """)
        
        with col3:
            st.markdown("""
            **📚 Study Helper**
            - Perfect for research papers
            - Great for study materials
            """)

if __name__ == "__main__":
    main()

# Requirements for your hackathon project:
# pip install streamlit PyPDF2 sentence-transformers faiss-cpu numpy

# To run the app:
# streamlit run studymate.py
