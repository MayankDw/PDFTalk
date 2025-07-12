import streamlit as st
import PyPDF2
import numpy as np
import pdfplumber
import fitz
import re
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import torch
import os

# Configure page
st.set_page_config(
    page_title="PDFTalk - AI PDF Question Answering",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: bold;
        color: white;
    }
    .confidence-high { background-color: #4caf50; }
    .confidence-medium { background-color: #ff9800; }
    .confidence-low { background-color: #f44336; }
</style>
""", unsafe_allow_html=True)

class StreamlitPDFProcessor:
    def __init__(self):
        if 'pdf_processor' not in st.session_state:
            st.session_state.pdf_processor = self._initialize_empty_processor()
    
    def _initialize_empty_processor(self):
        """Initialize processor without loading models"""
        return {
            'embedding_model': None,
            'chunks': [],
            'embeddings': None,
            'index': None,
            'bm25': None,
            'tokenized_chunks': [],
            'processed': False,
            'models_loaded': False
        }
    
    def _load_models(self):
        """Load AI models with proper device handling"""
        processor = st.session_state.pdf_processor
        
        if processor['models_loaded']:
            return
        
        try:
            # Set device to CPU for Streamlit Cloud compatibility
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            torch.set_default_device('cpu')
            
            with st.spinner("Loading AI models..."):
                # Use a smaller, faster model for deployment
                model_name = 'sentence-transformers/all-MiniLM-L6-v2'
                processor['embedding_model'] = SentenceTransformer(
                    model_name, 
                    device='cpu'
                )
                processor['models_loaded'] = True
                st.success("✅ AI models loaded successfully")
                
        except Exception as e:
            st.error(f"Error loading models: {e}")
            # Fallback to even simpler model
            try:
                processor['embedding_model'] = SentenceTransformer(
                    'sentence-transformers/paraphrase-MiniLM-L3-v2',
                    device='cpu'
                )
                processor['models_loaded'] = True
                st.warning("⚠️ Using fallback model for compatibility")
            except Exception as e2:
                st.error(f"Failed to load fallback model: {e2}")
                processor['models_loaded'] = False
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text using multiple methods and choose the best result"""
        pdf_content = pdf_file.read()
        texts = []
        
        # Method 1: PyMuPDF (usually best)
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            mupdf_text = ""
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    mupdf_text += f"[Page {page_num + 1}] {page_text}\n"
            doc.close()
            if mupdf_text.strip():
                texts.append(("PyMuPDF", mupdf_text, len(mupdf_text)))
        except Exception as e:
            st.warning(f"PyMuPDF extraction failed: {e}")
        
        # Method 2: pdfplumber (good for tables and complex layouts)
        try:
            pdf_file.seek(0)  # Reset file pointer
            with pdfplumber.open(pdf_file) as pdf:
                plumber_text = ""
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        plumber_text += f"[Page {page_num + 1}] {page_text}\n"
                if plumber_text.strip():
                    texts.append(("pdfplumber", plumber_text, len(plumber_text)))
        except Exception as e:
            st.warning(f"pdfplumber extraction failed: {e}")
        
        # Method 3: PyPDF2 (fallback)
        try:
            pdf_file.seek(0)  # Reset file pointer
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            pypdf2_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    pypdf2_text += f"[Page {page_num + 1}] {page_text}\n"
            if pypdf2_text.strip():
                texts.append(("PyPDF2", pypdf2_text, len(pypdf2_text)))
        except Exception as e:
            st.warning(f"PyPDF2 extraction failed: {e}")
        
        if not texts:
            st.error("Could not extract text from PDF")
            return ""
        
        # Choose the extraction with the most content
        best_method, best_text, best_length = max(texts, key=lambda x: x[2])
        st.success(f"✅ Best extraction method: **{best_method}** ({best_length:,} characters)")
        return self.clean_text(best_text)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', text)
        text = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 600, overlap: int = 150) -> List[str]:
        """Create overlapping chunks with better context preservation"""
        # Split by pages first if page markers exist
        if '[Page ' in text:
            page_sections = re.split(r'\[Page \d+\]', text)
            page_sections = [section.strip() for section in page_sections if section.strip()]
        else:
            page_sections = [text]
        
        all_chunks = []
        
        for section in page_sections:
            # Split by paragraphs
            paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
            if not paragraphs:
                paragraphs = [p.strip() for p in section.split('\n') if p.strip()]
            
            current_chunk = ""
            
            for paragraph in paragraphs:
                # If paragraph is very long, split by sentences
                if len(paragraph) > chunk_size:
                    sentences = [s.strip() + '.' for s in paragraph.split('.') if s.strip()]
                    for sentence in sentences:
                        if len(current_chunk + ' ' + sentence) > chunk_size and current_chunk:
                            all_chunks.append(current_chunk.strip())
                            # Create overlap
                            words = current_chunk.split()
                            overlap_words = words[-min(overlap//5, len(words)//2):]
                            current_chunk = ' '.join(overlap_words) + ' ' + sentence
                        else:
                            current_chunk = (current_chunk + ' ' + sentence) if current_chunk else sentence
                else:
                    # Add whole paragraph
                    if len(current_chunk + ' ' + paragraph) > chunk_size and current_chunk:
                        all_chunks.append(current_chunk.strip())
                        # Create overlap
                        words = current_chunk.split()
                        overlap_words = words[-min(overlap//5, len(words)//2):]
                        current_chunk = ' '.join(overlap_words) + ' ' + paragraph
                    else:
                        current_chunk = (current_chunk + ' ' + paragraph) if current_chunk else paragraph
            
            # Add final chunk
            if current_chunk.strip():
                all_chunks.append(current_chunk.strip())
        
        # Filter and clean chunks
        quality_chunks = []
        for chunk in all_chunks:
            if len(chunk) > 50 and len(chunk.split()) > 10:
                quality_chunks.append(chunk)
        
        return quality_chunks
    
    def process_pdf(self, pdf_file):
        """Process PDF and prepare for question answering"""
        processor = st.session_state.pdf_processor
        
        # Load models first if not loaded
        if not processor['models_loaded']:
            self._load_models()
            if not processor['models_loaded']:
                st.error("Cannot process PDF: Models failed to load")
                return False
        
        with st.spinner("Extracting text from PDF..."):
            text = self.extract_text_from_pdf(pdf_file)
            if not text:
                return False
        
        with st.spinner("Creating text chunks..."):
            processor['chunks'] = self.chunk_text(text)
            st.success(f"✅ Created {len(processor['chunks'])} high-quality text chunks")
        
        with st.spinner("Generating embeddings..."):
            processor['embeddings'] = processor['embedding_model'].encode(
                processor['chunks'], 
                show_progress_bar=False
            )
            st.success("✅ Generated semantic embeddings")
        
        with st.spinner("Building search indexes..."):
            # Build FAISS index
            dimension = processor['embeddings'].shape[1]
            processor['index'] = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(processor['embeddings'])
            processor['index'].add(processor['embeddings'])
            
            # Build BM25 index
            processor['tokenized_chunks'] = [chunk.lower().split() for chunk in processor['chunks']]
            processor['bm25'] = BM25Okapi(processor['tokenized_chunks'])
            
            processor['processed'] = True
            st.success("✅ Search indexes built successfully")
        
        return True
    
    def semantic_search(self, query: str, top_k: int = 8) -> List[tuple]:
        """Advanced hybrid search combining semantic and keyword matching"""
        processor = st.session_state.pdf_processor
        
        if not processor['processed'] or not processor['models_loaded']:
            return []
        
        # 1. Semantic search
        query_embedding = processor['embedding_model'].encode([query])
        faiss.normalize_L2(query_embedding)
        semantic_scores, semantic_indices = processor['index'].search(query_embedding, top_k)
        
        # 2. BM25 keyword search
        query_tokens = query.lower().split()
        bm25_scores = processor['bm25'].get_scores(query_tokens)
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # 3. Combine and rank results
        combined_scores = {}
        
        # Add semantic results (weight: 0.6)
        for i, idx in enumerate(semantic_indices[0]):
            if idx < len(processor['chunks']):
                combined_scores[idx] = semantic_scores[0][i] * 0.6
        
        # Add BM25 results (weight: 0.4)
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1
        for idx in bm25_indices:
            if idx < len(processor['chunks']) and max_bm25 > 0:
                normalized_bm25 = bm25_scores[idx] / max_bm25
                combined_scores[idx] = combined_scores.get(idx, 0) + (normalized_bm25 * 0.4)
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return chunks with their scores
        results = []
        for idx, score in sorted_results[:top_k]:
            results.append((processor['chunks'][idx], score, idx))
        
        return results
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Generate comprehensive answers using advanced RAG"""
        processor = st.session_state.pdf_processor
        
        if not processor['processed']:
            return {
                "answer": "No PDF has been uploaded and processed yet. Please upload a PDF first.",
                "confidence": 0.0,
                "source_chunks": []
            }
        
        # Get relevant chunks with scores
        search_results = self.semantic_search(question, top_k=8)
        
        if not search_results:
            return {
                "answer": "I couldn't find relevant information in the PDF to answer your question.",
                "confidence": 0.0,
                "source_chunks": []
            }
        
        # Analyze question type and adjust strategy
        question_lower = question.lower()
        is_technical = any(word in question_lower for word in ['develop', 'implement', 'create', 'build', 'code', 'application', 'python', 'algorithm'])
        is_definition = any(word in question_lower for word in ['what is', 'define', 'explain', 'describe'])
        is_howto = any(word in question_lower for word in ['how to', 'how do', 'steps', 'process', 'method'])
        
        # Select best chunks based on question type
        if is_technical or is_howto:
            relevant_chunks = [result[0] for result in search_results[:6]]
            base_confidence = 0.8
        elif is_definition:
            relevant_chunks = [result[0] for result in search_results[:3]]
            base_confidence = 0.85
        else:
            relevant_chunks = [result[0] for result in search_results[:4]]
            base_confidence = 0.75
        
        # Generate answer using template-based approach
        if is_definition:
            answer = self._generate_definition_answer(question, relevant_chunks[:3])
        elif is_technical or is_howto:
            answer = self._generate_technical_answer(question, relevant_chunks[:3])
        else:
            answer = self._generate_general_answer(question, relevant_chunks[:3])
        
        # Calculate confidence
        avg_score = np.mean([result[1] for result in search_results[:3]])
        confidence = base_confidence
        
        if len(answer) > 100:
            confidence *= 1.1
        if len(answer) < 50:
            confidence *= 0.8
        
        confidence *= (0.7 + 0.3 * min(avg_score, 1.0))
        confidence = min(confidence, 0.95)
        
        return {
            "answer": answer,
            "confidence": confidence,
            "source_chunks": relevant_chunks[:3]
        }
    
    def _generate_definition_answer(self, question: str, chunks: List[str]) -> str:
        """Generate answers for definition-type questions"""
        question_words = set(question.lower().split())
        
        best_chunk = ""
        best_score = 0
        
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(question_words & chunk_words)
            if overlap > best_score:
                best_score = overlap
                best_chunk = chunk
        
        if best_chunk:
            sentences = [s.strip() for s in best_chunk.split('.') if s.strip()]
            relevant_sentences = []
            
            for sentence in sentences:
                if any(word in sentence.lower() for word in question_words):
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                return '. '.join(relevant_sentences[:3]) + '.'
            else:
                return best_chunk[:400] + ('...' if len(best_chunk) > 400 else '')
        
        return "Based on the document content, I found relevant information but couldn't extract a clear definition."
    
    def _generate_technical_answer(self, question: str, chunks: List[str]) -> str:
        """Generate answers for technical/how-to questions"""
        combined_text = ' '.join(chunks)
        
        # Look for structured information
        if 'step' in combined_text.lower() or 'process' in combined_text.lower():
            sentences = [s.strip() for s in combined_text.split('.') if s.strip()]
            steps = []
            
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['step', 'first', 'second', 'then', 'next', 'finally']):
                    steps.append(sentence)
            
            if steps:
                return "Based on the document, here are the key steps/processes: " + '. '.join(steps[:5]) + '.'
        
        # Look for technical details
        question_keywords = [word for word in question.lower().split() if len(word) > 3]
        relevant_parts = []
        
        for chunk in chunks:
            chunk_lower = chunk.lower()
            if any(keyword in chunk_lower for keyword in question_keywords):
                relevant_parts.append(chunk[:300])
        
        if relevant_parts:
            return "Based on the technical content in the document: " + ' '.join(relevant_parts[:2])
        
        return combined_text[:500] + ('...' if len(combined_text) > 500 else '')
    
    def _generate_general_answer(self, question: str, chunks: List[str]) -> str:
        """Generate answers for general questions"""
        combined = ' '.join(chunks[:2])
        
        sentences = [s.strip() for s in combined.split('.') if s.strip()]
        question_words = set(question.lower().split())
        
        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words & sentence_words)
            if overlap > 0:
                scored_sentences.append((sentence, overlap))
        
        if scored_sentences:
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s[0] for s in scored_sentences[:3]]
            return '. '.join(top_sentences) + '.'
        
        return chunks[0][:400] + ('...' if len(chunks[0]) > 400 else '')

def get_confidence_badge(confidence: float) -> str:
    """Generate confidence badge HTML"""
    percentage = int(confidence * 100)
    if confidence >= 0.7:
        badge_class = "confidence-high"
    elif confidence >= 0.4:
        badge_class = "confidence-medium"
    else:
        badge_class = "confidence-low"
    
    return f'<span class="confidence-badge {badge_class}">Confidence: {percentage}%</span>'

def main():
    # Header
    st.markdown('<h1 class="main-header">📄 PDFTalk - AI PDF Question Answering</h1>', unsafe_allow_html=True)
    st.markdown("**Upload a PDF document and ask questions about its content using advanced AI**")
    
    # Initialize processor
    processor = StreamlitPDFProcessor()
    
    # Sidebar for PDF upload and status
    with st.sidebar:
        st.header("📁 PDF Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to analyze and ask questions about"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ **{uploaded_file.name}** uploaded")
            
            # Process PDF button
            if st.button("🔄 Process PDF", type="primary"):
                if processor.process_pdf(uploaded_file):
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = uploaded_file.name
                    st.balloons()
                else:
                    st.error("Failed to process PDF")
        
        # Status section
        st.header("📊 Status")
        if 'pdf_processed' in st.session_state and st.session_state.pdf_processed:
            chunks_count = len(st.session_state.pdf_processor['chunks'])
            st.success(f"✅ PDF Ready: {chunks_count} chunks")
            st.info(f"📄 **{st.session_state.get('pdf_name', 'Unknown')}**")
        else:
            st.warning("⏳ No PDF processed yet")
        
        # Instructions
        st.header("💡 How to Use")
        st.markdown("""
        1. **Upload** a PDF file
        2. **Process** the PDF (wait for completion)
        3. **Ask questions** about the content
        4. **Get answers** with confidence scores
        """)
    
    # Chat interface
    st.header("💬 Ask Questions")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "confidence" in message:
                st.markdown(get_confidence_badge(message["confidence"]), unsafe_allow_html=True)
    
    # Chat input (must be outside columns)
    if prompt := st.chat_input("Ask a question about the PDF content..."):
        if 'pdf_processed' not in st.session_state or not st.session_state.pdf_processed:
            st.error("Please upload and process a PDF first!")
        else:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = processor.answer_question(prompt)
                
                st.markdown(result["answer"])
                st.markdown(get_confidence_badge(result["confidence"]), unsafe_allow_html=True)
                
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result["answer"],
                    "confidence": result["confidence"]
                })
    
    # Sources section
    st.header("📚 Source References")
    
    # Show source chunks for the last answer
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        if 'pdf_processed' in st.session_state and st.session_state.pdf_processed:
            # Get the last question to show relevant sources
            last_question = None
            for i in range(len(st.session_state.messages) - 1, -1, -1):
                if st.session_state.messages[i]["role"] == "user":
                    last_question = st.session_state.messages[i]["content"]
                    break
            
            if last_question:
                search_results = processor.semantic_search(last_question, top_k=3)
                
                col1, col2 = st.columns(2)
                for i, (chunk, score, idx) in enumerate(search_results):
                    with (col1 if i % 2 == 0 else col2):
                        with st.expander(f"📄 Source {i+1} (Relevance: {score:.3f})"):
                            st.markdown(chunk[:400] + ("..." if len(chunk) > 400 else ""))
    else:
        st.info("💡 Ask a question to see relevant source excerpts from the PDF")
    
    
    # Footer
    st.markdown("---")
    st.markdown("**PDFTalk** - Advanced AI-powered PDF Question Answering | Built with Streamlit")

if __name__ == "__main__":
    main()