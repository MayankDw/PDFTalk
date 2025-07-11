from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import PyPDF2
import io
import numpy as np
import pdfplumber
import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import json
import os
from typing import List, Dict, Any
import uvicorn

app = FastAPI(title="PDFTalk - AI PDF Question Answering", version="3.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class AdvancedPDFProcessor:
    def __init__(self):
        print("Loading models...")
        # Use the best embedding model for semantic understanding
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.bm25 = None
        self.tokenized_chunks = []
        self.chunk_metadata = []  # Store metadata about each chunk
        print("Models loaded successfully!")
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text using multiple methods and choose the best result"""
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
                texts.append(("pymupdf", mupdf_text))
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")
        
        # Method 2: pdfplumber (good for tables and complex layouts)
        try:
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                plumber_text = ""
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        plumber_text += f"[Page {page_num + 1}] {page_text}\n"
                if plumber_text.strip():
                    texts.append(("pdfplumber", plumber_text))
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}")
        
        # Method 3: PyPDF2 (fallback)
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            pypdf2_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    pypdf2_text += f"[Page {page_num + 1}] {page_text}\n"
            if pypdf2_text.strip():
                texts.append(("pypdf2", pypdf2_text))
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
        
        if not texts:
            return "Could not extract text from PDF"
        
        # Choose the extraction with the most content
        best_text = max(texts, key=lambda x: len(x[1].strip()))[1]
        print(f"Best extraction method: {max(texts, key=lambda x: len(x[1].strip()))[0]}")
        return self.clean_text(best_text)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\\1 \\2', text)  # Add space between camelCase
        text = re.sub(r'([a-zA-Z])([0-9])', r'\\1 \\2', text)  # Add space between letters and numbers
        text = re.sub(r'([0-9])([a-zA-Z])', r'\\1 \\2', text)  # Add space between numbers and letters
        text = re.sub(r'\\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'\\n\\s*\\n\\s*\\n+', '\\n\\n', text)  # Fix excessive line breaks
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 600, overlap: int = 150) -> List[str]:
        """Create overlapping chunks with better context preservation"""
        # Split by pages first if page markers exist
        if '[Page ' in text:
            page_sections = re.split(r'\\[Page \\d+\\]', text)
            page_sections = [section.strip() for section in page_sections if section.strip()]
        else:
            page_sections = [text]
        
        all_chunks = []
        
        for section in page_sections:
            # Split by paragraphs
            paragraphs = [p.strip() for p in section.split('\\n\\n') if p.strip()]
            if not paragraphs:
                paragraphs = [p.strip() for p in section.split('\\n') if p.strip()]
            
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
                            current_chunk = current_chunk + ' ' + sentence if current_chunk else sentence
                else:
                    # Add whole paragraph
                    if len(current_chunk + ' ' + paragraph) > chunk_size and current_chunk:
                        all_chunks.append(current_chunk.strip())
                        # Create overlap
                        words = current_chunk.split()
                        overlap_words = words[-min(overlap//5, len(words)//2):]
                        current_chunk = ' '.join(overlap_words) + ' ' + paragraph
                    else:
                        current_chunk = current_chunk + ' ' + paragraph if current_chunk else paragraph
            
            # Add final chunk
            if current_chunk.strip():
                all_chunks.append(current_chunk.strip())
        
        # Filter and clean chunks
        quality_chunks = []
        for i, chunk in enumerate(all_chunks):
            if len(chunk) > 50 and len(chunk.split()) > 10:  # Minimum quality threshold
                quality_chunks.append(chunk)
        
        print(f"Created {len(quality_chunks)} high-quality chunks")
        return quality_chunks
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for all chunks"""
        print("Creating embeddings...")
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        return embeddings
    
    def build_index(self, embeddings: np.ndarray):
        """Build search indexes"""
        print("Building search indexes...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        # Build BM25 index for keyword search
        self.tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        print("Search indexes built successfully")
    
    def process_pdf(self, pdf_content: bytes):
        """Process PDF and prepare for question answering"""
        text = self.extract_text_from_pdf(pdf_content)
        self.chunks = self.chunk_text(text)
        self.embeddings = self.create_embeddings(self.chunks)
        self.build_index(self.embeddings)
        return f"PDF processed successfully! Extracted {len(self.chunks)} text chunks ready for questions."
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[tuple]:
        """Advanced hybrid search combining semantic and keyword matching"""
        if self.index is None or self.bm25 is None:
            return []
        
        # 1. Semantic search
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        semantic_scores, semantic_indices = self.index.search(query_embedding, top_k)
        
        # 2. BM25 keyword search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # 3. Combine and rank results
        combined_scores = {}
        
        # Add semantic results (weight: 0.6)
        for i, idx in enumerate(semantic_indices[0]):
            if idx < len(self.chunks):
                combined_scores[idx] = semantic_scores[0][i] * 0.6
        
        # Add BM25 results (weight: 0.4)
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1
        for idx in bm25_indices:
            if idx < len(self.chunks) and max_bm25 > 0:
                normalized_bm25 = bm25_scores[idx] / max_bm25
                combined_scores[idx] = combined_scores.get(idx, 0) + (normalized_bm25 * 0.4)
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return chunks with their scores
        results = []
        for idx, score in sorted_results[:top_k]:
            results.append((self.chunks[idx], score, idx))
        
        return results
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Generate comprehensive answers using advanced RAG"""
        if not self.chunks:
            return {
                "answer": "No PDF has been uploaded yet. Please upload a PDF first.",
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
            # For technical questions, use more context
            relevant_chunks = [result[0] for result in search_results[:6]]
            base_confidence = 0.8
        elif is_definition:
            # For definitions, focused context is better
            relevant_chunks = [result[0] for result in search_results[:3]]
            base_confidence = 0.85
        else:
            # General questions
            relevant_chunks = [result[0] for result in search_results[:4]]
            base_confidence = 0.75
        
        # Generate answer using template-based approach
        context = "\\n\\n".join([f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(relevant_chunks)])
        
        # Create structured answer
        answer_parts = []
        
        # Look for direct answers in the most relevant chunks
        top_chunks = relevant_chunks[:3]
        
        # Simple but effective answer generation
        if is_definition:
            answer = self._generate_definition_answer(question, top_chunks)
        elif is_technical or is_howto:
            answer = self._generate_technical_answer(question, top_chunks)
        else:
            answer = self._generate_general_answer(question, top_chunks)
        
        # Calculate confidence based on relevance scores and answer quality
        avg_score = np.mean([result[1] for result in search_results[:3]])
        
        # Adjust confidence based on answer characteristics
        confidence = base_confidence
        if len(answer) > 100:
            confidence *= 1.1
        if len(answer) < 50:
            confidence *= 0.8
        
        # Boost confidence based on search relevance
        confidence *= (0.7 + 0.3 * min(avg_score, 1.0))
        confidence = min(confidence, 0.95)  # Cap at 95%
        
        return {
            "answer": answer,
            "confidence": confidence,
            "source_chunks": relevant_chunks[:3]
        }
    
    def _generate_definition_answer(self, question: str, chunks: List[str]) -> str:
        """Generate answers for definition-type questions"""
        # Look for key terms in the question
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
            # Try to find the most relevant sentences
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
            # Try to extract steps or processes
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
        # Combine the most relevant content
        combined = ' '.join(chunks[:2])
        
        # Extract the most relevant sentences
        sentences = [s.strip() for s in combined.split('.') if s.strip()]
        question_words = set(question.lower().split())
        
        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words & sentence_words)
            if overlap > 0:
                scored_sentences.append((sentence, overlap))
        
        if scored_sentences:
            # Sort by relevance and take top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s[0] for s in scored_sentences[:3]]
            return '. '.join(top_sentences) + '.'
        
        # Fallback to first chunk
        return chunks[0][:400] + ('...' if len(chunks[0]) > 400 else '')

pdf_processor = AdvancedPDFProcessor()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        content = await file.read()
        result = pdf_processor.process_pdf(content)
        return {"message": result, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/ask-question")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question", "")
    
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    result = pdf_processor.answer_question(question)
    return result

@app.get("/status")
async def get_status():
    return {
        "pdf_loaded": len(pdf_processor.chunks) > 0,
        "chunks_count": len(pdf_processor.chunks)
    }

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=8005)