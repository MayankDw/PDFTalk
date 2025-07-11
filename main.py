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
from transformers import pipeline
from rank_bm25 import BM25Okapi
import json
import os
from typing import List, Dict, Any
import uvicorn

app = FastAPI(title="PDFTalk - AI PDF Question Answering", version="1.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class PDFProcessor:
    def __init__(self):
        print("Loading models...")
        # Use a more powerful embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        # Use a high-accuracy but efficient QA model
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-large-squad2",
            tokenizer="deepset/roberta-large-squad2",
            return_overflowing_tokens=True,
            stride=128,
            max_answer_len=300
        )
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.bm25 = None
        self.tokenized_chunks = []
        print("Models loaded successfully!")
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        # Try multiple extraction methods for best quality
        texts = []
        
        # Method 1: PyMuPDF (usually best for text extraction)
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            mupdf_text = ""
            for page in doc:
                mupdf_text += page.get_text() + "\n"
            doc.close()
            if mupdf_text.strip():
                texts.append(("pymupdf", mupdf_text))
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")
        
        # Method 2: pdfplumber (good for complex layouts)
        try:
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                plumber_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        plumber_text += page_text + "\n"
                if plumber_text.strip():
                    texts.append(("pdfplumber", plumber_text))
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}")
        
        # Method 3: PyPDF2 (fallback)
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            pypdf2_text = ""
            for page in pdf_reader.pages:
                pypdf2_text += page.extract_text() + "\n"
            if pypdf2_text.strip():
                texts.append(("pypdf2", pypdf2_text))
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
        
        # Choose the best extraction (longest meaningful text)
        if not texts:
            return "Could not extract text from PDF"
        
        best_text = max(texts, key=lambda x: len(x[1].strip()))[1]
        return self.clean_text(best_text)
    
    def clean_text(self, text: str) -> str:
        # Clean and normalize extracted text
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between words
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)  # Add space between letters and numbers
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)  # Add space between numbers and letters
        # Remove extra spaces and clean up
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 150) -> List[str]:
        # Advanced chunking: split by paragraphs and sentences
        # First split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Split long paragraphs by sentences
            sentences = [s.strip() + '.' for s in paragraph.split('.') if s.strip()]
            
            for sentence in sentences:
                # Check if adding this sentence exceeds chunk size
                potential_chunk = current_chunk + ' ' + sentence if current_chunk else sentence
                
                if len(potential_chunk) > chunk_size and current_chunk:
                    # Save current chunk and start new one
                    chunks.append(current_chunk.strip())
                    
                    # Create overlap with previous chunk
                    words = current_chunk.split()
                    if len(words) > overlap // 4:  # Use word-based overlap
                        overlap_text = ' '.join(words[-(overlap // 4):])
                        current_chunk = overlap_text + ' ' + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = potential_chunk
        
        # Add the final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks and ensure minimum quality
        quality_chunks = []
        for chunk in chunks:
            if len(chunk) > 100 and len(chunk.split()) > 15:  # Minimum length requirements
                quality_chunks.append(chunk)
        
        return quality_chunks if quality_chunks else chunks  # Fallback to all chunks if filtering is too aggressive
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        embeddings = self.embedding_model.encode(chunks)
        return embeddings
    
    def build_index(self, embeddings: np.ndarray):
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        # Build BM25 index for hybrid search
        self.tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
    
    def process_pdf(self, pdf_content: bytes):
        text = self.extract_text_from_pdf(pdf_content)
        self.chunks = self.chunk_text(text)
        self.embeddings = self.create_embeddings(self.chunks)
        self.build_index(self.embeddings)
        return f"PDF processed successfully! Extracted {len(self.chunks)} text chunks."
    
    def semantic_search(self, query: str, top_k: int = 8) -> List[str]:
        if self.index is None or self.bm25 is None:
            return []
        
        # Hybrid search: combine semantic and keyword-based search
        
        # 1. Semantic search with embeddings
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        semantic_scores, semantic_indices = self.index.search(query_embedding, top_k)
        
        # 2. BM25 keyword search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # 3. Combine results with weighted scoring
        combined_scores = {}
        
        # Add semantic results (weight: 0.7)
        for i, idx in enumerate(semantic_indices[0]):
            if idx < len(self.chunks):
                combined_scores[idx] = combined_scores.get(idx, 0) + (semantic_scores[0][i] * 0.7)
        
        # Add BM25 results (weight: 0.3)
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1
        for idx in bm25_indices:
            if idx < len(self.chunks) and max_bm25 > 0:
                normalized_bm25 = bm25_scores[idx] / max_bm25
                combined_scores[idx] = combined_scores.get(idx, 0) + (normalized_bm25 * 0.3)
        
        # Sort by combined score and return top chunks
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        relevant_chunks = []
        for idx, score in sorted_indices[:top_k]:
            relevant_chunks.append(self.chunks[idx])
        
        return relevant_chunks
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        if not self.chunks:
            return {
                "answer": "No PDF has been uploaded yet. Please upload a PDF first.",
                "confidence": 0.0,
                "source_chunks": []
            }
        
        relevant_chunks = self.semantic_search(question)
        
        if not relevant_chunks:
            return {
                "answer": "I couldn't find relevant information in the PDF to answer your question.",
                "confidence": 0.0,
                "source_chunks": []
            }
        
        # Try multiple context combinations for better results
        best_result = None
        best_score = 0
        
        # Try different context combinations for best results
        context_strategies = [
            (relevant_chunks[:2], "focused"),
            (relevant_chunks[:4], "medium"),
            (relevant_chunks[:6], "comprehensive")
        ]
        
        for context_chunks, strategy in context_strategies:
            context = " ".join(context_chunks)
            
            # Ensure context isn't too long for the model
            if len(context) > 3500:
                # Truncate context intelligently
                context = context[:3500]
                last_period = context.rfind('.')
                if last_period > 3000:
                    context = context[:last_period + 1]
            
            try:
                # Use multiple approaches for better accuracy
                results = []
                
                # Standard QA
                result1 = self.qa_pipeline(
                    question=question,
                    context=context,
                    max_answer_len=250,
                    handle_impossible_answer=True,
                    top_k=3
                )
                
                # If result1 is a list, take the best one
                if isinstance(result1, list):
                    result1 = max(result1, key=lambda x: x['score'])
                
                results.append(result1)
                
                # Choose the best result
                for result in results:
                    if result['score'] > 0.01:  # Only consider results with some confidence
                        # Enhanced scoring based on answer quality
                        adjusted_score = result['score']
                        answer_text = result['answer'].lower()
                        question_lower = question.lower()
                        
                        # Boost for answer length (more complete answers)
                        if len(result['answer']) > 30:
                            adjusted_score *= 1.4
                        elif len(result['answer']) > 15:
                            adjusted_score *= 1.2
                        
                        # Boost if answer contains key question words
                        question_words = set(question_lower.split())
                        answer_words = set(answer_text.split())
                        common_words = question_words & answer_words
                        if len(common_words) > 0:
                            adjusted_score *= (1.0 + 0.1 * len(common_words))
                        
                        # Boost for answers that don't just repeat the question
                        if not answer_text.startswith(question_lower[:20]):
                            adjusted_score *= 1.1
                        
                        # Penalize very short answers
                        if len(result['answer']) < 10:
                            adjusted_score *= 0.5
                        
                        # Boost based on context strategy
                        if strategy == "focused":
                            adjusted_score *= 1.1  # Focused context often gives better answers
                        
                        if adjusted_score > best_score:
                            best_result = result
                            best_score = adjusted_score
                            best_result['source_chunks'] = context_chunks
                            
            except Exception as e:
                print(f"QA processing error: {e}")
                continue
        
        if best_result is None or best_score < 0.02:
            # Try a more direct text search as fallback
            question_keywords = question.lower().split()
            for chunk in relevant_chunks[:5]:
                chunk_lower = chunk.lower()
                matches = sum(1 for word in question_keywords if word in chunk_lower)
                if matches >= 2:  # If chunk contains at least 2 question keywords
                    return {
                        "answer": f"Based on the document content: {chunk[:300]}{'...' if len(chunk) > 300 else ''}",
                        "confidence": min(0.6, 0.3 + 0.1 * matches),
                        "source_chunks": [chunk]
                    }
            
            return {
                "answer": "I couldn't find relevant information in the PDF to answer your question. Please try rephrasing your question or check if the information exists in the document.",
                "confidence": best_score if best_result else 0.0,
                "source_chunks": relevant_chunks[:3]
            }
        
        # Final answer validation and confidence adjustment
        final_answer = best_result['answer']
        final_confidence = min(best_score, 0.99)  # Cap at 99%
        
        # Additional validation
        if len(final_answer) < 5:
            final_confidence *= 0.5
        
        return {
            "answer": final_answer,
            "confidence": final_confidence,
            "source_chunks": best_result.get('source_chunks', relevant_chunks[:3])
        }

pdf_processor = PDFProcessor()

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
    uvicorn.run(app, host="127.0.0.1", port=8003)