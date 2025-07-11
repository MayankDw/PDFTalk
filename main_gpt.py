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
import openai
import tiktoken
import json
import os
from typing import List, Dict, Any
import uvicorn

app = FastAPI(title="PDFTalk - AI PDF Question Answering", version="2.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class PDFProcessor:
    def __init__(self):
        print("Loading models...")
        # Use the best embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # Initialize OpenAI client for high-quality answers
        openai.api_key = os.getenv('OPENAI_API_KEY', 'sk-your_key_here')
        self.openai_client = openai
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
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
        text = re.sub(r'\\s+', ' ', text)
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'\\n\\s*\\d+\\s*\\n', '\\n', text)
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\\1 \\2', text)  # Add space between words
        text = re.sub(r'([a-zA-Z])(\\d)', r'\\1 \\2', text)  # Add space between letters and numbers
        text = re.sub(r'(\\d)([a-zA-Z])', r'\\1 \\2', text)  # Add space between numbers and letters
        # Remove extra spaces and clean up
        text = re.sub(r'\\s+', ' ', text).strip()
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        # Advanced chunking: split by paragraphs and sentences
        # First split by paragraphs
        paragraphs = [p.strip() for p in text.split('\\n\\n') if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in text.split('\\n') if p.strip()]
        
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
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text for GPT models"""
        return len(self.tokenizer.encode(text))
    
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
        
        # Prepare context for GPT with proper token management
        context_chunks = relevant_chunks[:6]  # Use top 6 chunks
        context = "\\n\\n".join([f"[Excerpt {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Ensure we don't exceed token limits (leave room for question and response)
        max_context_tokens = 3000
        while self.count_tokens(context) > max_context_tokens and context_chunks:
            context_chunks = context_chunks[:-1]  # Remove last chunk
            context = "\\n\\n".join([f"[Excerpt {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)])
        
        if not context_chunks:
            return {
                "answer": "The relevant content is too long to process. Please try a more specific question.",
                "confidence": 0.0,
                "source_chunks": relevant_chunks[:3]
            }
        
        # Create the prompt for GPT
        system_prompt = """You are an expert AI assistant that answers questions based strictly on provided PDF content. 

Rules:
1. Answer ONLY using information from the provided excerpts
2. If the answer isn't in the excerpts, say "The provided content doesn't contain enough information to answer this question"
3. Be precise and detailed when information is available
4. Reference which excerpt(s) you used
5. Don't make assumptions beyond what's written"""
        
        user_prompt = f"""Based on the following excerpts from a PDF document, please answer this question:

**Question:** {question}

**PDF Content:**
{context}

**Instructions:** Provide a comprehensive answer based solely on the information in these excerpts. If the information is insufficient, clearly state that."""
        
        try:
            # Use GPT-3.5-turbo for high-quality responses
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.1,  # Low temperature for consistent, factual answers
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Calculate confidence based on answer quality and content
            confidence = 0.85  # Start with high confidence for GPT responses
            
            # Adjust confidence based on answer characteristics
            if "doesn't contain enough information" in answer.lower():
                confidence = 0.2
            elif "insufficient" in answer.lower() or "not enough" in answer.lower():
                confidence = 0.3
            elif len(answer) < 20:
                confidence = 0.4
            elif "based on the provided" in answer.lower() or "according to" in answer.lower():
                confidence = 0.9  # High confidence when GPT explicitly references sources
            
            # Boost confidence for detailed answers
            if len(answer) > 100:
                confidence = min(confidence * 1.1, 0.95)
            
            return {
                "answer": answer,
                "confidence": confidence,
                "source_chunks": context_chunks
            }
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            
            # Fallback to direct content extraction if OpenAI fails
            fallback_answer = f"Based on the document content, here are the most relevant sections:\\n\\n{context_chunks[0][:400]}{'...' if len(context_chunks[0]) > 400 else ''}"
            
            if len(context_chunks) > 1:
                fallback_answer += f"\\n\\nAdditional relevant content:\\n{context_chunks[1][:300]}{'...' if len(context_chunks[1]) > 300 else ''}"
            
            return {
                "answer": fallback_answer,
                "confidence": 0.7,  # Good confidence for direct extraction
                "source_chunks": context_chunks[:2]
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
    uvicorn.run(app, host="127.0.0.1", port=8004)