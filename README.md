# PDFTalk - AI PDF Question Answering Tool

An advanced AI-powered tool that allows you to upload PDF documents and ask questions about their content using semantic search and state-of-the-art question-answering models.

## Features

- **📄 Smart PDF Processing**: Multiple extraction methods (PyMuPDF, pdfplumber, PyPDF2)
- **🔍 Advanced Search**: Hybrid semantic + keyword search with BM25
- **🤖 High-Accuracy AI**: Template-based answer generation with 75-95% confidence
- **💬 Chat Interface**: Interactive Streamlit web app with real-time responses
- **📊 Source Attribution**: Shows relevant text chunks with relevance scores
- **🎯 Question Intelligence**: Detects technical, definition, and how-to questions
- **⚡ Fast Processing**: Optimized chunking and embedding generation
- **🔧 Self-Contained**: No external API dependencies required

## Technology Stack

- **Frontend**: Streamlit (Interactive web interface)
- **AI Models**: 
  - Sentence Transformers (all-mpnet-base-v2) for embeddings
  - Advanced RAG with template-based answer generation
- **Vector Search**: FAISS + BM25 hybrid search
- **PDF Processing**: PyMuPDF + pdfplumber + PyPDF2
- **Backend**: Python with session state management

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd PDFTalk
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run streamlit_app.py --server.port 8080
```

Or use the launcher script:
```bash
python run_streamlit.py
```

4. Open your browser and navigate to `http://localhost:8080`

## Usage

1. **Upload PDF**: Use the upload area to select or drag & drop a PDF file
2. **Wait for Processing**: The system will extract text and create embeddings
3. **Ask Questions**: Type questions about the PDF content in the chat interface
4. **Review Answers**: Get AI-generated answers with confidence scores and source references

## API Endpoints

- `GET /`: Main application interface
- `POST /upload-pdf`: Upload and process PDF files
- `POST /ask-question`: Submit questions and get answers
- `GET /status`: Check system status and PDF load state

## Model Accuracy

The tool uses high-accuracy models:
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions, optimized for semantic similarity)
- **QA Model**: deepset/roberta-base-squad2 (fine-tuned on SQuAD 2.0 dataset)
- **Vector Search**: FAISS with cosine similarity for efficient retrieval

## Contributing

This is an open-source project. Contributions are welcome! Please feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## License

Open source - feel free to use, modify, and distribute.