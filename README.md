# PDFTalk - AI PDF Question Answering Tool

An advanced AI-powered tool that allows you to upload PDF documents and ask questions about their content using semantic search and state-of-the-art question-answering models.

## Features

- **PDF Upload**: Drag & drop or browse to upload PDF files
- **Semantic Search**: Uses sentence transformers for high-accuracy content retrieval
- **Question Answering**: Employs RoBERTa-based QA model for precise answers
- **Confidence Scoring**: Shows confidence levels for each answer
- **Source Attribution**: Displays relevant text chunks used for answers
- **Real-time Chat Interface**: Interactive Q&A experience
- **Open Source**: Ready for community contributions

## Technology Stack

- **Backend**: FastAPI (Python)
- **AI Models**: 
  - Sentence Transformers (all-MiniLM-L6-v2) for embeddings
  - RoBERTa-base-squad2 for question answering
- **Vector Search**: FAISS for efficient similarity search
- **PDF Processing**: PyPDF2 for text extraction
- **Frontend**: Bootstrap 5, Vanilla JavaScript
- **Deployment**: Uvicorn ASGI server

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

3. Run the application:
```bash
python main.py
```

4. Open your browser and navigate to `http://localhost:8000`

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