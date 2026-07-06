# Document-Based AI Assistant (Backend)

This is the backend for the RAG-powered Document AI Assistant. It provides a robust, scalable, and fast API for ingesting documents (PDF, DOCX, TXT), extracting their content (including OCR for images), vectorizing them, and answering user queries using an LLM.

## Tech Stack Overview

- **Core Framework**: [FastAPI](https://fastapi.tiangolo.com/) for high-performance, asynchronous endpoints.
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) for storing document embeddings and performing lightning-fast similarity (RAG) searches.
- **LLM & Embeddings**: Google Gemini API
  - `models/gemini-embedding-001` to generate vector embeddings.
  - `gemini-2.0-flash` for reasoning, answering questions, and performing Vision OCR on scanned documents.
- **Streaming**: `sse-starlette` to stream tokens back to the frontend in real-time via Server-Sent Events (SSE).
- **Security & Limiting**: `slowapi` to prevent abuse via IP-based rate limiting.

### Document Extraction Libraries
Different libraries are used based on the file type to ensure maximum data quality:
- **`pdfplumber`**: For parsing native, machine-readable text and layout from standard PDFs.
- **`pypdfium2`**: An ultra-fast PDF renderer used to turn PDF pages into images. This is triggered as a fallback when a PDF is scanned or image-only.
- **`python-docx`**: For natively parsing Microsoft Word paragraphs and grouping them into synthetic "pages" for chunking.
- **`langchain-text-splitters`**: Using `RecursiveCharacterTextSplitter` and `MarkdownTextSplitter` to semantically chunk documents without breaking sentences.

---

## How Chunking & Retrieval Works (RAG Pipeline)

The core feature of this backend is Retrieval-Augmented Generation (RAG). Here is exactly how a document goes from an uploaded file to an intelligent answer:

### 1. Extraction & Ingestion
When a file hits the `/upload` endpoint, it is saved to a temporary directory. The backend determines the correct parsing strategy based on the file extension:
- **TXT**: Read directly as a single long page.
- **DOCX**: `python-docx` extracts paragraphs sequentially. Because Word documents don't have hard "page" boundaries in the same way PDFs do, the backend groups paragraphs into ~3000-character synthetic pages so they can be referenced and cited easily.
- **PDF**: Processed page-by-page via `pdfplumber`. 
  - **Vision OCR Fallback**: If a page yields fewer than 30 characters, the backend assumes it is a scanned document or image. It invokes `pypdfium2` to render the page at 2x resolution, passes that image to `gemini-2.0-flash`, and uses the Vision API to perfectly transcribe the text structure.

### 2. Semantic Chunking
Once the text is extracted, it must be broken down into digestible pieces for the LLM to search against. 
- The backend uses LangChain's `MarkdownTextSplitter` (which inherits from `RecursiveCharacterTextSplitter`).
- Documents are chunked into **1000-character blocks** with a **200-character overlap**. 
- The overlap ensures that if a sentence or thought crosses a chunk boundary, context isn't lost. 
- Crucially, every chunk retains metadata: `document_id`, `filename`, and the original `source_page_number`.

### 3. Vectorization & Storage
- Each chunk is passed to `get_embeddings` which calls the `models/gemini-embedding-001` API.
- The text chunk, its embedding vector, and its metadata are stored in **ChromaDB**. 
- By default, Chroma runs ephemerally in-memory, but can be configured to connect to a cloud tenant via the `.env` file.

### 4. Retrieval & Querying
When a user asks a question via the `/chat` endpoint:
- The user's query is converted into an embedding using the exact same Gemini embedding model.
- The backend queries ChromaDB, performing a cosine similarity search to find the top 5 chunks that semantically match the user's question.
- The backend extracts these 5 chunks and injects them into the `System Prompt` as exact reference material, alongside instructions demanding that the LLM cite its sources (e.g., `[1]`, `[2]`).

### 5. Generation & Streaming
- The constructed prompt is sent to `gemini-2.0-flash`.
- As the model generates the answer, `sse-starlette` captures the output and streams it back to the client token-by-token over an HTTP connection.
- After the text finishes generating, a final JSON payload containing the exact citations (including the original chunk text and page numbers) is sent so the frontend can display clickable source panels.

---

## Environment Setup

Copy `.env.example` to `.env` and fill in your variables:

```ini
GEMINI_API_KEY="your-api-key"
ALLOWED_ORIGINS="http://localhost:3000,http://192.168.56.1:3000"
```

### Running Locally

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate # Mac/Linux

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
