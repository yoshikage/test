# Simple RAG Knowledge Base

## Overview

This application is a simple Retrieval Augmented Generation (RAG) system. It allows users to upload documents (PDF, DOCX, TXT, and CSV for Q&A pairs) which are then processed, indexed using FAISS, and made available for querying. Users can ask questions, and the system will retrieve relevant information from the uploaded documents to generate an answer using a pre-trained language model.

## Features

*   **Document Upload:** Supports uploading `.txt`, `.pdf`, `.docx`, and `.csv` files.
*   **Content Parsing:** Extracts text from various document formats.
    *   CSV files are specifically parsed for Question/Answer pairs (columns should be named 'Question'/'Q' and 'Answer'/'A', case-insensitive).
*   **Vector Indexing:** Uses FAISS for efficient similarity search on document embeddings.
*   **Sentence Embeddings:** Employs `sentence-transformers` (specifically, `all-MiniLM-L6-v2` model) to generate embeddings for text chunks and questions.
*   **Question Answering:** Utilizes a Hugging Face `transformers` pipeline (`distilbert-base-cased-distilled-squad`) to generate answers based on retrieved context.
*   **Web Interface:** Simple Flask-based UI for uploading documents and asking questions.
*   **Persistent Index:** The FAISS vector index is saved locally, allowing knowledge to persist across sessions.

## Setup and Installation

Follow these steps to set up and run the application locally.

1.  **Clone or Download the Code:**
    *   If using Git: `git clone <repository_url>`
    *   Alternatively, download the source code as a ZIP file and extract it.
    *   Navigate into the `knowledge_base_app` directory.

2.  **Python Version:**
    *   Ensure you have Python 3.8 or newer installed. You can check with `python --version`.

3.  **Create a Virtual Environment:**
    *   It's highly recommended to use a virtual environment to manage dependencies.
    *   In your terminal, from the `knowledge_base_app` directory, run:
        ```bash
        python -m venv venv
        ```

4.  **Activate the Virtual Environment:**
    *   **Windows:**
        ```bash
        venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    *   Your terminal prompt should change to indicate that the virtual environment is active (e.g., `(venv) ...`).

5.  **Install Dependencies:**
    *   With the virtual environment active, install the required packages:
        ```bash
        pip install -r requirements.txt
        ```
    *   This command will download and install all necessary libraries, including Flask, FAISS, PyPDF2, python-docx, sentence-transformers, numpy, and transformers/torch. The first time you run the app, the sentence transformer and QA models might take some time to download.

## Running the Application

1.  **Start the Flask Server:**
    *   Ensure your virtual environment is active and you are in the `knowledge_base_app` directory.
    *   Run the application using:
        ```bash
        python app.py
        ```
    *   You should see output indicating the Flask development server is running, typically on `http://127.0.0.1:5000/`.

2.  **Access the Application:**
    *   Open your web browser and go to: `http://127.0.0.1:5000/`

## How to Use

1.  **Upload Documents:**
    *   Navigate to the "Upload Documents" page (this is the default page).
    *   Click "Choose file", select a supported document (`.txt`, `.pdf`, `.docx`, `.csv`), and click "Upload".
    *   The system will process the file, generate embeddings, and add them to the knowledge base.
    *   For CSV files, ensure they have columns named 'Question' (or 'Q') and 'Answer' (or 'A'). Each row will be treated as a direct Q&A pair.
    *   You will receive feedback messages on the success or failure of the upload and processing.

2.  **Ask Questions:**
    *   Navigate to the "Query Knowledge Base" page using the link in the navigation bar.
    *   Type your question into the input field and click "Ask".
    *   The system will search the knowledge base for relevant information and display an answer.
    *   If the most relevant information comes from a CSV Q&A pair, that answer will be shown directly. Otherwise, a generative model will formulate an answer based on text chunks from other document types.
    *   You can expand the "Show Retrieved Context" section to see the text snippets used to generate the answer.

## File Paths and Data Persistence

*   **Uploaded Files:** Uploaded documents are stored in the `knowledge_base_app/uploads/` directory.
*   **FAISS Index:** The FAISS vector index, which stores the embeddings of your documents, is saved to `knowledge_base_app/knowledge_base.index`. This file allows the knowledge base to persist even if you restart the application. If this file doesn't exist when the app starts, a new one will be created.
*   **Document Metadata:** Information linking the vectors in FAISS back to the original text chunks or Q&A pairs (the `document_store` in `app.py`) is currently stored in-memory. This means if you restart the application, while the FAISS index itself is reloaded, the `document_store` is rebuilt as new data is processed. For a more robust persistence of this metadata, it would need to be saved to and loaded from a file (e.g., a JSON file) alongside the FAISS index.

---
This README provides a comprehensive guide for users to set up and use the application.I have verified `knowledge_base_app/requirements.txt` and it is correct.
I have created `knowledge_base_app/README.md` with all the specified sections, ensuring the `FAISS_INDEX_PATH` mentioned matches the one in `app.py` (`knowledge_base.index`).

Both parts of the subtask are complete.
