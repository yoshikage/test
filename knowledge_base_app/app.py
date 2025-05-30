import os
import numpy as np
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

from document_processor import (
    parse_pdf, parse_docx, parse_txt, parse_csv,
    chunk_text, get_embedding
)
from faiss_utils import create_faiss_index, add_vectors_to_index, load_index, save_index

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'csv'}
FAISS_INDEX_PATH = "knowledge_base.index" # Path to save/load FAISS index

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Determine embedding dimensionality (from 'all-MiniLM-L6-v2' model)
embedding_dim = 384

# Initialize or load FAISS index and document store
faiss_index = load_index(FAISS_INDEX_PATH)
if faiss_index is None:
    faiss_index = create_faiss_index(embedding_dim)
document_store = [] # This will be rebuilt if index is loaded, or built from scratch
# For simplicity, we are not saving/loading document_store alongside the index in this version.
# A more robust solution would save document_store (e.g., as JSON) and reload it with the index.
# If FAISS_INDEX_PATH exists and is loaded, document_store will be out of sync initially.
# This simple setup assumes document_store is built up along with the index in memory.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file_route(): # Renamed to avoid conflict with any potential 'upload_file' variable
    message = ""
    if request.method == 'POST':
        if 'file' not in request.files:
            message = 'No file part in the request.'
            return render_template('upload.html', message=message)
        
        file = request.files['file']
        if file.filename == '':
            message = 'No file selected.'
            return render_template('upload.html', message=message)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure the UPLOAD_FOLDER directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            file.save(file_path)
            message = f"File '{filename}' uploaded successfully. Processing..."

            try:
                file_ext = filename.rsplit('.', 1)[1].lower()
                
                if file_ext in ['pdf', 'docx', 'txt']:
                    raw_text = ""
                    if file_ext == 'pdf':
                        raw_text = parse_pdf(file_path)
                    elif file_ext == 'docx':
                        raw_text = parse_docx(file_path)
                    elif file_ext == 'txt':
                        raw_text = parse_txt(file_path)
                    
                    if not raw_text.strip():
                        message += " No text could be extracted from the document."
                        return render_template('upload.html', message=message)

                    chunks = chunk_text(raw_text, chunk_size=500, overlap=50) # Example chunk size/overlap
                    
                    if not chunks:
                        message += " Document was empty or too small to chunk."
                        return render_template('upload.html', message=message)

                    embeddings = get_embedding(chunks) # Returns a numpy array of embeddings
                    
                    if embeddings.ndim == 1: # Ensure it's 2D for faiss
                        embeddings = np.array([embeddings])
                    
                    add_vectors_to_index(faiss_index, embeddings.astype(np.float32))
                    
                    for i, chunk in enumerate(chunks):
                        # The ID in FAISS corresponds to the order of addition.
                        # document_store's index will be FAISS_ID.
                        document_store.append({'text': chunk, 'source': filename, 'type': 'chunk', 'id': faiss_index.ntotal - len(chunks) + i})
                    message += f" Processed and added {len(chunks)} text chunks to the knowledge base."

                elif file_ext == 'csv':
                    qa_pairs = parse_csv(file_path)
                    if not qa_pairs:
                        message += " No Q&A pairs found or CSV format incorrect."
                        return render_template('upload.html', message=message)

                    questions = [pair['question'] for pair in qa_pairs]
                    if not questions:
                        message += " No questions found in CSV Q&A pairs."
                        return render_template('upload.html', message=message)
                        
                    embeddings = get_embedding(questions)

                    if embeddings.ndim == 1: # Ensure it's 2D for faiss
                        embeddings = np.array([embeddings])

                    add_vectors_to_index(faiss_index, embeddings.astype(np.float32))
                    
                    for i, pair in enumerate(qa_pairs):
                        document_store.append({
                            'question': pair['question'], 
                            'answer': pair['answer'], 
                            'source': filename, 
                            'type': 'csv_qa',
                            'id': faiss_index.ntotal - len(qa_pairs) + i
                        })
                    message += f" Processed and added {len(qa_pairs)} Q&A pairs to the knowledge base."
                
                else:
                    message += " File type not supported for processing."

                # Persist the updated index (optional, but good for data retention)
                save_index(faiss_index, FAISS_INDEX_PATH)

            except Exception as e:
                message += f" An error occurred during processing: {str(e)}"
                # Consider more detailed logging for debugging
                print(f"Error processing {filename}: {e}")


            return render_template('upload.html', message=message)
        else:
            message = 'Allowed file types are txt, pdf, docx, csv.'
            return render_template('upload.html', message=message)

    return render_template('upload.html', message=message)

if __name__ == '__main__':
    # Create uploads folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
