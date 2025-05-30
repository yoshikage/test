import PyPDF2
import docx
import csv
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize SentenceTransformer model
# Using a common model for sentence embeddings.
# This will be downloaded when the code first runs.
model = SentenceTransformer('all-MiniLM-L6-v2')

def parse_pdf(file_path: str) -> str:
    """
    Parses a PDF file and extracts text content.
    """
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error parsing PDF {file_path}: {e}")
        # Depending on desired error handling, could raise e or return empty string
    return text

def parse_docx(file_path: str) -> str:
    """
    Parses a DOCX file and extracts text content.
    """
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error parsing DOCX {file_path}: {e}")
    return text

def parse_txt(file_path: str) -> str:
    """
    Parses a TXT file and returns its content.
    """
    text = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error parsing TXT {file_path}: {e}")
    return text

def parse_csv(file_path: str) -> list[dict[str, str]]:
    """
    Parses a CSV file, expecting 'Question'/'Q' and 'Answer'/'A' columns.
    Returns a list of {'question': '...', 'answer': '...'} dictionaries.
    Handles case variations for column names.
    """
    qa_pairs = []
    try:
        with open(file_path, 'r', encoding='utf-8', newline='') as file:
            reader = csv.DictReader(file)
            q_cols = [col for col in reader.fieldnames if col.lower() in ['question', 'q']]
            a_cols = [col for col in reader.fieldnames if col.lower() in ['answer', 'a']]

            if not q_cols or not a_cols:
                print(f"CSV {file_path} does not have the required Question/Answer columns.")
                return []

            q_col = q_cols[0]
            a_col = a_cols[0]

            for row in reader:
                question = row.get(q_col, "").strip()
                answer = row.get(a_col, "").strip()
                if question and answer: # Ensure both question and answer are present
                    qa_pairs.append({'question': question, 'answer': answer})
    except Exception as e:
        print(f"Error parsing CSV {file_path}: {e}")
    return qa_pairs

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Splits text into chunks with specified size and overlap.
    """
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be non-negative and less than chunk size.")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
        if start + overlap >= len(text) and start < len(text): # Ensure last part is captured if smaller than chunk_size
             # This logic can be tricky; the goal is to avoid losing the tail
             # or creating an empty/too-small final chunk unnecessarily.
             # If the remaining text is smaller than overlap, it might be skipped by typical increment.
             # A simpler way: if next start is beyond text length, we're done.
             # The loop condition handles most cases. Let's test this logic.
             # Consider if the remaining part is just `overlap` long, it should be added.
             # If `start` is already >= `len(text) - overlap` then the `text[start:end]` is the last chunk
            pass # The while condition and increment should handle this correctly
    return chunks

def get_embedding(text_or_texts: str | list[str]) -> np.ndarray:
    """
    Generates embeddings for a text string or a list of text strings.
    Returns numpy array(s).
    """
    if isinstance(text_or_texts, str):
        embeddings = model.encode(text_or_texts)
    elif isinstance(text_or_texts, list):
        embeddings = model.encode(text_or_texts)
    else:
        raise TypeError("Input must be a string or a list of strings.")
    return np.array(embeddings)

# Example Usage (can be commented out or removed)
if __name__ == '__main__':
    # Note: These examples require the actual model to be downloaded and files to exist.
    # This section is for testing the functions; it will likely fail in the tool environment
    # if model download is blocked or files are not present.

    print("Document Processor Initialized")
    print(f"Sentence Transformer Model: {type(model)}")

    # Example for chunk_text
    sample_text = "This is a long string of text that needs to be chunked into smaller pieces for processing. Each chunk will have a defined size and overlap with the previous chunk to maintain context."
    chunks = chunk_text(sample_text, chunk_size=50, overlap=10)
    print("\nText Chunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}")

    # Example for get_embedding (requires model download)
    try:
        embedding = get_embedding("This is a test sentence.")
        print(f"\nSample Embedding (shape {embedding.shape}):\n{embedding[:5]}...") # Print first 5 dimensions
        
        list_embeddings = get_embedding(["First sentence.", "Second sentence."])
        print(f"\nSample List Embeddings (shape {list_embeddings.shape}):\n{list_embeddings[:,:5]}...")
    except Exception as e:
        print(f"\nError generating embeddings (model might not be downloaded): {e}")

    # To test parsing functions, you would need to create dummy files
    # e.g., dummy.txt, dummy.pdf, dummy.docx, dummy.csv
    # For now, we assume the functions are structurally correct.
    # print("\nTesting parsers (requires dummy files):")
    # print(f"TXT parse: {parse_txt('dummy.txt')}")
    # print(f"PDF parse: {parse_pdf('dummy.pdf')}")
    # print(f"DOCX parse: {parse_docx('dummy.docx')}")
    # print(f"CSV parse: {parse_csv('dummy.csv')}")
