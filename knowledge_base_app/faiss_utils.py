import faiss
import numpy as np
import os

def create_faiss_index(d: int):
  """
  Creates a new FAISS index.

  Args:
    d: The dimensionality of the vectors.

  Returns:
    A faiss.IndexFlatL2 index.
  """
  index = faiss.IndexFlatL2(d)
  return index

def add_vectors_to_index(index: faiss.Index, vectors: np.ndarray):
  """
  Adds vectors to the FAISS index.

  Args:
    index: The FAISS index instance.
    vectors: A 2D numpy array of vectors to add.
             FAISS expects float32 vectors.
  """
  if vectors.ndim == 1: # Handle single vector case
      vectors = np.array([vectors], dtype=np.float32)
  elif vectors.ndim == 2:
      vectors = vectors.astype(np.float32)
  else:
      raise ValueError("Input vectors must be 1D or 2D numpy arrays.")
  
  index.add(vectors)

def save_index(index: faiss.Index, path: str):
  """
  Saves the FAISS index to a file.

  Args:
    index: The FAISS index to save.
    path: The file path to save the index to.
  """
  try:
    faiss.write_index(index, path)
    print(f"FAISS index saved to {path}")
  except Exception as e:
    print(f"Error saving FAISS index to {path}: {e}")

def load_index(path: str) -> faiss.Index | None:
  """
  Loads a FAISS index from a file.

  Args:
    path: The file path to load the index from.

  Returns:
    The loaded FAISS index, or None if loading fails or file doesn't exist.
  """
  try:
    if os.path.exists(path):
      index = faiss.read_index(path)
      print(f"FAISS index loaded from {path}")
      return index
    else:
      print(f"FAISS index file not found at {path}. A new index will be created if needed.")
      return None
  except Exception as e:
    print(f"Error loading FAISS index from {path}: {e}")
    return None

# Example usage (optional, can be removed or commented out)
if __name__ == '__main__':
  vector_dimensionality = 128
  my_index = create_faiss_index(vector_dimensionality)
  print(f"Successfully created a FAISS index for {vector_dimensionality}-dimensional vectors.")
  
  # Example: Add some dummy vectors
  dummy_vectors = np.random.rand(100, vector_dimensionality).astype('float32')
  add_vectors_to_index(my_index, dummy_vectors)
  print(f"Total vectors in index after adding dummies: {my_index.ntotal}")

  # Example: Save and load index
  index_file_path = "my_faiss_index.index"
  save_index(my_index, index_file_path)
  loaded_index = load_index(index_file_path)
  if loaded_index:
    print(f"Loaded index has {loaded_index.ntotal} vectors.")
  
  # Clean up dummy file
  if os.path.exists(index_file_path):
    os.remove(index_file_path)
