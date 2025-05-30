import unittest
import os
import tempfile
import numpy as np
import faiss # Import faiss to check types

from knowledge_base_app.faiss_utils import (
    create_faiss_index,
    add_vectors_to_index,
    save_index,
    load_index
)

class TestFaissUtils(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.index_path = os.path.join(self.test_dir.name, "test_faiss.index")
        self.vector_dim = 128 # Example dimensionality

    def tearDown(self):
        self.test_dir.cleanup()

    def test_create_faiss_index(self):
        index = create_faiss_index(self.vector_dim)
        self.assertIsNotNone(index)
        self.assertIsInstance(index, faiss.Index)
        self.assertEqual(index.d, self.vector_dim)
        self.assertEqual(index.ntotal, 0)

    def test_add_vectors_to_index_single_vector(self):
        index = create_faiss_index(self.vector_dim)
        vector = np.random.rand(self.vector_dim).astype(np.float32)
        add_vectors_to_index(index, vector)
        self.assertEqual(index.ntotal, 1)

    def test_add_vectors_to_index_multiple_vectors(self):
        index = create_faiss_index(self.vector_dim)
        num_vectors = 10
        vectors = np.random.rand(num_vectors, self.vector_dim).astype(np.float32)
        add_vectors_to_index(index, vectors)
        self.assertEqual(index.ntotal, num_vectors)

    def test_add_vectors_to_index_wrong_dim(self):
        index = create_faiss_index(self.vector_dim)
        # Vector with incorrect dimensionality
        vector_wrong_dim = np.random.rand(self.vector_dim + 1).astype(np.float32)
        with self.assertRaises(Exception): # FAISS typically raises RuntimeError for dimension mismatch
            add_vectors_to_index(index, vector_wrong_dim)

    def test_add_vectors_to_index_not_float32(self):
        index = create_faiss_index(self.vector_dim)
        vector_int = np.random.randint(0, 100, size=self.vector_dim).astype(np.int32)
        # add_vectors_to_index converts to float32, so this should not fail
        # but we can check if it handles the conversion
        add_vectors_to_index(index, vector_int) 
        self.assertEqual(index.ntotal, 1)
        # To truly test this, we might need to inspect the index or ensure no error,
        # as our function explicitly casts to float32.

    def test_save_and_load_index(self):
        index = create_faiss_index(self.vector_dim)
        num_vectors = 5
        vectors = np.random.rand(num_vectors, self.vector_dim).astype(np.float32)
        add_vectors_to_index(index, vectors)

        save_index(index, self.index_path)
        self.assertTrue(os.path.exists(self.index_path))

        loaded_index = load_index(self.index_path)
        self.assertIsNotNone(loaded_index)
        self.assertIsInstance(loaded_index, faiss.Index)
        self.assertEqual(loaded_index.d, self.vector_dim)
        self.assertEqual(loaded_index.ntotal, num_vectors)

    def test_load_index_non_existent(self):
        non_existent_path = os.path.join(self.test_dir.name, "non_existent.index")
        loaded_index = load_index(non_existent_path)
        self.assertIsNone(loaded_index)

    def test_add_vectors_to_index_3d_array(self):
        index = create_faiss_index(self.vector_dim)
        vectors_3d = np.random.rand(2, 5, self.vector_dim).astype(np.float32) # Invalid shape
        with self.assertRaises(ValueError):
            add_vectors_to_index(index, vectors_3d)

if __name__ == '__main__':
    unittest.main()
