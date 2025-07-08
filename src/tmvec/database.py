import os
from typing import List

import faiss
import numpy as np


def xstr(s):
    """
    Convert None to an empty string.
    """
    return '' if s is None else str(s)


def strx(s):
    """
    Convert an empty string to None.
    """
    return None if s == '' else s


def save_database(headers: List[str],
                  embeddings: np.ndarray,
                  fasta_filepath: str,
                  tm_vec_weights: str = None,
                  protrans_model_path: str = None,
                  output_path: str = None) -> None:
    """
    Save protein sequence data, embeddings, and related files to a compressed NumPy archive.

    Args:
        headers (list): A list of headers for the protein sequences.
        embeddings (np.ndarray): A NumPy array containing the embeddings for the protein sequences.
        fasta_filepath (str): The file path to the input FASTA file.
        tm_vec_weights (str): The file path to the TM-vec weights file.
        protrans_model_path (str): The file path to the ProTrans model.
        output_path (str): The file path for the output compressed NumPy archive.

    Returns:
        None
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    input_fasta = np.array(os.path.abspath(fasta_filepath))
    try:
        tm_vec_weights = np.array(os.path.abspath(tm_vec_weights))
    except TypeError:
        tm_vec_weights = np.array(xstr(tm_vec_weights))

    try:
        protrans_model_path = np.array(os.path.abspath(protrans_model_path))
    except TypeError:
        protrans_model_path = np.array(xstr(protrans_model_path))

    np.savez_compressed(output_path,
                        headers=headers,
                        embeddings=embeddings,
                        input_fasta=input_fasta,
                        tm_vec_weights=tm_vec_weights,
                        protrans_model_path=protrans_model_path)


def build_vector_index(vector_db: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a vector index using the Faiss library for efficient similarity search.

    Args:
        vector_db (np.ndarray): A NumPy array containing the vectors to be indexed.

    Returns:
        faiss.IndexFlatIP: A Faiss index object for the input vectors.
    """
    d = vector_db.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(vector_db)
    index.add(vector_db)
    return index


def load_database(path: str) -> tuple:
    """
    Load a compressed NumPy archive containing protein sequence data, embeddings, and related files.

    Args:
        path (str): The file path to the compressed NumPy archive.

    Returns:
        tuple: A tuple containing the following elements:
            - embeddings (np.ndarray): A NumPy array containing the embeddings for the protein sequences.
            - index (faiss.IndexFlatIP): A Faiss index object for the embeddings.
            - headers (np.ndarray): Headers for the protein sequences.
            - input_fasta (str): Absolute file path to the input FASTA file.
            - tm_vec_weights (str): Absolute file path to the TM-vec weights file.
            - protrans_model_path (str): Absolute file path to the ProTrans model.
    """
    database = np.load(path)
    embeddings = database['embeddings']
    headers = database['headers']

    input_fasta = database['input_fasta'].item()
    tm_vec_weights = database['tm_vec_weights'].item()
    protrans_model_path = database['protrans_model_path'].item()

    tm_vec_weights = strx(tm_vec_weights)
    protrans_model_path = strx(protrans_model_path)

    # Build an indexed database
    index = build_vector_index(embeddings)

    return embeddings, index, headers, input_fasta, tm_vec_weights, protrans_model_path


def query(index, queries: np.ndarray, k: int = 10):
    """
    Performs a similarity search on the given index using the provided queries.

    Args:
        index (faiss.Index): The index to search against.
        queries (numpy.ndarray): The queries to search for.
        k (int, optional): The number of nearest neighbors to return. Defaults to 10.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - values (numpy.ndarray): The distances to the nearest neighbors.
            - indexes (numpy.ndarray): The indexes of the nearest neighbors.
    """
    faiss.normalize_L2(queries)
    values, indexes = index.search(queries, k)
    return values, indexes


def get_metadata_for_neighbors(indexes, headers):
    """
    Retrieves the metadata for the nearest neighbors based on their indexes.

    Args:
        indexes (numpy.ndarray): An array of indexes representing the nearest neighbors.
        target_headers (list or numpy.ndarray): A list or array containing the metadata headers.

    Returns:
        numpy.ndarray: A 2D array containing the metadata for the nearest neighbors.
    """
    near_ids = []
    for i in range(indexes.shape[0]):
        meta = headers[indexes[i]]
        near_ids.append(list(meta))
    return np.array(near_ids)
