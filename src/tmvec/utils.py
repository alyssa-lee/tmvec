#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import numpy as np
import pandas as pd
import torch
from pysam import FastxFile


@dataclass
class SessionTree:
    """
    Creates a model for session dir.
    root/
        checkpoints/
        logs/
        params.json
        dataset_indices.pkl
    """
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)

        self.root.mkdir(exist_ok=True, parents=True)
        self.checkpoints.mkdir(exist_ok=True, parents=True)
        self.logs.mkdir(exist_ok=True, parents=True)

    @property
    def params(self):
        return self.root / "params.json"

    @property
    def checkpoints(self):
        return self.root / "checkpoints"

    @property
    def indices(self):
        return self.root / "dataset_indices.pkl"

    @property
    def logs(self):
        return self.root / "logs/"

    @property
    def last_ckpt(self):
        return self.checkpoints / "last.ckpt"

    @property
    def best_ckpt(self):
        if (self.checkpoints / "best.ckpt").exists():
            self.checkpoints / "best.ckpt"
        return self.last_ckpt

    def dump_indices(self, indices):
        with open(self.indices, 'wb') as pk:
            pickle.dump(indices, pk)


def load_fasta_as_dict(fasta_file: str,
                       sort: bool = True,
                       max_len: int = None) -> Dict[str, str]:
    """
    Load FASTA file as dict of headers to sequences

    Args:
        fasta_file (str): Path to FASTA file. Can be compressed.
        sorted (bool): Sort sequences by length.
        max_len (int): Maximum length of sequences to include.

    Returns:sq
        Dict[str, str]: Dictionary of FASTA entries sorted by length.
    """

    seqs_dict = {}
    with FastxFile(fasta_file) as f:
        for i, entry in enumerate(f):
            seqs_dict[entry.name] = entry.sequence

    if sort:
        seqs_dict = dict(sorted(seqs_dict.items(), key=lambda x: len(x[1])))

    if max_len:
        seqs_dict = {k: v for k, v in seqs_dict.items() if len(v) <= max_len}

    return seqs_dict


def create_batched_sequence_dataset(
    sequences: Dict[str, str],
    max_tokens_per_batch: int = 1024
) -> Generator[Tuple[List[str], List[str]], None, None]:

    batch_headers, batch_sequences, num_tokens = [], [], 0
    for header, seq in sequences.items():
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)

    yield batch_headers, batch_sequences


# Generate random proteins
def generate_proteins(n_prots):

    PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
    np.random.seed(42)
    proteins = []
    for _ in range(n_prots):
        prot = "".join(
            np.random.choice(list(PROTEIN_ALPHABET),
                             size=np.random.randint(20, 100)))
        proteins.append(prot)
    return proteins


# Predict the TM-score for a pair of proteins (inputs are TM-Vec embeddings)
def cosine_similarity(output_seq1: torch.Tensor,
                      output_seq2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the cosine similarity between two protein embeddings

    Args:
        output_seq1 (torch.Tensor): Protein embedding for sequence 1
        output_seq2 (torch.Tensor): Protein embedding for sequence 2

    Returns:
        torch.Tensor: Cosine similarity between the two embeddings
    """

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    dist_seq = cos(output_seq1, output_seq2)

    return dist_seq


def save_results(values, near_ids, headers, output_file):
    """
    Outputs the results based on the specified format.

    Args:
        values (numpy.ndarray): An array containing the values (e.g., scores) for each query.
        near_ids (numpy.ndarray): A 2D array containing the metadata for the nearest neighbors.
        headers (list or numpy.ndarray): A list or array containing the metadata headers.
        output_format (str): The desired output format (e.g., 'tabular').
        output_file (str): The file path to write the output.
    """
    save_tabular_format(values, near_ids, headers, output_file)


def save_tabular_format(values, near_ids, headers, output_file):
    """
    Outputs the results in a tabular format.

    Args:
        values (numpy.ndarray): An array containing the values (e.g., scores) for each query.
        near_ids (numpy.ndarray): A 2D array containing the metadata for the nearest neighbors.
        headers (list or numpy.ndarray): A list or array containing the metadata headers.
        output_file (str): The file path to write the output.
    """
    nids = pd.DataFrame(near_ids, index=headers)
    nids.index.name = 'query_id'
    nids = pd.melt(nids.reset_index(),
                   id_vars='query_id',
                   var_name='rank',
                   value_name='database_id')
    nids['rank'] = nids['rank'] + 1

    tms = pd.DataFrame(values, index=headers)
    tms = pd.melt(tms, var_name='query_id', value_name='tm-score')
    nids = pd.concat((nids, tms[['tm-score']]), axis=1)
    nids = nids.sort_values(['query_id', 'rank'])
    nids.to_csv(output_file, sep='\t', index=None)


def save_embeddings(seqs, queries, output_embeddings_file, output_fmt='npz'):
    """
    Outputs the embeddings to a file.

    Args:
        queries (numpy.ndarray): An array containing the query embeddings.
        output_embeddings_file (str): The file path to write the embeddings.
    """
    if output_embeddings_file is not None:
        if output_fmt == 'npz':
            np.save(output_embeddings_file, queries)
        elif output_fmt == 'skbio':
            try:
                import skbio
                from skbio.embedding import ProteinVector
            except ImportError as e:
                raise ValueError(
                    "The 'skbio' output format requires the scikit-bio library to be installed. "
                    "Please install tmvec[scikit-bio] and try again.") from e

            cls = ProteinVector
            pvs = (cls(q, s) for s, q in zip(seqs, queries))
            skbio.io.write(pvs,
                           format='embed',
                           into=str(output_embeddings_file))

        else:
            raise ValueError(f'Output format {output_fmt} not supported.')


def format_ids(str1, str2):
    """
    Formats two strings by padding the shorter string with spaces to match the length of the longer string.

    Args:
        str1 (str): The first string.
        str2 (str): The second string.

    Returns:
        tuple: A tuple containing the formatted strings (str1, str2).
    """
    max_length = max(len(str1), len(str2))
    str1 = str1.ljust(max_length + 1)
    str2 = str2.ljust(max_length + 1)
    return str1, str2


if __name__ == '__main__':
    pass
