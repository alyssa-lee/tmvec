import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from transformers import PreTrainedTokenizer, T5EncoderModel, T5Tokenizer


class ProtLM:
    """
    Base class for protein language models (LMs).
    Handles loading, tokenization, and embedding extraction.
    """
    def __init__(self,
                 model_path: Union[str, Path],
                 tokenizer_path: Union[str, Path],
                 cache_dir: Optional[Union[str, Path]] = None,
                 compile_model: bool = False,
                 threads: int = 1) -> None:
        """
        Initialize the ProtLM model.

        Args:
            model_path: Path to the pretrained model.
            tokenizer_path: Path to the tokenizer.
            cache_dir: Directory for cache (optional).
            compile_model: Whether to compile the model for faster inference.
            threads: Number of CPU threads to use.
        """
        self.model_path: Union[str, Path] = model_path
        self.tokenizer_path: Union[str, Path] = tokenizer_path
        self.cache_dir: Optional[Union[str, Path]] = cache_dir
        self.compile_model: bool = compile_model
        self.threads: int = threads

        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[torch.nn.Module] = None
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Set Torch CPU threads globally
        torch.set_num_threads(self.threads)

    def _prepare_model(self) -> None:
        """Prepare model for inference (set device and compilation if enabled)."""
        if self.model is None:
            raise ValueError("Model is not loaded yet.")
        self.model.eval()
        self.model.to(self.device)
        if self.compile_model:
            self.model = torch.compile(self.model,
                                       mode="reduce-overhead",
                                       dynamic=True)

    def forward(self, inputs: dict) -> Tensor:
        """
        Perform a forward pass through the model.

        Args:
            inputs: Tokenized inputs.

        Returns:
            Numpy array of embeddings.
        """
        if self.model is None:
            raise ValueError("Model is not initialized.")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.detach().cpu().numpy()

    def tokenize(self, sequences: List[str]) -> dict:
        """
        Tokenize input sequences.

        Args:
            sequences: List of protein sequences.

        Returns:
            Tokenized inputs as tensors.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        return self.tokenizer.batch_encode_plus(sequences,
                                                padding=True,
                                                return_tensors="pt",
                                                add_special_tokens=True)

    def batch_embed(self, sequences: List[str]) -> Tuple[dict, Tensor]:
        """
        Embed a batch of protein sequences.

        Args:
            sequences: List of protein sequences.

        Returns:
            Tuple of (tokenized inputs, embeddings).
        """
        inputs = self.tokenize(sequences)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        embeddings = self.forward(inputs)
        return inputs, embeddings

    def remove_special_tokens(self,
                              embeddings: Union[Tensor, List],
                              attention_mask: Tensor,
                              shift_start: int = 0,
                              shift_end: int = -1) -> List[Tensor]:
        """
        Remove special tokens from embeddings.

        Args:
            embeddings: Full sequence embeddings.
            attention_mask: Attention masks.
            shift_start: Tokens to trim from start.
            shift_end: Tokens to trim from end.

        Returns:
            List of cleaned embeddings.
        """
        cleaned: List[Tensor] = []
        for emb, mask in zip(embeddings, attention_mask):
            seq_len = int((mask == 1).sum())
            seq_emb = emb[shift_start:seq_len + shift_end]
            cleaned.append(seq_emb)
        return cleaned

    def get_sequence_embeddings(self, sequences: List[str]) -> List[Tensor]:
        """
        Get cleaned sequence embeddings for input sequences.

        Args:
            sequences: List of protein sequences.

        Returns:
            List of cleaned sequence embeddings.
        """
        inputs, embeddings = self.batch_embed(sequences)
        cleaned = self.remove_special_tokens(embeddings,
                                             inputs["attention_mask"])
        return cleaned


class ProtT5Encoder(ProtLM):
    """
    Protein T5 encoder implementation using Hugging Face Transformers.
    """
    def __init__(self,
                 model_path: Union[str, Path],
                 tokenizer_path: Union[str, Path],
                 cache_dir: Optional[Union[str, Path]] = None,
                 compile_model: bool = False,
                 local_files_only: bool = False,
                 threads: int = 1) -> None:
        """
        Initialize the ProtT5 encoder.

        Args:
            model_path: Path to T5 encoder model.
            tokenizer_path: Path to tokenizer.
            cache_dir: Directory for cache (optional).
            compile_model: Whether to compile the model for faster inference.
            local_files_only: Whether to only use local files for loading.
            threads: Number of CPU threads to use.
        """
        super().__init__(model_path, tokenizer_path, cache_dir, compile_model,
                         threads)

        # Load tokenizer and model
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
            tokenizer_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            legacy=True)
        self.model: T5EncoderModel = T5EncoderModel.from_pretrained(
            model_path, cache_dir=cache_dir, local_files_only=local_files_only)

        self._prepare_model()

    def tokenize(self, sequences):
        seqs = [
            " ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in sequences
        ]

        inp = super().tokenize(seqs)
        return inp


class ESMEncoder(ProtLM):
    def __init__(self,
                 model_path,
                 tokenizer_path,
                 cache_dir=None,
                 compile_model=False,
                 local_files_only=False,
                 threads=1):
        from transformers import EsmModel, EsmTokenizer

        super().__init__(model_path, tokenizer_path, cache_dir, compile_model,
                         threads)
        self.tokenizer = EsmTokenizer.from_pretrained(
            tokenizer_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            legacy=True)
        self.model = EsmModel.from_pretrained(
            model_path, cache_dir=cache_dir, local_files_only=local_files_only)

    def remove_special_tokens(self,
                              embeddings,
                              attention_mask,
                              shift_start=1,
                              shift_end=-1):
        embs = super().remove_special_tokens(embeddings, attention_mask,
                                             shift_start, shift_end)
        return embs


class Ankh(ProtT5Encoder):
    def tokenize(self, sequences):
        protein_sequences = [list(seq) for seq in sequences]
        inp = self.tokenizer.batch_encode_plus(
            protein_sequences,
            add_special_tokens=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        return inp
