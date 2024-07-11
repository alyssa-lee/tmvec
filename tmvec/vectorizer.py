import warnings
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from tmvec import PLM_ONNX, PLM_REPO, TMVEC_REPO
from tmvec.embedding import ProtT5Encoder
from tmvec.model import (TransformerEncoderModule,
                         TransformerEncoderModuleConfig)


class TMVec:
    def __init__(self,
                 tmvec_model: TransformerEncoderModule = None,
                 cache_dir: str = None,
                 protlm_path: str = None,
                 protlm_tokenizer_path: str = None,
                 local_files_only: bool = False):

        self.tmvec_model = tmvec_model
        self.protlm_path = protlm_path
        self.protlm_tokenizer_path = protlm_tokenizer_path
        self.cache_dir = cache_dir
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.backend = None
        self.compile_model = None
        # load from the repo
        if not self.tmvec_model:
            self.tmvec_config = TransformerEncoderModuleConfig.from_pretrained(
                TMVEC_REPO)
            self.tmvec_model = TransformerEncoderModule.from_pretrained(
                TMVEC_REPO, config=self.tmvec_config)

        # move model to device
        self.tmvec_model.to(self.device)
        # switch model to inference mode
        self.tmvec_model.eval()

        if self.protlm_tokenizer_path is None:
            self.protlm_tokenizer_path = PLM_REPO
        if str(self.device) == "cuda":
            if self.protlm_path is None:
                self.protlm_path = PLM_REPO
            self.backend = "torch"
            self.compile_model = True
        elif str(self.device) == "cpu":
            if self.protlm_path is None:
                self.protlm_path = PLM_ONNX
            self.backend = "onnx"
            self.compile_model = False
        self.embedder = ProtT5Encoder(self.protlm_path,
                                      self.protlm_tokenizer_path,
                                      self.cache_dir,
                                      local_files_only=local_files_only,
                                      backend=self.backend,
                                      compile_model=self.compile_model,
                                      threads=1)

    def embed_single_protein(self, sequence: List[str]):
        embedding = self.embedder.get_sequence_embeddings(sequence)[0]
        return embedding

    def embedding_to_vector(self, embedding: np.ndarray):
        prot_embedding = torch.tensor(embedding).unsqueeze(0).to(self.device)
        padding = torch.zeros(prot_embedding.shape[0:2]).type(
            torch.BoolTensor).to(self.device)
        tm_vec_embedding = self.tmvec_model(prot_embedding,
                                            src_mask=None,
                                            src_key_padding_mask=padding)

        return tm_vec_embedding.cpu().detach().numpy()

    def vectorize_proteins(self, sequences: List[str]):
        embed_all_sequences = []
        for seq in tqdm(sequences,
                        desc="Vectorizing sequences",
                        total=len(sequences),
                        miniters=len(sequences) // 100):
            prottrans_embedding = self.embed_single_protein([seq])
            embedded_sequence = self.embedding_to_vector(prottrans_embedding)
            embed_all_sequences.append(embedded_sequence)

        return np.concatenate(embed_all_sequences, axis=0)

    def delete_vectorizer(self):
        """
        Removes TMVec model to free up memory
        for alignment model.
        """

        del self.tmvec_model
        torch.cuda.empty_cache()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config_path: str, **kwargs):
        config = TransformerEncoderModuleConfig.from_json_file(config_path)
        model = TransformerEncoderModule.load_from_checkpoint(checkpoint_path,
                                                              config=config)
        return cls(model, **kwargs)

    @classmethod
    def from_pretrained(cls, model_folder: str, **kwargs):
        try:
            model = TransformerEncoderModule.from_pretrained(model_folder)
        except TypeError:
            warnings.warn(
                "Model not found locally. Loading from HuggingFace Hub."
                "This will require an internet connection.")
            model = None
        return cls(model, **kwargs)
