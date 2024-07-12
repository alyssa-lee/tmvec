import re
import warnings
from types import MethodType

# silence future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class ProtLM:
    def __init__(self,
                 model_path,
                 tokenizer_path,
                 cache_dir,
                 backend="torch",
                 compile_model=False,
                 threads=1):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.cache_dir = cache_dir
        self.backend = backend
        self.compile_model = compile_model
        self.threads = threads
        self.tokenizer = None
        self.model = None
        self.device = None
        self.forward_pass = None

        # ignore compile model for non-torch backends
        if backend != "torch":
            self.compile_model = False

    def init_torch(self):
        import torch
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(self.device)
        if self.compile_model:
            self.model = torch.compile(self.model,
                                       mode="reduce-overhead",
                                       dynamic=True)
        # set torch threads
        torch.set_num_threads(self.threads)

        def forward_pass(self, inp):
            with torch.no_grad():
                out = self.model(**inp)

            out = out.last_hidden_state.detach().cpu().numpy()

            return out

        self.forward_pass = MethodType(forward_pass, self)

    def init_onnx(self):
        import onnxruntime as rt
        import torch
        from optimum.onnxruntime import ORTModel

        # TODO: troubleshot ONNX run on GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        sess_options = rt.SessionOptions()
        sess_options.intra_op_num_threads = self.threads
        sess_options.inter_op_num_threads = self.threads
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.model = ORTModel.from_pretrained(self.model_path,
                                              model_save_dir=self.cache_dir,
                                              session_options=sess_options,
                                              providers=[
                                                  "TensorrtExecutionProvider",
                                                  "CUDAExecutionProvider",
                                                  "CPUExecutionProvider"
                                              ]).model

        def forward_pass(self, inp):
            onnx_input = {
                self.model.get_inputs()[0].name:
                inp["input_ids"].cpu().numpy(),
                self.model.get_inputs()[1].name:
                inp["attention_mask"].cpu().numpy()
            }
            out = self.model.run(["last_hidden_state"], onnx_input)[0]

            return out

        self.forward_pass = MethodType(forward_pass, self)

    def tokenize(self, sequences):
        inp = self.tokenizer.batch_encode_plus(sequences,
                                               padding=True,
                                               return_tensors="pt",
                                               add_special_tokens=True)
        return inp

    def batch_embed(self, sequences):

        inp = self.tokenize(sequences).to(self.device)
        embs = self.forward_pass(inp)

        return inp, embs

    def remove_special_tokens(self,
                              embeddings,
                              attention_mask,
                              shift_start=0,
                              shift_end=-1):
        clean_embeddings = []
        embeddings = list(embeddings)
        for emb, mask in zip(embeddings, attention_mask):
            seq_len = (mask == 1).sum()
            seq_emb = emb[shift_start:seq_len + shift_end]
            clean_embeddings.append(seq_emb)

        return clean_embeddings

    def get_sequence_embeddings(self, sequences):
        inp, embs = self.batch_embed(sequences)
        embs = self.remove_special_tokens(embs, inp["attention_mask"])
        return embs


class ProtT5Encoder(ProtLM):
    def __init__(self,
                 model_path,
                 tokenizer_path,
                 cache_dir=None,
                 backend="torch",
                 compile_model=False,
                 local_files_only=False,
                 threads=1):
        from transformers import T5Tokenizer
        super().__init__(model_path, tokenizer_path, cache_dir, compile_model,
                         threads)
        self.tokenizer = T5Tokenizer.from_pretrained(
            tokenizer_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            legacy=True)
        if backend == "torch":
            from transformers import T5EncoderModel
            self.model = T5EncoderModel.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                local_files_only=local_files_only)
            self.init_torch()
        elif backend == "onnx":
            self.init_onnx()

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
                 backend="torch",
                 compile_model=False,
                 local_files_only=False,
                 threads=1):
        from transformers import EsmModel, EsmTokenizer

        super().__init__(model_path, tokenizer_path, cache_dir, compile_model,
                         threads)
        if backend == "torch":
            self.tokenizer = EsmTokenizer.from_pretrained(
                tokenizer_path,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                legacy=True)
            self.model = EsmModel.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                local_files_only=local_files_only)

            self.init_torch()
        else:
            raise ValueError("Backend is not supported for ESM model.")

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
        inp = self.tokenizer.batch_encode_plus(
            sequences,
            add_special_tokens=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        return inp
