# Paper
TM-Vec: template modeling vectors for fast homology detection and alignment: https://www.biorxiv.org/content/10.1101/2022.07.25.501437v1

[Embed sequences with TM-vec](https://colab.research.google.com/github/tymor22/tm-vec/blob/master/google_colabs/Embed_sequences_using_TM_Vec.ipynb)

# Installation

First create a conda environment with python>=3.9 installed.  If you are using cpu, use

```conda create -n tmvec python -c pytorch```

Once your conda enviroment is installed, install `tmvec` via:

```pip install git+https://github.com/valentynbez/tmvec.git```

If you are using a GPU, you may need to reinstall the gpu version of pytorch.
See the [pytorch](https://pytorch.org/) webpage for more details.

# Run TM-Vec from the command line

If the computer is connected to the internet, then all the models will be downloaded automatically. If the computer is not connected to the internet, then the models will need to be downloaded manually, and the paths to the models will need to be specified.

```bash
tmvec build-db \
    --input-fasta small_embed.fasta \
    --output db_test/small_fasta
```
To query a sequences against a database use:
```bash
tmvec search \
    --query small_embed.fasta \
    --database db_test/small_fasta.npz \
    --output db_test/result.tsv
```

We suggest to make first runs on a smaller batches with internet connection. After first run models will be downloaded to `cache` directory, and afterwards can be manually inputted into CLI in case computing nodes do not have access to the internet.

# Embed sequences with pLMs (experimental)
In order to run this command, run `pip install -e git+https://github.com/user/project.git#egg=tmvec[embed]`.

**Available models:**
- ProtT5
- ESM
- Ankh

```bash
tmvec embed --input-fasta small_embed.fasta \
    --model-type esm \
    --model-path facebook/esm2_t6_8M_UR50D \
    --cache-dir cache \
    --output-file small-embed.h5py
```
Parameter `model-path` can be both `huggingface` repo or a path in a local filesystem. If repo is provided, the model will be downloaded to `cache-dir`.

# CPU/GPU difference

For CPU execution, we utilize ONNX protein language models, which give a slight speedup. For GPU, we use a standard `ProtT5` with `torch.compile` directive, which seems to be faster than ONNX.
