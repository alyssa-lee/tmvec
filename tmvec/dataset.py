import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Define collate function for batches
def collate_fn(batch, pad_id=1, max_len=1024):
    S1_tensors = [t['Embed_sequence_1'] for t in batch]
    S2_tensors = [t['Embed_sequence_2'] for t in batch]
    combined_tensors = S1_tensors + S2_tensors

    padded_tensors = torch.nn.utils.rnn.pad_sequence(combined_tensors,
                                                     padding_value=pad_id,
                                                     batch_first=True)
    padded_tensors = padded_tensors[:, :max_len, :]

    batch_length = len(batch)
    S1_tensors_padded = padded_tensors[:batch_length]
    S2_tensors_padded = padded_tensors[batch_length:]

    pad_labels_seq1 = torch.zeros(S1_tensors_padded.shape[0:2]).type(
        torch.BoolTensor)
    pad_labels_seq2 = torch.zeros(S2_tensors_padded.shape[0:2]).type(
        torch.BoolTensor)
    pad_labels_seq1[S1_tensors_padded[:, :, 0] == pad_id] = True  #pad_id
    pad_labels_seq2[S2_tensors_padded[:, :, 0] == pad_id] = True  #pad_id

    tm_scores = torch.tensor([t['tm_score'] for t in batch])

    return (S1_tensors_padded, S2_tensors_padded, pad_labels_seq1,
            pad_labels_seq2, tm_scores)


class tm_score_embeds_dataset(Dataset):
    """TM score dataset."""
    def __init__(self, hdf_file, tm_pairs):
        """
        Args:
            hdf_file (string): Path to the pickle file with the embeddings and tm_scores.
            tm_pairs (string): Path to the TSV file with pairs of proteins with TM scores.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #with gzip.open(pickle_file, "rb") as f:
        #    self.tm_score_emb_data = pickle.load(f)
        # load pairs
        self.tm_score_emb_data = pd.read_csv(tm_pairs,
                                             sep="\t",
                                             header=None,
                                             low_memory=False)
        self.hdf_file = hdf_file

    def __len__(self):
        return len(self.tm_score_emb_data)

    def __getitem__(self, idx):
        # open HDF file
        with h5py.File(self.hdf_file, 'r') as f:
            idx1, idx2, tm_score = self.tm_score_emb_data.loc[idx].values
            seq1 = f[idx1]["emb"][:].astype(np.float32)
            seq2 = f[idx2]["emb"][:].astype(np.float32)
            sample = {
                'Embed_sequence_1': torch.Tensor(seq1),
                'Embed_sequence_2': torch.Tensor(seq2),
                'tm_score': tm_score
            }
        return sample


#Function to construct datasets to be fed to dataloader
def construct_datasets(hdf_file,
                       tm_pairs,
                       train_prop=.9,
                       val_prop=.05,
                       test_prop=.05):
    total_samples = len(
        tm_score_embeds_dataset(hdf_file=hdf_file, tm_pairs=tm_pairs))
    sampleable_values = np.arange(total_samples)

    train_n_to_sample = int(len(sampleable_values) * train_prop)
    val_n_to_sample = int(len(sampleable_values) * val_prop)
    test_n_to_sample = int(len(sampleable_values) * test_prop)

    train_indices = np.random.choice(sampleable_values,
                                     train_n_to_sample,
                                     replace=False)
    sampleable_values = sampleable_values[
        ~np.isin(sampleable_values, train_indices)]
    val_indices = np.random.choice(sampleable_values,
                                   val_n_to_sample,
                                   replace=False)
    sampleable_values = sampleable_values[~np.
                                          isin(sampleable_values, val_indices)]
    test_indices = np.random.choice(sampleable_values,
                                    test_n_to_sample,
                                    replace=False)

    # Make train, test, and validation datasets using torch subset
    train_ds = torch.utils.data.Subset(
        tm_score_embeds_dataset(hdf_file=hdf_file, tm_pairs=tm_pairs),
        train_indices)
    val_ds = torch.utils.data.Subset(
        tm_score_embeds_dataset(hdf_file=hdf_file, tm_pairs=tm_pairs),
        val_indices)
    test_ds = torch.utils.data.Subset(
        tm_score_embeds_dataset(hdf_file=hdf_file, tm_pairs=tm_pairs),
        test_indices)

    return (train_ds, val_ds, test_ds)
