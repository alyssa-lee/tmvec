from unittest.mock import MagicMock, patch

import pytest
import torch

from tmvec.embedding import ProtLM, ProtT5Encoder  # adjust import path


@pytest.fixture
def dummy_sequences():
    return ["MAKVLILGFAAGLVGATVA", "MKTAYIAKQRQISFVKSHFSRQ"]


def test_protlm_init_defaults():
    model = ProtLM("dummy_model", "dummy_tokenizer", "/tmp")
    assert model.model_path == "dummy_model"
    assert model.tokenizer_path == "dummy_tokenizer"
    assert model.device in [torch.device("cpu"), torch.device("cuda")]
    assert model.compile_model is False
    assert model.threads == 1


def test_protlm_forward_and_tokenize(dummy_sequences):
    model = ProtLM("dummy_model", "dummy_tokenizer", "/tmp")
    # Mock tokenizer and model
    model.tokenizer = MagicMock()
    model.model = MagicMock()
    model.model.eval = MagicMock()
    model.model.to = MagicMock()
    model.forward = MagicMock(return_value=torch.randn(2, 10, 128))

    # Mock tokenizer output
    model.tokenizer.batch_encode_plus.return_value = {
        "input_ids": torch.randint(0, 20, (2, 10)),
        "attention_mask": torch.ones((2, 10)),
    }

    inputs, embs = model.batch_embed(dummy_sequences)

    assert "input_ids" in inputs
    assert embs.shape[0] == 2


def test_remove_special_tokens():
    model = ProtLM("dummy_model", "dummy_tokenizer", "/tmp")
    embeddings = torch.randn(2, 10, 128)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    cleaned = model.remove_special_tokens(embeddings,
                                          attention_mask,
                                          shift_start=0,
                                          shift_end=-1)
    assert isinstance(cleaned, list)
    assert all(isinstance(x, torch.Tensor) for x in cleaned)


def test_get_sequence_embeddings_calls_batch_embed_and_cleanup(
        dummy_sequences):
    model = ProtLM("dummy_model", "dummy_tokenizer", "/tmp")
    # Mock batch_embed
    model.batch_embed = MagicMock(
        return_value=({
            "attention_mask": torch.ones((2, 5))
        }, torch.randn(2, 5, 128)))
    # Mock remove_special_tokens
    model.remove_special_tokens = MagicMock(
        return_value=[torch.randn(4, 128),
                      torch.randn(3, 128)])

    result = model.get_sequence_embeddings(dummy_sequences)
    assert len(result) == 2
    model.batch_embed.assert_called_once()
    model.remove_special_tokens.assert_called_once()


@patch("tmvec.embedding.T5Tokenizer")
@patch("tmvec.embedding.T5EncoderModel")
def test_prott5encoder_init_loads_model(mock_encoder, mock_tokenizer):
    mock_tokenizer.from_pretrained.return_value = MagicMock()
    mock_encoder.from_pretrained.return_value = MagicMock()

    encoder = ProtT5Encoder("dummy_model", "dummy_tokenizer", "/tmp")
    assert encoder.tokenizer is not None
    assert encoder.model is not None
    encoder.model.eval.assert_called_once()
    encoder.model.to.assert_called_once()
