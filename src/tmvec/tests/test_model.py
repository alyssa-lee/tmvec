import pytest

from tmvec import TMVEC_REPO
from tmvec.model import (TransformerEncoderModule,
                         TransformerEncoderModuleConfig)


@pytest.fixture
def config() -> TransformerEncoderModuleConfig:
    """Fixture for TransformerEncoderModuleConfig."""
    return TransformerEncoderModuleConfig()


@pytest.fixture
def model(config: TransformerEncoderModuleConfig) -> TransformerEncoderModule:
    """Fixture for TransformerEncoderModule."""
    return TransformerEncoderModule(config)


def test_from_hub():
    """Test loading model from the Hugging Face Hub."""
    model = TransformerEncoderModule.from_pretrained(TMVEC_REPO)
    assert isinstance(model, TransformerEncoderModule)


def test_local_model(model: TransformerEncoderModule):
    """Test local model initialization."""
    assert isinstance(model, TransformerEncoderModule)


def test_has_compile_ctx(model: TransformerEncoderModule):
    """Check if model has _compiler_ctx attribute (needed for training)."""
    assert hasattr(model, "_compiler_ctx")


def test_push_to_hub_callable(model: TransformerEncoderModule):
    """Check if model has push_to_hub method and it is callable."""
    assert hasattr(model, "push_to_hub")
    assert callable(model.push_to_hub)


def test_config_json_methods(config: TransformerEncoderModuleConfig, tmp_path):
    """Test to_json_file and from_json_file methods."""
    json_file = tmp_path / "config.json"

    # Check if methods exist and are callable
    assert hasattr(config, "to_json_file")
    assert hasattr(config, "from_json_file")
    assert callable(config.to_json_file)
    assert callable(config.from_json_file)

    # Test saving and loading
    config.to_json_file(str(json_file))
    loaded_config = TransformerEncoderModuleConfig.from_json_file(
        str(json_file))
    assert isinstance(loaded_config, TransformerEncoderModuleConfig)
