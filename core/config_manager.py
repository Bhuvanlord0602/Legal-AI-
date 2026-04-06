"""Configuration manager for legal translation model."""

import json
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Model configuration for training and inference."""

    # Model architecture
    model_name: str = "Helsinki-NLP/Opus-MT-en-mul"  # Fallback to MarianMT
    source_lang: str = "en"
    target_lang: str = "kan"
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 2e-4
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Data parameters
    max_seq_length: int = 128
    train_test_split: float = 0.9
    validation_split: float = 0.1
    
    # Tokenizer
    tokenizer_type: str = "sentencepiece"  # "sentencepiece" or "huggingface"
    vocab_size: int = 32000
    
    # Paths
    corpus_path: str = "data/legal_corpus.csv"
    model_save_path: str = "models/legal_translation_model"
    tokenizer_path: str = "models/tokenizer"
    
    # Hardware
    use_gpu: bool = True
    device: str = "cuda"  # "cuda" or "cpu"
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    log_steps: int = 100

    def save(self, filepath: str):
        """Save config to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """Load config from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)


def get_default_config() -> ModelConfig:
    """Get default configuration."""
    return ModelConfig()


def save_config(config: ModelConfig, path: str = "config/model_config.json"):
    """Save model configuration."""
    config.save(path)


def load_config(path: str = "config/model_config.json") -> ModelConfig:
    """Load model configuration."""
    if Path(path).exists():
        return ModelConfig.load(path)
    return get_default_config()
