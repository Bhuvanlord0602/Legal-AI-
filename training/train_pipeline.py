"""Training pipeline for legal translation model using HuggingFace Transformers."""

import csv
from pathlib import Path
from typing import List, Tuple
import json

from core.config_manager import ModelConfig, get_default_config


class LegalTranslationCorpus:
    """Handler for bilingual legal corpus."""

    def __init__(self, corpus_path: str):
        self.corpus_path = Path(corpus_path)
        self.en_texts = []
        self.kn_texts = []
        self.pairs = []

    def load(self) -> Tuple[List[str], List[str]]:
        """Load bilingual corpus from CSV."""
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        with open(self.corpus_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                en = row.get("english", "").strip()
                kn = row.get("kannada", "").strip()
                if en and kn:
                    self.en_texts.append(en)
                    self.kn_texts.append(kn)
                    self.pairs.append((en, kn))

        return self.en_texts, self.kn_texts

    def split(
        self, train_ratio: float = 0.9, val_ratio: float = 0.05
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Split corpus into train/val/test."""
        total = len(self.pairs)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        train = self.pairs[:train_size]
        val = self.pairs[train_size : train_size + val_size]
        test = self.pairs[train_size + val_size :]

        return train, val, test

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "total_pairs": len(self.pairs),
            "language_pair": "en-kan",
            "english_texts": self.en_texts,
            "kannada_texts": self.kn_texts,
        }

    def save_splits(self, output_dir: str = "data/splits"):
        """Save train/val/test splits."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train, val, test = self.split()

        def save_split(name, data):
            filepath = output_path / f"{name}.csv"
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["english", "kannada"])
                writer.writerows(data)
            return filepath

        train_path = save_split("train", train)
        val_path = save_split("validation", val)
        test_path = save_split("test", test)

        print(f"✓ Train split: {len(train)} pairs -> {train_path}")
        print(f"✓ Validation split: {len(val)} pairs -> {val_path}")
        print(f"✓ Test split: {len(test)} pairs -> {test_path}")

        return train_path, val_path, test_path


class TrainingPipeline:
    """Training pipeline for legal translation model."""

    def __init__(self, config: ModelConfig = None):
        self.config = config or get_default_config()
        self.corpus = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self):
        """Prepare training data from corpus."""
        print("Loading corpus...")
        self.corpus = LegalTranslationCorpus(self.config.corpus_path)
        self.corpus.load()

        print(f"✓ Loaded {len(self.corpus.pairs)} translation pairs")

        print("\nSplitting corpus...")
        self.train_data, self.val_data, self.test_data = self.corpus.split(
            train_ratio=self.config.train_test_split,
            val_ratio=self.config.validation_split,
        )

        print(f"✓ Train: {len(self.train_data)} pairs")
        print(f"✓ Validation: {len(self.val_data)} pairs")
        print(f"✓ Test: {len(self.test_data)} pairs")

        return self.train_data, self.val_data, self.test_data

    def save_corpus_info(self, output_path: str = "data/corpus_info.json"):
        """Save corpus information."""
        if self.corpus is None:
            raise ValueError("Corpus not loaded. Call prepare_data() first.")

        info = {
            "corpus_info": self.corpus.to_dict(),
            "training_config": self.config.to_dict(),
            "split_sizes": {
                "train": len(self.train_data) if self.train_data else 0,
                "validation": len(self.val_data) if self.val_data else 0,
                "test": len(self.test_data) if self.test_data else 0,
            },
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        print(f"✓ Corpus info saved to {output_path}")

    def train(self):
        """
        Train the neural translation model.
        (Placeholder for actual training loop)
        """
        if self.train_data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        print("\n" + "=" * 70)
        print("Starting Model Training")
        print("=" * 70)
        print(f"Model: {self.config.model_name}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Device: {self.config.device}")
        print("=" * 70)

        # TODO: Implement actual training using transformers library
        print("\nNote: Full training implementation requires:")
        print("  - transformers library with trainer")
        print("  - GPU support (CUDA)")
        print("  - Pre-trained model downloading")
        print("\nTo implement full training:")
        print("  1. Install: pip install transformers datasets")
        print("  2. Use HuggingFace Trainer for fine-tuning")
        print("  3. Save model checkpoints")

    def evaluate(self):
        """Evaluate the model on test set."""
        print("\n" + "=" * 70)
        print("Model Evaluation")
        print("=" * 70)

        if self.test_data is None:
            raise ValueError("Test data not available. Call prepare_data() first.")

        print(f"Test set size: {len(self.test_data)} pairs")
        print("\nSample test cases:")
        for i, (en, kn) in enumerate(self.test_data[:3], 1):
            print(f"\n{i}. EN: {en}")
            print(f"   KN: {kn}")

        # TODO: Compute BLEU/ROUGE scores on test set


def main():
    """Example usage of training pipeline."""
    print("Legal Translation Training Pipeline")
    print("=" * 70)

    # Load config
    config = get_default_config()
    print("\nConfig loaded:")
    print(f"  Corpus: {config.corpus_path}")
    print(f"  Model: {config.model_name}")

    # Initialize pipeline
    pipeline = TrainingPipeline(config)

    # Prepare data
    print("\n" + "=" * 70)
    print("Data Preparation")
    print("=" * 70)
    pipeline.prepare_data()

    # Save corpus info
    pipeline.save_corpus_info()

    # Save data splits
    print("\nSaving data splits...")
    pipeline.corpus.save_splits()

    # Save config
    print("\nSaving configuration...")
    config.save("config/model_config.json")
    print("✓ Config saved to config/model_config.json")

    # Show training instructions
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Install transformers: pip install transformers datasets")
    print("2. Fine-tune model: python train_model.py")
    print("3. Evaluate: python evaluate_model.py")


if __name__ == "__main__":
    main()
