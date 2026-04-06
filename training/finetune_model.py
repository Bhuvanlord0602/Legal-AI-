"""Fine-tuning script for legal translation model using HuggingFace Transformers."""

import csv
from pathlib import Path
import json
import sys
from typing import Optional

# Optional HuggingFace imports
try:
    from transformers import (
        MarianMTModel,
        MarianTokenizer,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
    )
    from datasets import Dataset, DatasetDict
    import torch

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers/datasets not installed. Install with:")
    print("  pip install transformers datasets torch")


from core.config_manager import ModelConfig, get_default_config
from tokenization.tokenizer_sp import LegalTokenizer
from evaluation.metrics_eval import evaluate_translation_pair


class TranslationDataset:
    """Load and prepare translation data for fine-tuning."""

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.data = []

    def load(self) -> list:
        """Load bilingual pairs from CSV."""
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                en = row.get("english", "").strip()
                kn = row.get("kannada", "").strip()
                if en and kn:
                    self.data.append({"en": en, "kn": kn})

        print(f"✓ Loaded {len(self.data)} translation pairs from {self.csv_path}")
        return self.data

    def to_hf_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset format."""
        if not self.data:
            raise ValueError("No data loaded. Call load() first.")

        return Dataset.from_dict({
            "source": [d["en"] for d in self.data],
            "target": [d["kn"] for d in self.data],
        })

    def create_splits(self, train_ratio: float = 0.8) -> DatasetDict:
        """Create train/test splits as HuggingFace DatasetDict."""
        if not self.data:
            raise ValueError("No data loaded. Call load() first.")

        hf_dataset = self.to_hf_dataset()
        split_ds = hf_dataset.train_test_split(test_size=1 - train_ratio)

        return split_ds


class LegalTranslationFineTuner:
    """Fine-tune a pre-trained translation model on legal corpus."""

    def __init__(self, config: ModelConfig = None):
        if not HF_AVAILABLE:
            print("ERROR: HuggingFace transformers not installed.")
            print("Install with: pip install transformers datasets torch")
            sys.exit(1)

        self.config = config or get_default_config()
        self.model = None
        self.tokenizer = None
        self.dataset = None

    def load_model_and_tokenizer(self):
        """Load pre-trained model and tokenizer."""
        print(f"\nLoading model: {self.config.model_name}")

        try:
            self.model = MarianMTModel.from_pretrained(self.config.model_name)
            self.tokenizer = MarianTokenizer.from_pretrained(self.config.model_name)
            print("✓ Model and tokenizer loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("Using fallback: Helsinki-NLP/Opus-MT-en-mul")
            self.config.model_name = "Helsinki-NLP/Opus-MT-en-mul"
            self.model = MarianMTModel.from_pretrained(self.config.model_name)
            self.tokenizer = MarianTokenizer.from_pretrained(self.config.model_name)

    def prepare_data(self):
        """Prepare translation data."""
        print(f"\nPreparing data from: {self.config.corpus_path}")

        dataset_loader = TranslationDataset(self.config.corpus_path)
        dataset_loader.load()
        self.dataset = dataset_loader.create_splits(
            train_ratio=self.config.train_test_split
        )

        print(f"✓ Train: {len(self.dataset['train'])} pairs")
        print(f"✓ Test: {len(self.dataset['test'])} pairs")

    def preprocess_data(self):
        """Preprocess data for model training."""
        def tokenize_fn(batch):
            inputs = self.tokenizer(
                batch["source"],
                max_length=self.config.max_seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            labels = self.tokenizer(
                batch["target"],
                max_length=self.config.max_seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": labels["input_ids"],
            }

        print("\nTokenizing dataset...")
        self.dataset = self.dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=self.config.batch_size,
            remove_columns=["source", "target"],
        )
        print("✓ Tokenization complete")

    def fine_tune(self):
        """Fine-tune the model."""
        print("\n" + "=" * 70)
        print("Starting Fine-tuning")
        print("=" * 70)

        # Create output directory
        output_dir = Path(self.config.model_save_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=float(self.config.learning_rate),
            save_steps=1000,
            eval_steps=500,
            logging_steps=100,
            save_total_limit=3,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        # Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            processing_class=self.tokenizer,
            data_collator=data_collator,
        )

        print(f"Device: {self.config.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Epochs: {self.config.num_epochs}")
        print("=" * 70)

        # Train
        trainer.train()

        # Save model
        model_path = output_dir / "final_model"
        self.model.save_pretrained(str(model_path))
        self.tokenizer.save_pretrained(str(model_path))
        print(f"\n✓ Model saved to {model_path}")

        # Save config
        config_path = output_dir / "training_config.json"
        self.config.save(str(config_path))
        print(f"✓ Config saved to {config_path}")

        return trainer

    def evaluate_sample(self):
        """Evaluate on a sample translation."""
        print("\n" + "=" * 70)
        print("Sample Translation Evaluation")
        print("=" * 70)

        if self.dataset is None:
            raise ValueError("Dataset not prepared. Call prepare_data() first.")

        # Get first sample from test set
        sample = self.dataset["test"][0]
        input_ids = torch.tensor([sample["input_ids"]])

        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=128)

        translated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        source_text = self.tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        reference_text = self.tokenizer.decode(sample["labels"], skip_special_tokens=True)

        print(f"\nTest sample:")
        print(f"Input: {source_text}")
        print(f"Reference: {reference_text}")
        print(f"Translation: {translated[0]}")

        # Compute BLEU
        try:
            metrics = evaluate_translation_pair(reference_text, translated[0])
            print(f"BLEU: {metrics['bleu']:.4f}")
            print(f"ROUGE-L F1: {metrics['rouge_f_score']:.4f}")
        except Exception as e:
            print(f"Evaluation: {e}")


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("Legal Translation Model Fine-tuning")
    print("=" * 70)

    # Load config
    config = get_default_config()

    # Initialize fine-tuner
    finetuner = LegalTranslationFineTuner(config)

    # Load model
    finetuner.load_model_and_tokenizer()

    # Prepare data
    finetuner.prepare_data()

    # Preprocess
    finetuner.preprocess_data()

    # Fine-tune
    finetuner.fine_tune()

    # Evaluate
    finetuner.evaluate_sample()

    print("\n" + "=" * 70)
    print("Fine-tuning Complete!")
    print(f"Model saved to: {config.model_save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
