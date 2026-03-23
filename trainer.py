import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import os
import json
import logging
from tqdm import tqdm
from typing import Optional, Dict, List
import wandb
from datetime import datetime

logger = logging.getLogger("Trainer")

class TextDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        if os.path.isfile(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                text = f.read()
            self._process_text(text)
        elif os.path.isdir(data_path):
            for filename in os.listdir(data_path):
                filepath = os.path.join(data_path, filename)
                if os.path.isfile(filepath) and filename.endswith((".txt", ".json", ".jsonl")):
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                    self._process_text(text)

    def _process_text(self, text: str):
        tokens = self.tokenizer.encode(text)
        for i in range(0, len(tokens) - self.max_length, self.max_length // 2):
            self.examples.append(tokens[i:i + self.max_length])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(tokens, dtype=torch.long)
        }

class SimpleTokenizer:
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2

    def encode(self, text: str) -> List[int]:
        # Simple character-level tokenization
        tokens = [self.bos_token_id]
        for char in text:
            tokens.append(ord(char) % self.vocab_size)
        tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        chars = []
        for token in tokens:
            if token in [self.pad_token_id, self.eos_token_id, self.bos_token_id]:
                continue
            try:
                chars.append(chr(token % 128))
            except:
                chars.append("?")
        return "".join(chars)

class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        use_wandb: bool = False
    ):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.tokenizer = SimpleTokenizer(config.get("vocab_size", 50257))
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler("cuda") if config.get("fp16", True) and torch.cuda.is_available() else None

        self.global_step = 0
        self.best_loss = float("inf")
        self.training_history = []

        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="rag2-training", config=config)

    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        no_decay = ["bias", "LayerNorm.weight", "RMSNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.get("weight_decay", 0.01)
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.get("learning_rate", 5e-5),
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Setup schedulers
        warmup_steps = self.config.get("warmup_steps", 1000)
        max_steps = self.config.get("max_steps", 100000)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_steps - warmup_steps,
            eta_min=self.config.get("learning_rate", 5e-5) * 0.1
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_epochs: int = 1
    ):
        """Train the model"""
        if self.optimizer is None:
            self.setup_optimizer()

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.get("batch_size", 8),
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        logger.info(f"Starting training on {self.device}")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
        logger.info(f"Training batches: {len(train_dataloader)}")

        accumulation_steps = self.config.get("gradient_accumulation_steps", 4)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        logging_steps = self.config.get("logging_steps", 100)
        save_steps = self.config.get("save_steps", 5000)

        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass with mixed precision
                if self.scaler is not None:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(**batch)
                        loss = outputs["loss"] / accumulation_steps
                    self.scaler.scale(loss).backward()
                else:
                    outputs = self.model(**batch)
                    loss = outputs["loss"] / accumulation_steps
                    loss.backward()

                # Optimizer step
                if (step + 1) % accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                epoch_loss += loss.item() * accumulation_steps

                # Logging
                if self.global_step % logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "step": self.global_step
                    })

                    log_entry = {
                        "step": self.global_step,
                        "loss": avg_loss,
                        "lr": lr,
                        "epoch": epoch
                    }
                    self.training_history.append(log_entry)

                    if self.use_wandb:
                        wandb.log(log_entry)

                # Save checkpoint
                if self.global_step % save_steps == 0:
                    self.save_checkpoint(epoch)

            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

            # Evaluate
            if eval_dataset is not None:
                eval_loss = self.evaluate(eval_dataset)
                logger.info(f"Evaluation loss: {eval_loss:.4f}")

                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self.save_checkpoint(epoch, is_best=True)

            self.save_checkpoint(epoch)

        logger.info("Training completed!")
        return self.training_history

    def evaluate(self, eval_dataset: Dataset) -> float:
        """Evaluate the model"""
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.config.get("batch_size", 8),
            shuffle=False
        )

        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs["loss"].item()

        self.model.train()
        return total_loss / len(eval_dataloader)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_loss": self.best_loss,
            "config": self.config
        }

        if is_best:
            path = os.path.join(checkpoint_dir, "best_model.pt")
        else:
            path = os.path.join(checkpoint_dir, f"checkpoint-{self.global_step}.pt")

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

        # Save training history
        history_path = os.path.join(checkpoint_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer and checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]

        logger.info(f"Loaded checkpoint from {checkpoint_path} at step {self.global_step}")
        return checkpoint["epoch"]