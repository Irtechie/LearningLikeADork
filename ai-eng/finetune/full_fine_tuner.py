# Purpose: Module for full fine tuner.
# Created: 2026-01-08
# Author: MWR

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 5e-5
    max_seq_length: int = 1024
    gradient_accumulation_steps: int = 1
    save_total_limit: int = 1


class TrainingProgressCallback(TrainerCallback):
    def __init__(self, log_every_steps: int = 1) -> None:
        self._log_every_steps = max(1, log_every_steps)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        if step % self._log_every_steps != 0:
            return
        parts = [f"[full-finetune] step {step}"]
        epoch = logs.get("epoch")
        if epoch is not None:
            parts.append(f"epoch={epoch:.2f}")
        loss = logs.get("loss")
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        lr = logs.get("learning_rate")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        print(" ".join(parts))


class JsonlInstructionDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_seq_length: int):
        self._records = list(self._load_records(path))
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int):
        record = self._records[idx]
        prompt = self._format_prompt(record)
        response = (record.get("output") or "").strip()
        full_text = f"{prompt}{response}"

        prompt_ids = self._tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self._max_seq_length,
        )["input_ids"]
        full = self._tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self._max_seq_length,
        )

        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    @staticmethod
    def _format_prompt(record: dict) -> str:
        instruction = (record.get("instruction") or "").strip()
        input_text = (record.get("input") or "").strip()
        if input_text:
            return f"{instruction}\n\nInput:\n{input_text}\n\nAnswer:\n"
        return f"{instruction}\n\nAnswer:\n"

    @staticmethod
    def _load_records(path: Path) -> Iterable[dict]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    checkpoints = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None

    def _step(p: Path) -> int:
        suffix = p.name.split("checkpoint-")[-1]
        return int(suffix) if suffix.isdigit() else -1

    checkpoints.sort(
        key=lambda p: (_step(p), p.stat().st_mtime),
        reverse=True,
    )
    return checkpoints[0]


class FullFineTuner:
    def __init__(
        self,
        model_path: str,
        train_file: str,
        output_dir: str,
        training: Optional[TrainingSettings] = None,
    ) -> None:
        self.model_path = model_path
        self.train_file = Path(train_file)
        self.output_dir = Path(output_dir)
        self.training = training or TrainingSettings()

    def train(self) -> None:
        print("[full-finetune] starting full fine-tune run")
        print(f"[full-finetune] model_path: {self.model_path}")
        print(f"[full-finetune] train_file: {self.train_file}")
        print(f"[full-finetune] output_dir: {self.output_dir}")
        print(
            "[full-finetune] training settings: "
            f"epochs={self.training.epochs} "
            f"batch_size={self.training.batch_size} "
            f"learning_rate={self.training.learning_rate} "
            f"max_seq_length={self.training.max_seq_length}"
        )

        if not self.train_file.exists():
            raise FileNotFoundError(
                f"[full-finetune] train_file not found: {self.train_file}"
            )

        config = AutoConfig.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        quant_config = getattr(config, "quantization_config", None)
        if isinstance(quant_config, dict):
            if quant_config.get("load_in_4bit") or quant_config.get("_load_in_4bit"):
                raise RuntimeError(
                    "[full-finetune] 4-bit model detected; use a full-precision base model."
                )
            if quant_config.get("load_in_8bit") or quant_config.get("_load_in_8bit"):
                raise RuntimeError(
                    "[full-finetune] 8-bit model detected; use a full-precision base model."
                )
        print("[full-finetune] model check: full-precision required (no 4-bit/8-bit).")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("[full-finetune] loading model...")
        use_cuda = torch.cuda.is_available()
        use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
        device = "cuda" if use_cuda else "cpu"
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        precision_label = "bf16" if use_bf16 else "fp32"
        print(f"[full-finetune] precision: {precision_label} on {device}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            device_map=device,
            dtype=dtype,
        )
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id

        print("[full-finetune] building dataset...")
        dataset = JsonlInstructionDataset(
            path=self.train_file,
            tokenizer=tokenizer,
            max_seq_length=self.training.max_seq_length,
        )
        print(f"[full-finetune] dataset size: {len(dataset)}")

        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        )

        args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.training.epochs,
            per_device_train_batch_size=self.training.batch_size,
            gradient_accumulation_steps=self.training.gradient_accumulation_steps,
            learning_rate=self.training.learning_rate,
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=self.training.save_total_limit,
            fp16=False,
            bf16=use_bf16,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=collator,
            callbacks=[TrainingProgressCallback()],
        )

        print("[full-finetune] training...")
        resume_checkpoint = _latest_checkpoint(self.output_dir)
        if resume_checkpoint:
            print("[full-finetune] resuming from checkpoint:", resume_checkpoint)
            trainer.train(resume_from_checkpoint=str(resume_checkpoint))
        else:
            print("[full-finetune] no checkpoint found; starting fresh optimizer")
            trainer.train()

        print("[full-finetune] saving full model and tokenizer...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))
        pt_path = self.output_dir / "model.pt"
        print(f"[full-finetune] saving torch state dict: {pt_path}")
        torch.save(trainer.model.state_dict(), pt_path)
        print("[full-finetune] done")
