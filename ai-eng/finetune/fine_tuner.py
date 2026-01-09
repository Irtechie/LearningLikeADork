# Purpose: Module for fine tuner.
# Created: 2026-01-07
# Author: MWR

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


@dataclass(frozen=True)
class LoRASettings:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-4
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
        parts = [f"[lora] step {step}"]
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


class FineTuner:
    def __init__(
        self,
        model_path: str,
        train_file: str,
        output_dir: str,
        lora: Optional[LoRASettings] = None,
        training: Optional[TrainingSettings] = None,
        use_4bit: bool = True,
    ) -> None:
        self.model_path = model_path
        self.train_file = Path(train_file)
        self.output_dir = Path(output_dir)
        self.lora = lora or LoRASettings()
        self.training = training or TrainingSettings()
        self.use_4bit = use_4bit

    def train(self) -> None:
        print("[lora] starting LoRA run")
        print(f"[lora] model_path: {self.model_path}")
        print(f"[lora] train_file: {self.train_file}")
        print(f"[lora] output_dir: {self.output_dir}")
        print(
            "[lora] settings: "
            f"r={self.lora.r} alpha={self.lora.alpha} dropout={self.lora.dropout}"
        )
        print(
            "[lora] training settings: "
            f"epochs={self.training.epochs} "
            f"batch_size={self.training.batch_size} "
            f"learning_rate={self.training.learning_rate} "
            f"max_seq_length={self.training.max_seq_length}"
        )

        if not self.train_file.exists():
            raise FileNotFoundError(f"[lora] train_file not found: {self.train_file}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quant_config = None
        if self.use_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("[lora] 4-bit training requires CUDA.")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        print("[lora] loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            quantization_config=quant_config,
        )
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id

        print("[lora] preparing LoRA adapters...")
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        if self.use_4bit:
            model = prepare_model_for_kbit_training(model)

        adapter_config = self.output_dir / "adapter_config.json"
        if adapter_config.exists():
            print("[lora] loading existing adapter from", self.output_dir)
            model = PeftModel.from_pretrained(
                model,
                str(self.output_dir),
                is_trainable=True,
            )
        else:
            lora_config = LoraConfig(
                r=self.lora.r,
                lora_alpha=self.lora.alpha,
                lora_dropout=self.lora.dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
            model = get_peft_model(model, lora_config)

        print("[lora] building dataset...")
        dataset = JsonlInstructionDataset(
            path=self.train_file,
            tokenizer=tokenizer,
            max_seq_length=self.training.max_seq_length,
        )
        print(f"[lora] dataset size: {len(dataset)}")

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
            fp16=torch.cuda.is_available(),
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=collator,
            callbacks=[TrainingProgressCallback()],
        )

        print("[lora] training...")
        resume_checkpoint = _latest_checkpoint(self.output_dir)
        if resume_checkpoint:
            print("[lora] resuming from checkpoint:", resume_checkpoint)
            trainer.train(resume_from_checkpoint=str(resume_checkpoint))
        else:
            print("[lora] no checkpoint found; starting fresh optimizer")
            trainer.train()

        print("[lora] saving adapter and tokenizer...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))
        print("[lora] done")
