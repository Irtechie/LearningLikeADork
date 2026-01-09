# Purpose: Fine-tuned adapter that uses the same interface with LoRA weights.
# Created: 2026-01-06
# Author: MWR

"""Fine-tuned adapter that uses the same interface with LoRA weights."""

from pathlib import Path

from adapters.hf.local_hf import LocalHFAdapter


class FinetunedAdapter(LocalHFAdapter):
    def __init__(
        self,
        base_model_path: str,
        lora_path: str,
        max_new_tokens: int = 128,
    ) -> None:
        lora_dir = Path(lora_path)
        if not lora_dir.exists():
            raise FileNotFoundError(f"[adapter] lora_path not found: {lora_dir}")

        print("[adapter] loading base model for finetuned adapter...")
        super().__init__(model_path=base_model_path, max_new_tokens=max_new_tokens)

        print(f"[adapter] loading LoRA adapter: {lora_dir}")
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(
            self.model,
            str(lora_dir),
            is_trainable=False,
        )
        self.model.eval()
