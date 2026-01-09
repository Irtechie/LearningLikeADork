# Purpose: Local model adapter.
# Created: 2026-01-06
# Author: MWR

import time
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from unsloth import FastLanguageModel
    _UNSLOTH_AVAILABLE = True
except Exception:
    FastLanguageModel = None
    _UNSLOTH_AVAILABLE = False


def _is_unsloth_model(config, model_path: str) -> bool:
    return bool(
        getattr(config, "unsloth_fixed", False)
        or getattr(config, "unsloth_version", None)
        or "unsloth" in str(model_path).lower()
    )


class LocalHFAdapter:
    def __init__(self, model_path: str, max_new_tokens: int = 128):
        print("[adapter] loading tokenizer...")
        config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        use_cuda = torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32

        # For 4-bit quantized models, must use explicit device placement
        # "auto" miscalculates memory requirements for quantized models
        device_map = "cuda" if use_cuda else "cpu"
        max_memory = None
    
        quant_config = None

        wants_4bit = False
        config_quant = getattr(config, "quantization_config", None)
        if isinstance(config_quant, dict):
            wants_4bit = bool(
                config_quant.get("load_in_4bit")
                or config_quant.get("_load_in_4bit")
            )
        elif config_quant is not None:
            wants_4bit = bool(getattr(config_quant, "load_in_4bit", False))

        if wants_4bit:
            if not use_cuda:
                raise RuntimeError(
                    "[adapter] 4-bit model requires CUDA; CPU load is not supported."
                )
            try:
                import bitsandbytes  # noqa: F401
            except Exception as exc:
                raise RuntimeError(
                    "[adapter] bitsandbytes is required for 4-bit loading. "
                    "Install bitsandbytes or use a non-quantized model."
                ) from exc

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        if _UNSLOTH_AVAILABLE and _is_unsloth_model(config, model_path):
            print("[adapter] loading model (unsloth)...")
            max_seq_length = getattr(config, "max_position_embeddings", None)
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=wants_4bit,
                device_map=device_map,
                local_files_only=True,
                trust_remote_code=True,
            )
            FastLanguageModel.for_inference(self.model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
            )
            print("[adapter] loading model (explicit CUDA)...")
            # When using 4-bit quantization, the model loads directly to GPU
            # low_cpu_mem_usage can actually cause CPU OOM during meta->device transfer
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
                max_memory=max_memory,
                low_cpu_mem_usage=False,  # Disabled to avoid CPU OOM with quantized models
                local_files_only=True,
                trust_remote_code=True,
                quantization_config=quant_config,
            )

        self.model.eval()
        # Avoid warnings when using greedy decoding (do_sample=False).
        gen_config = getattr(self.model, "generation_config", None)
        if gen_config is not None:
            gen_config.top_p = None
            gen_config.top_k = None
            gen_config.temperature = None
        self.max_new_tokens = max_new_tokens
        print("[adapter] model loaded")

    @torch.no_grad()
    def generate(self, user_input: str) -> str:
        device = next(self.model.parameters()).device
        print(f"[generate] device: {device}")

        print("[generate] tokenizing...")
        t0 = time.time()
        inputs = self.tokenizer(
            user_input,
            return_tensors="pt",
        ).to(device)
        print(f"[generate] tokenize took {time.time() - t0:.2f}s")

        print("[generate] calling model.generate()...")
        t0 = time.time()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        print(f"[generate] generate took {time.time() - t0:.2f}s")

        print("[generate] decoding...")
        text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )
        print("[generate] done")

        return text.strip()
