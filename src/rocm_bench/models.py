"""Model loading and management utilities."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str, dtype=torch.float16, device_map="auto"):
    """Load a HuggingFace model with ROCm-optimized settings."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def get_model_info(model) -> dict:
    """Get model memory and parameter info."""
    param_count = sum(p.numel() for p in model.parameters())
    param_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    return {
        "parameters": param_count,
        "parameter_size_gb": round(param_size_gb, 3),
        "dtype": str(next(model.parameters()).dtype),
        "device_map": str(getattr(model, "hf_device_map", "unknown")),
    }
