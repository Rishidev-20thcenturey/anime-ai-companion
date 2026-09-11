"""N8 Qwen3-4B text-encoder smoke test.

Run on a CUDA T4/P100-class environment after installing requirements.
This intentionally loads Qwen3-4B in 8-bit mode and keeps it frozen.
"""
import torch

from .text_encoder import RAYTextEncoder


PROMPT = "a golden retriever puppy sitting on a wooden porch, soft morning light"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("N8 Qwen3-4B smoke test requires a CUDA GPU (Kaggle T4/P100).")

    device_name = torch.cuda.get_device_name(0)
    print(f"device={device_name}")
    try:
        encoder = RAYTextEncoder(max_tokens=256, load_in_8bit=True)
        hidden, text_mask = encoder.encode_texts([PROMPT])
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "Qwen3-4B 8-bit text encoder ran out of GPU memory. "
            "Use a clean T4 session and ensure no other large models are loaded."
        ) from exc

    assert hidden.shape[0] == 1
    assert hidden.shape[2] == 2560, hidden.shape
    assert hidden.shape[:2] == text_mask.shape, (hidden.shape, text_mask.shape)
    assert all(not p.requires_grad for p in encoder.parameters())

    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"prompt={PROMPT}")
    print(f"embedding_shape={tuple(hidden.shape)}  # [1, seq_len, 2560]")
    print(f"padding_mask_shape={tuple(text_mask.shape)}")
    print(f"hidden_size={encoder.hidden_size}")
    print(f"gpu_memory_allocated_gb={allocated_gb:.2f}")
    print(f"gpu_memory_reserved_gb={reserved_gb:.2f}")
    print("qwen3_text_encoder_smoke: PASS")


if __name__ == "__main__":
    main()
