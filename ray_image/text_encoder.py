"""RAY-IMAGE text encoders.

N8 uses a frozen Qwen3-4B encoder with the Qwen tokenizer. A small legacy
encoder is retained so existing N0-N7/N5 smoke checkpoints remain loadable.
"""
import torch
from torch import nn


class LegacyRAYTextEncoder(nn.Module):
    """Original lightweight transformer encoder kept for legacy checkpoints."""

    def __init__(self, vocab_size=8192, dim=256, max_tokens=64, heads=4, depth=4):
        super().__init__()
        self.token = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(1, max_tokens, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.hidden_size = dim

    def forward(self, tokens, mask=None):
        x = self.token(tokens) + self.pos[:, :tokens.shape[1]]
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.norm(x)


class RAYTextEncoder(nn.Module):
    """Frozen Qwen/Qwen3-4B text encoder for N8.

    The model is loaded through Transformers + bitsandbytes in 8-bit mode and
    kept entirely frozen. The returned conditioning sequence is the final
    transformer hidden state, with hidden size 2560 for Qwen3-4B.
    """

    MODEL_ID = "Qwen/Qwen3-4B"
    HIDDEN_SIZE = 2560

    def __init__(self, model_name=MODEL_ID, max_tokens=256, load_in_8bit=True):
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "N8 Qwen text encoding requires transformers, bitsandbytes, and accelerate."
            ) from exc

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.hidden_size = int(self.model.config.hidden_size)
        if self.hidden_size != self.HIDDEN_SIZE:
            raise ValueError(
                f"Expected Qwen3-4B hidden size {self.HIDDEN_SIZE}, got {self.hidden_size}"
            )

    @property
    def input_device(self):
        return next(self.model.parameters()).device

    def tokenize(self, texts, max_length=None):
        max_length = max_length or self.max_tokens
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    @torch.no_grad()
    def encode_texts(self, texts, max_length=None):
        """Return (last_hidden_state, padding_mask) for a batch of strings."""
        encoded = self.tokenize(texts, max_length=max_length)
        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
        outputs = self.model.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        text_mask = encoded.get("attention_mask")
        if text_mask is None:
            text_mask = torch.zeros(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        else:
            text_mask = ~text_mask.bool()
        return hidden, text_mask

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None):
        """Return the final Qwen hidden state for already-tokenized inputs."""
        input_ids = input_ids.to(self.input_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.input_device)
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state
