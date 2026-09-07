import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RAYCaptionDataset(Dataset):
    """Image/caption dataset backed by a JSONL manifest.

    Each line must be: {"image": "relative/path.png", "text": "caption"}
    """

    def __init__(self, manifest, image_size=64):
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.items = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(self.root / item["image"]).convert("RGB")
        return self.transform(image), item["text"]


def build_vocab(texts, max_vocab=8192):
    counts = {}
    for text in texts:
        for token in text.lower().split():
            counts[token] = counts.get(token, 0) + 1
    words = sorted(counts, key=counts.get, reverse=True)[: max_vocab - 2]
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({w: i + 2 for i, w in enumerate(words)})
    return vocab


def encode_text(text, vocab, max_tokens=64):
    ids = [vocab.get(tok, 1) for tok in text.lower().split()[:max_tokens]]
    ids += [0] * (max_tokens - len(ids))
    return torch.tensor(ids, dtype=torch.long)
