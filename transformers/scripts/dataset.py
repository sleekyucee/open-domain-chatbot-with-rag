#dataset
import os
import json
import torch
from torch.utils.data import Dataset
from typing import Tuple
import re
from utils import (
    SPECIAL_TOKENS, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX,
    nltk_tokenize, build_vocab, load_vocab, save_vocab,
    load_glove
)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z0-9.,!?'\"]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class Seq2SeqDataset(Dataset):
    def __init__(self, config, split: str = "train"):
        data_cfg = config["data_settings"]
        model_cfg = config["model_settings"]

        self.split = split
        self.max_len = data_cfg.get("max_len", 50)
        self.min_freq = data_cfg["min_freq"]
        self.embedding_dim = model_cfg["emb_dim"]
        self.json_path = data_cfg["dataset_path"]
        self.vocab_path = data_cfg["vocab_path"]
        self.glove_path = data_cfg["glove_path"]

        with open(self.json_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        self.data = [ex for ex in all_data if ex.get("split") == split]

        if os.path.exists(self.vocab_path):
            self.word2idx, self.idx2word = load_vocab(self.vocab_path)
        else:
            if split != "train":
                raise ValueError("Vocabulary must exist for non-training splits")
            self.word2idx, self.idx2word = build_vocab(self.data, lambda t: nltk_tokenize(clean_text(t)), self.min_freq, SPECIAL_TOKENS)
            save_vocab(self.word2idx, self.idx2word, self.vocab_path)

        self.embedding_matrix = load_glove(self.glove_path, self.word2idx, self.embedding_dim)
        self.pairs = self.tokenize_pairs()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src, tgt = self.pairs[idx]
        return torch.tensor(src), torch.tensor(tgt)

    def encode(self, tokens):
        ids = [SOS_IDX] + [self.word2idx.get(tok, UNK_IDX) for tok in tokens] + [EOS_IDX]
        ids = ids[:self.max_len]
        return ids + [PAD_IDX] * (self.max_len - len(ids))

    def tokenize_pairs(self):
        pairs = []
        for ex in self.data:
            src_text = clean_text(ex["input"])
            tgt_text = clean_text(ex["response"])
            src_tokens = nltk_tokenize(src_text)
            tgt_tokens = nltk_tokenize(tgt_text)
            pairs.append((self.encode(src_tokens), self.encode(tgt_tokens)))
        return pairs

