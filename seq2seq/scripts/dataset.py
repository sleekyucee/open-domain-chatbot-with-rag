#dataset
import os
import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from utils import (
    SPECIAL_TOKENS, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX,
    validate_paths, validate_split, tokenize_text, 
    build_vocab, load_vocab, save_vocab, 
    encode_sequence, load_glove_embeddings
)

class Seq2SeqDataset(Dataset):
    def __init__(self, 
                 json_path: str,
                 glove_path: str,
                 vocab_path: str,
                 embedding_dim: int = 100,
                 max_len: int = 40,
                 min_freq: int = 2,
                 split: str = "train") -> None:
        
        #validate inputs
        validate_paths(json_path)
        validate_split(split)
        
        self.split = split
        self.max_len = max_len
        self.min_freq = min_freq
        
        #load and filter data
        with open(json_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        self.data = [ex for ex in all_data if ex.get("split") == split]

        #handle vocabulary
        if os.path.exists(vocab_path):
            self.word2idx, self.idx2word = load_vocab(vocab_path)
        else:
            if split != "train":
                raise ValueError("Vocabulary must exist for non-training splits")
            self.word2idx, self.idx2word = build_vocab(self.data, min_freq)
            save_vocab(vocab_path, self.word2idx, self.idx2word)

        #load embeddings
        self.embedding_matrix = load_glove_embeddings(
            glove_path, self.word2idx, embedding_dim
        )
        self.embedding_matrix = torch.tensor(self.embedding_matrix, dtype=torch.float)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ex = self.data[idx]
        src = encode_sequence(tokenize_text(ex["input"]), self.word2idx, self.max_len)
        tgt = encode_sequence(tokenize_text(ex["response"]), self.word2idx, self.max_len)
        return torch.tensor(src), torch.tensor(tgt)

    def get_vocab_size(self) -> int:
        return len(self.word2idx)

    def get_embedding_matrix(self) -> torch.Tensor:
        return self.embedding_matrix

    def show_example(self, idx: int = 0) -> None:
        src, tgt = self.__getitem__(idx)
        print("Source:", " ".join(self.idx2word[i.item()] for i in src if i.item() != PAD_IDX))
        print("Target:", " ".join(self.idx2word[i.item()] for i in tgt if i.item() != PAD_IDX))