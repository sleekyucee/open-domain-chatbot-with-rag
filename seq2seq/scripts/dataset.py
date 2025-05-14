#dataset
import os
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import nltk_tokenize, build_vocab, save_vocab, load_vocab, load_glove

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

class Seq2SeqDataset(Dataset):
    def __init__(self, config, split="train"):
        data_cfg = config["data_settings"]
        self.split = split
        self.max_len = data_cfg.get("max_len", 50)
        self.embedding_dim = config["model_settings"]["emb_dim"]
        self.json_path = data_cfg["dataset_path"]
        self.glove_path = data_cfg["glove_path"]
        self.vocab_path = data_cfg["vocab_path"]
        self.min_freq = data_cfg["min_freq"]

        #load and filter split data
        self.data = self.load_split_data()

        #build or load vocabulary
        if os.path.exists(self.vocab_path):
            self.word2idx, self.idx2word = load_vocab(self.vocab_path)
            print(f"Loaded vocab from {self.vocab_path}")
        else:
            self.word2idx, self.idx2word = build_vocab(self.data, nltk_tokenize, self.min_freq, SPECIAL_TOKENS)
            save_vocab(self.word2idx, self.idx2word, self.vocab_path)
            print(f"Saved vocab to {self.vocab_path}")

        #build embedding matrix
        self.embedding_matrix = load_glove(self.glove_path, self.word2idx, self.embedding_dim)

        #tokenize pairs
        self.pairs = self.tokenize_pairs()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

    def load_split_data(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        return [ex for ex in all_data if ex.get("split") == self.split]

    def encode(self, tokens):
        ids = [SOS_IDX] + [self.word2idx.get(tok, UNK_IDX) for tok in tokens] + [EOS_IDX]
        ids = ids[:self.max_len]
        return ids + [PAD_IDX] * (self.max_len - len(ids))

    def tokenize_pairs(self):
        print(f"Tokenizing data ({self.split})...")
        pairs = []
        for ex in tqdm(self.data):
            src_tokens = nltk_tokenize(ex["input"])
            tgt_tokens = nltk_tokenize(ex["response"])
            pairs.append((self.encode(src_tokens), self.encode(tgt_tokens)))
        return pairs

