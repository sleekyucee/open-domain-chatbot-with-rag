#utils
import os
import json
import pickle
import random
import numpy as np
import torch
import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tokenize(text):
    import re
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower()).strip()
    return text.split()


def build_vocab(data, tokenize_fn, min_freq=2, special_tokens=None):
    special_tokens = special_tokens or ["<pad>", "<sos>", "<eos>", "<unk>"]
    freq = {}
    for ex in data:
        for token in tokenize_fn(ex["input"] + " " + ex["response"]):
            freq[token] = freq.get(token, 0) + 1

    vocab = special_tokens.copy()
    for word, count in freq.items():
        if count >= min_freq:
            vocab.append(word)

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word


def save_vocab(word2idx, idx2word, path):
    with open(path, "wb") as f:
        pickle.dump((word2idx, idx2word), f)


def load_vocab(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_glove(glove_path, word2idx, embedding_dim=100):
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype="float32")
            embeddings_index[word] = vector

    matrix = np.zeros((len(word2idx), embedding_dim))
    for word, idx in word2idx.items():
        if word in embeddings_index:
            matrix[idx] = embeddings_index[word]
        else:
            matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

    return torch.tensor(matrix, dtype=torch.float)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def decode_tokens(token_ids, idx2word):
    tokens = [idx2word.get(i, "<unk>") for i in token_ids if idx2word.get(i) != "<pad>"]
    return " ".join(tokens).replace(" <eos>", "").strip()

