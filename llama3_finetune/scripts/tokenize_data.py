#tokenize_data
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

def tokenize_data(input_txt_path, output_pt_path, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token


    tokenized_data = []

    with open(input_txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Tokenizing"):
        line = line.strip()
        if not line:
            continue
        tokens = tokenizer(
            line,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        tokenized_data.append({
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0)
        })

    torch.save(tokenized_data, output_pt_path)
    print(f"Saved tokenized dataset to: {output_pt_path}")

if __name__ == "__main__":
    input_file = "/content/drive/MyDrive/open-domain-chatbot-with-rag/dataset/formatted_llama3_train.txt"
    output_file = "/content/drive/MyDrive/open-domain-chatbot-with-rag/dataset/tokenized_llama3_train.pt"
    tokenize_data(input_file, output_file)

