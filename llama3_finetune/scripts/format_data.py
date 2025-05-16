import json
import os
from tqdm import tqdm

def format_dialogues(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    formatted_lines = []

    for pair in tqdm(pairs, desc="Formatting"):
        user = pair.get("input", "").strip()
        assistant = pair.get("response", "").strip()

        if not user or not assistant:
            continue

        formatted = f"<|user|>: {user}\n<|assistant|>: {assistant}\n"
        formatted_lines.append(formatted)

    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.writelines(formatted_lines)

    print(f"Saved formatted file to: {output_path}")

if __name__ == "__main__":
    input_file = "/content/drive/MyDrive/open-domain-chatbot-with-rag/dataset/merged_pairs.json"
    output_file = "/content/drive/MyDrive/open-domain-chatbot-with-rag/dataset/formatted_llama3_train.txt"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    format_dialogues(input_file, output_file)

