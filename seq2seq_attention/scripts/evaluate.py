#evaluate
import os
import math
import torch
import wandb
from torch.utils.data import DataLoader
from utils import load_config, decode_tokens
from dataset import Seq2SeqDataset
from model import Encoder, AttentionDecoder, Seq2SeqWithAttention
import evaluate


def generate_tokens(model, src, max_len, sos_idx, eos_idx, pad_idx, device):
    model.eval()
    generated = []

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src.to(device))
        input = torch.tensor([sos_idx] * src.size(0)).to(device)

        for _ in range(max_len):
            output, hidden = model.decoder(input, hidden, encoder_outputs)
            top1 = output.argmax(1)
            generated.append(top1.unsqueeze(1))
            input = top1

        tokens = torch.cat(generated, dim=1)
    return tokens


def evaluate_model(config_path):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load test set
    test_data = Seq2SeqDataset(config, split="test")
    test_loader = DataLoader(test_data, batch_size=1)

    vocab_size = len(test_data.word2idx)
    emb_matrix = test_data.embedding_matrix
    model_cfg = config["model_settings"]
    pad_idx = model_cfg["pad_idx"]
    sos_idx = model_cfg["sos_idx"]
    eos_idx = model_cfg["eos_idx"]

    #initialize model
    encoder = Encoder(vocab_size, model_cfg["emb_dim"], model_cfg["hidden_dim"],
                      model_cfg["n_layers"], model_cfg["dropout"], pad_idx)

    decoder = AttentionDecoder(vocab_size, model_cfg["emb_dim"], model_cfg["hidden_dim"],
                               model_cfg["n_layers"], model_cfg["dropout"], pad_idx)

    model = Seq2SeqWithAttention(encoder, decoder, device, sos_idx, eos_idx).to(device)

    #load checkpoint
    checkpoint_path = os.path.join("models", f"{config['experiment_settings']['experiment_name']}_best.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    #init wandb
    use_wandb = config["experiment_settings"].get("use_wandb", False)
    if use_wandb:
        wandb.init(
            project=config["experiment_settings"]["project"],
            name=config["experiment_settings"]["experiment_name"] + "_eval",
            config=config
        )

    #evaluation
    predictions, references = [], []
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    total_loss, total_tokens = 0, 0

    for src, trg in test_loader:
        src, trg = src.to(device), trg.to(device)
        output = model(src, trg, teacher_forcing_ratio=0.0)
        output_dim = output.shape[-1]
        output_flat = output[:, 1:].reshape(-1, output_dim)
        trg_flat = trg[:, 1:].reshape(-1)

        loss = criterion(output_flat, trg_flat)
        total_loss += loss.item()
        total_tokens += (trg_flat != pad_idx).sum().item()

        pred_ids = generate_tokens(model, src, config["data_settings"]["max_len"],
                                   sos_idx, eos_idx, pad_idx, device)[0]
        ref_ids = trg[0]

        pred_text = decode_tokens(pred_ids.tolist(), test_data.idx2word)
        ref_text = decode_tokens(ref_ids.tolist(), test_data.idx2word)

        predictions.append(pred_text)
        references.append(ref_text)

    #compute metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])["bleu"]
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    meteor_score = meteor.compute(predictions=predictions, references=references)["meteor"]
    avg_loss = total_loss / len(test_loader)
    perplexity = math.exp(avg_loss) if avg_loss < 50 else float("inf")

    #print results
    print(f"\nEvaluation Metrics")
    print(f"BLEU Score     : {bleu_score:.4f}")
    print(f"ROUGE Scores   : {rouge_scores}")
    print(f"METEOR Score   : {meteor_score:.4f}")
    print(f"Perplexity     : {perplexity:.4f}\n")

    if use_wandb:
        wandb.log({
            "BLEU": bleu_score,
            "ROUGE-1": rouge_scores["rouge1"],
            "ROUGE-2": rouge_scores["rouge2"],
            "ROUGE-L": rouge_scores["rougeL"],
            "METEOR": meteor_score,
            "Perplexity": perplexity
        })
        wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML (e.g., configs/attention_lstm.yaml)")
    args = parser.parse_args()

    evaluate_model(args.config)

