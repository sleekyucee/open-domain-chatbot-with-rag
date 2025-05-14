#train
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_config, set_seed, count_parameters
from dataset import Seq2SeqDataset
from model import Encoder, Decoder, Seq2Seq
import wandb

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0.0)  
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            total_loss += loss.item()

    return total_loss / len(loader)


def train(config_path):
    #load config
    config = load_config(config_path)
    set_seed(config["data_settings"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load dataset
    print("Loading datasets...")
    train_data = Seq2SeqDataset(config, split="train")
    valid_data = Seq2SeqDataset(config, split="valid")

    train_loader = DataLoader(train_data, batch_size=config["train_settings"]["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config["train_settings"]["batch_size"])

    #model setup
    vocab_size = len(train_data.word2idx)
    emb_matrix = train_data.embedding_matrix
    model_cfg = config["model_settings"]

    encoder = Encoder(vocab_size,
                      model_cfg["emb_dim"],
                      model_cfg["hidden_dim"],
                      model_cfg["n_layers"],
                      model_cfg["dropout"],
                      model_cfg["pad_idx"],
                      model_cfg["rnn_type"])

    decoder = Decoder(vocab_size,
                      model_cfg["emb_dim"],
                      model_cfg["hidden_dim"],
                      model_cfg["n_layers"],
                      model_cfg["dropout"],
                      model_cfg["pad_idx"],
                      model_cfg["rnn_type"])

    model = Seq2Seq(encoder, decoder, device,
                    model_cfg["sos_idx"], model_cfg["eos_idx"], model_cfg["rnn_type"]).to(device)

    print(f"Model has {count_parameters(model):,} trainable parameters")

    model.encoder.embedding.weight.data.copy_(emb_matrix)
    model.decoder.embedding.weight.data.copy_(emb_matrix)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train_settings"]["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=model_cfg["pad_idx"])

    #wandB setup
    use_wandb = config["experiment_settings"].get("use_wandb", False)
    if use_wandb:
        wandb.init(
            project=config["experiment_settings"]["project"],
            name=config["experiment_settings"]["experiment_name"],
            config=config
        )
        wandb.watch(model)

    #training loop
    best_val_loss = float("inf")
    num_epochs = config["train_settings"]["num_epochs"]
    clip = config["train_settings"]["clip_grad"]
    tf_ratio = config["train_settings"]["teacher_forcing_ratio"]
    ckpt_dir = "models"
    os.makedirs(ckpt_dir, exist_ok=True)
    experiment_name = config["experiment_settings"]["experiment_name"]
    checkpoint_path = os.path.join(ckpt_dir, f"{experiment_name}_best.pt")

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg, tf_ratio)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        val_loss = evaluate(model, valid_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if use_wandb:
            wandb.log({
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "epoch": epoch + 1
            })

        #save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config_path': config_path
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

    print("Training complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML (gru.yaml or lstm.yaml)")
    args = parser.parse_args()

    train(args.config)

