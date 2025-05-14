import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_config, set_seed, count_parameters
from dataset import Seq2SeqDataset
from model import Encoder, Decoder, Seq2Seq
from logger import Logger

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
    #load config and seed
    config = load_config(config_path)
    set_seed(config["data_settings"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load datasets
    print("Loading datasets...")
    train_data = Seq2SeqDataset(config, split="train")
    valid_data = Seq2SeqDataset(config, split="valid")
    train_loader = DataLoader(train_data, batch_size=config["train_settings"]["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config["train_settings"]["batch_size"])

    #model setup
    vocab_size = len(train_data.word2idx)
    emb_matrix = train_data.embedding_matrix
    model_cfg = config["model_settings"]

    encoder = Encoder(vocab_size, model_cfg["emb_dim"], model_cfg["hidden_dim"],
                      model_cfg["n_layers"], model_cfg["dropout"], model_cfg["pad_idx"], model_cfg["rnn_type"])
    decoder = Decoder(vocab_size, model_cfg["emb_dim"], model_cfg["hidden_dim"],
                      model_cfg["n_layers"], model_cfg["dropout"], model_cfg["pad_idx"], model_cfg["rnn_type"])

    model = Seq2Seq(encoder, decoder, device, model_cfg["sos_idx"], model_cfg["eos_idx"], model_cfg["rnn_type"]).to(device)

    print(f"Model has {count_parameters(model):,} trainable parameters")

    model.encoder.embedding.weight.data.copy_(emb_matrix)
    model.decoder.embedding.weight.data.copy_(emb_matrix)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train_settings"]["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=model_cfg["pad_idx"])

    #wandb logger setup
    mode = "online" if config["experiment_settings"].get("use_wandb", False) else "offline"
    wandb_logger = Logger(
        experiment_name=config["experiment_settings"]["experiment_name"],
        project=config["experiment_settings"]["project"],
        mode=mode
    )

    #model save dir
    os.makedirs("models", exist_ok=True)
    experiment_name = config["experiment_settings"]["experiment_name"]
    checkpoint_path = os.path.join("models", f"{experiment_name}_best.pt")

    #training loop
    best_val_loss = float("inf")
    num_epochs = config["train_settings"]["num_epochs"]
    tf_ratio = config["train_settings"]["teacher_forcing_ratio"]
    clip = config["train_settings"]["clip_grad"]

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

        #log to wandb
        wandb_logger.log({
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
            wandb_logger.save_file(checkpoint_path)

    print(f"\nTraining complete.")
    wandb_logger.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML (e.g., configs/gru.yaml)")
    args = parser.parse_args()

    train(args.config)

