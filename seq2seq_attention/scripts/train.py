#train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import load_config, set_seed, count_parameters
from dataset import Seq2SeqDataset
from model import Encoder, AttentionDecoder, Seq2SeqWithAttention
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
    config = load_config(config_path)
    set_seed(config["data_settings"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading datasets...")
    train_data = Seq2SeqDataset(config, split="train")
    valid_data = Seq2SeqDataset(config, split="valid")
    train_loader = DataLoader(train_data, batch_size=config["train_settings"]["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config["train_settings"]["batch_size"])

    vocab_size = len(train_data.word2idx)
    emb_matrix = train_data.embedding_matrix
    model_cfg = config["model_settings"]

    encoder = Encoder(vocab_size, model_cfg["emb_dim"], model_cfg["hidden_dim"],
                      model_cfg["n_layers"], model_cfg["dropout"], model_cfg["pad_idx"])
    decoder = AttentionDecoder(vocab_size, model_cfg["emb_dim"], model_cfg["hidden_dim"],
                               model_cfg["n_layers"], model_cfg["dropout"], model_cfg["pad_idx"])
    model = Seq2SeqWithAttention(encoder, decoder, device, model_cfg["sos_idx"], model_cfg["eos_idx"]).to(device)

    print(f"Model has {count_parameters(model):,} trainable parameters")
    model.encoder.embedding.weight.data.copy_(emb_matrix)
    model.decoder.embedding.weight.data.copy_(emb_matrix)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train_settings"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=model_cfg["pad_idx"])

    mode = "online" if config["experiment_settings"].get("use_wandb", False) else "offline"
    wandb_logger = Logger(
        experiment_name=config["experiment_settings"]["experiment_name"],
        project=config["experiment_settings"]["project"],
        mode=mode
    )

    os.makedirs("models", exist_ok=True)
    checkpoint_path = os.path.join("models", f"{config['experiment_settings']['experiment_name']}_best.pt")

    best_val_loss = float("inf")
    patience = config["train_settings"].get("early_stopping_patience", 2)
    no_improve_count = 0
    num_epochs = config["train_settings"]["num_epochs"]
    tf_ratio = config["train_settings"]["teacher_forcing_ratio"]
    clip = config["train_settings"]["clip_grad"]

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        for src, trg in tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100, leave=False):
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
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        wandb_logger.log({
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config_path': config_path
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
            wandb_logger.save_file(checkpoint_path)
        else:
            no_improve_count += 1
            print(f"No improvement. Early stopping counter: {no_improve_count}/{patience}")
            if no_improve_count >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print("\nTraining complete.")
    wandb_logger.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    train(args.config)
