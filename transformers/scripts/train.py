#train
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Seq2SeqDataset
from model import TransformerSeq2Seq
from utils import load_config, set_seed, count_parameters
from logger import Logger
from tqdm import tqdm

def shift_targets(trg):
    return trg[:, :-1], trg[:, 1:]t

def evaluate(model, loader, criterion, device, pad_idx):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            trg_input, trg_y = shift_targets(trg)
            output = model(src, trg_input)
            output = output.reshape(-1, output.size(-1))
            trg_y = trg_y.reshape(-1)
            loss = criterion(output, trg_y)
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

    model = TransformerSeq2Seq(
        vocab_size,
        model_cfg["emb_dim"],
        model_cfg["hidden_dim"],
        model_cfg["n_heads"],
        model_cfg["n_layers"],
        model_cfg["dropout"],
        model_cfg["pad_idx"]
    ).to(device)

    print(f"Model has {count_parameters(model):,} trainable parameters")
    model.embedding.weight.data.copy_(emb_matrix)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train_settings"]["learning_rate"],
        weight_decay=float(config["train_settings"].get("weight_decay", 0))
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss(
        ignore_index=model_cfg["pad_idx"],
        label_smoothing=0.1
    )

    mode = "online" if config["experiment_settings"].get("use_wandb", False) else "offline"
    wandb_logger = Logger(
        experiment_name=config["experiment_settings"]["experiment_name"],
        project=config["experiment_settings"]["project"],
        mode=mode
    )

    os.makedirs("models", exist_ok=True)
    checkpoint_path = os.path.join("models", f"{config['experiment_settings']['experiment_name']}_best.pt")

    best_val_loss = float("inf")
    no_improve_count = 0
    patience = config["train_settings"].get("early_stopping_patience", 5)

    for epoch in range(config["train_settings"]["num_epochs"]):
        model.train()
        total_train_loss = 0
        print(f"\n--- Epoch {epoch+1}/{config['train_settings']['num_epochs']} ---")

        for src, trg in tqdm(train_loader, ncols=100, desc=f"Epoch {epoch+1}"):
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            trg_input, trg_y = shift_targets(trg)
            output = model(src, trg_input)
            output = output.reshape(-1, output.size(-1))
            trg_y = trg_y.reshape(-1)
            loss = criterion(output, trg_y)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["train_settings"]["clip_grad"])
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        val_loss = evaluate(model, valid_loader, criterion, device, model_cfg["pad_idx"])
        scheduler.step(val_loss)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        wandb_logger.log({
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm.item()
        })

        if epoch > 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
            wandb_logger.save_file(checkpoint_path)
        else:
            no_improve_count += 1
            print(f"No improvement. Early stopping counter: {no_improve_count}/{patience}")
            if no_improve_count >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")
    wandb_logger.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)

