# scripts/logger.py
import os
import wandb

class Logger:
    def __init__(self, experiment_name, project="llama3-finetune", mode="offline", log_dir="logs/.wandb_offline"):
        if mode == "offline":
            os.environ["WANDB_MODE"] = "offline"
            os.makedirs(log_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = log_dir
        else:
            os.environ["WANDB_MODE"] = "online"

        self.logger = wandb.init(project=project, name=experiment_name)

    def log(self, metrics):
        try:
            self.logger.log(metrics)
        except Exception as e:
            print(f"[wandb log error] {e}")

    def save_file(self, file_path):
        try:
            wandb.save(file_path)
            print(f"Logged file to WandB: {file_path}")
        except Exception as e:
            print(f"[wandb file save error] {e}")

    def finish(self):
        try:
            self.logger.finish()
        except Exception as e:
            print(f"[wandb finish error] {e}")

