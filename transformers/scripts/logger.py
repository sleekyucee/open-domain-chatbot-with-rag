#logger
import os
import wandb

class Logger:
    def __init__(self, experiment_name, project="open_domain_chatbot", mode="online", log_dir="logs/.wandb_offline"):
        """
        Initializes the WandB logger.
        :param experiment_name: Name of the run
        :param project: Project name for WandB
        :param mode: "online" or "offline"
        :param log_dir: Directory to store offline logs
        """
        if mode == "offline":
            os.environ["WANDB_MODE"] = "offline"
            os.makedirs(log_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = log_dir
        else:
            os.environ["WANDB_MODE"] = "online"

        self.logger = wandb.init(project=project, name=experiment_name)

    def log(self, metrics):
        """Logs metrics like loss, accuracy, etc."""
        try:
            self.logger.log(metrics)
        except Exception as e:
            print(f"[wandb log error] {e}")

    def save_file(self, file_path):
        """Logs a file to WandB."""
        try:
            filename_only = os.path.basename(file_path)
            wandb.save(filename_only)
            print(f"Logged file to WandB: {filename_only}")
        except Exception as e:
            print(f"[wandb file save error] {e}")

    def finish(self):
        """Ends the WandB run cleanly."""
        try:
            self.logger.finish()
        except Exception as e:
            print(f"[wandb finish error] {e}")


