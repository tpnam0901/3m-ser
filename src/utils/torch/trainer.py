import datetime
import logging
import os
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch import nn
from transformers import BertTokenizer

from src.data.dataloader import build_train_test_dataset
from src.models import networks
from src.utils.loggings import get_log_text


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Model
        try:
            model_fn = getattr(networks, opt.model_type)
        except AttributeError:
            raise NotImplementedError("Model {} is not implemented".format(opt.model_type))
        self.model = model_fn(
            num_classes=opt.num_classes,
            num_attention_head=opt.num_attention_head,
            dropout=opt.dropout,
        )

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Preapre the checkpoint directory
        self.opt.checkpoint_dir = self.checkpoint_dir = os.path.join(
            os.path.abspath(opt.checkpoint_dir), opt.model_type, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self.log_dir = os.path.join(self.checkpoint_dir, "logs")
        self.weight_dir = os.path.join(self.checkpoint_dir, "weights")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.weight_dir, exist_ok=True)
        self.init_logger()
        self.opt.save(self.opt)

        # Build dataset
        self.train_ds, self.test_ds = build_train_test_dataset(opt.data_root)

        # Build optimizer and criterion
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=opt.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opt.learning_rate_step_size, gamma=opt.learning_rate_gamma
        )

    def init_logger(self):
        """Create logger for training with two handlers: one for writing to a file and one for writing to the console.
        The logger has two loggers: hide and show. hide is used for writing the loss writing the loss and
        metrics to a file. show is used for writing the loss and metrics to both the file and the console.
        The log file is saved in the log_dir.
        """
        logger = logging.getLogger("training")
        logger.setLevel(logging.INFO)
        logging.basicConfig(format="%(asctime)s - Training: %(message)s")
        file_handler = logging.FileHandler(os.path.join(self.log_dir, "log.txt"))
        logger.addHandler(file_handler)

    def train_step(self, batch):
        self.optimizer.zero_grad()

        # Prepare batch
        text, label, sprectrome = batch["text"], batch["label"], batch["sprectrome"]
        label = torch.tensor([int(label)])
        input_ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)

        # Move inputs to cpu or gpu
        sprectrome = sprectrome.to(self.device)
        label = label.to(self.device)
        input_ids = input_ids.to(self.device)

        # Forward pass
        output = self.model(input_ids, sprectrome)
        loss = self.criterion(output, label)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Calculate accuracy
        _, preds = torch.max(output, 1)
        accuracy = torch.mean((preds == label).float())
        return loss.detach().cpu().item(), accuracy.detach().cpu().item()

    def eval_step(self, batch):
        # Prepare batch
        text, label, sprectrome = batch["text"], batch["label"], batch["sprectrome"]
        label = torch.tensor([int(label)])
        input_ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)

        # Move inputs to cpu or gpu
        sprectrome = sprectrome.to(self.device)
        label = label.to(self.device)
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            # Forward pass
            output = self.model(input_ids, sprectrome)
            loss = self.criterion(output, label)
            # Calculate accuracy
            _, preds = torch.max(output, 1)
            accuracy = torch.mean((preds == label).float())
        return loss.detach().cpu().item(), accuracy.detach().cpu().item()

    def train_epoch(
        self,
        train_ds: List,
        global_step: int,
        global_epoch: int,
    ) -> Tuple[int, int]:
        """Train the model for one epoch.

        Args:
            train_ds (Union[tf.data.Dataset, tf.keras.utils.Sequence]): Training dataset.
            global_epoch (int): The current epoch.
            global_step (int): The current step.
        Returns:
            Tuple[int, int]: The updated global_epoch and global_step.
        """
        logger = logging.getLogger("training")
        pbar = tqdm.tqdm(total=len(train_ds))
        pbar.update(1)

        total_losses = {}
        total_metrics = {}
        # Start training model
        for batch in train_ds:
            if batch["sprectrome"].shape[2] > 65:
                losses, metrics = self.train_step(batch)
                # Log the losses and metrics
                log_text = get_log_text(losses, total_losses, name="loss")
                if metrics is not None:
                    log_text += get_log_text(metrics, total_metrics, name="metric")
                pbar.set_description(log_text)
                pbar.update(1)
                global_step += 1

        # Sumarize the epoch with the mean of the losses, metrics
        log_summary = "Epoch summary:\n\t\t\t\t"
        for k, v in total_losses.items():
            log_summary += "{}: {:.4f} ".format(k, sum(v) / len(v))
            mlflow.log_metric(f"epoch_{k}", sum(v) / len(v))
        log_summary += "\n\t\t\t\t"
        for k, v in total_metrics.items():
            log_summary += "{}: {:.4f} ".format(k, sum(v) / len(v))
            mlflow.log_metric(f"epoch_{k}", sum(v) / len(v))
        logger.info(log_summary)

        global_epoch += 1
        return global_step, global_epoch

    def evaluate(
        self,
        test_ds: List,
    ) -> Tuple[Dict, Dict]:
        """Evaluate the model on the validation set.

        Args:
            val_ds (Union[tf.data.Dataset, tf.keras.utils.Sequence]): Validation dataset.

        Returns:
            Tuple[Dict, Dict]: The losses and metrics on the validation set.
        """
        logger = logging.getLogger("training")
        logger.info("Starting validation on validation set with {} samples".format(len(test_ds)))
        total_losses = {}
        total_metrics = {}

        self.model.eval()
        # Start evaluating model
        with tqdm.tqdm(total=len(test_ds)) as pbar:
            pbar.update(1)
            for batch in test_ds:
                losses, metrics = self.eval_step(batch)
                log_text = get_log_text(losses, total_losses, name="loss")
                if metrics is not None:
                    log_text += get_log_text(metrics, total_metrics, name="metric")
                pbar.update(1)
        self.model.train()

        # Sumarize the validation with the mean of the losses, metrics
        log_summary = "Validation summary:\n\t\t\t\t"
        for k, v in total_losses.items():
            log_summary += "{}: {:.4f} ".format(k, sum(v) / len(v))
        log_summary += "\n\t\t\t\t"
        for k, v in total_metrics.items():
            log_summary += "{}: {:.4f} ".format(k, sum(v) / len(v))
        logger.info(log_summary)

        return total_losses, total_metrics

    def save_model(self, path: str):
        torch.save(self.model, path)

    def train(self):
        self.model.train()
        self.model.to(self.device)

        best_test_loss, best_acc_loss = float("inf"), 0.0
        mlflow.set_tracking_uri(uri=f'file://{os.path.join(self.log_dir, "mlruns")}')
        with mlflow.start_run():
            global_step = 1
            for epoch in range(self.opt.num_epochs):
                logger = logging.getLogger("training")
                logger.info("Start training epoch {}/{}".format(epoch + 1, self.opt.num_epochs))
                global_step, _ = self.train_epoch(self.train_ds, global_step, epoch + 1)
                total_losses, total_metrics = self.evaluate(self.test_ds)
                for v in total_losses.values():
                    loss = np.mean(v)
                    if loss < best_test_loss:
                        logger.info(
                            "Loss improved from {:.4f} to {:.4f}, saving model to {}".format(
                                best_test_loss, loss, os.path.join(self.weight_dir, "mmsera_best_val_loss.pt")
                            )
                        )
                        best_test_loss = loss
                        self.save_model(os.path.join(self.weight_dir, "mmsera_best_val_loss.pt"))
                    mlflow.log_metric("test_loss", loss)

                for v in total_metrics.values():
                    acc = np.mean(v)
                    if acc > best_acc_loss:
                        logger.info(
                            "Accuracy improved from {:.4f} to {:.4f}, saving model to {}".format(
                                best_acc_loss, acc, os.path.join(self.weight_dir, "mmsera_best_val_acc.pt")
                            )
                        )
                        best_acc_loss = acc
                        self.save_model(os.path.join(self.weight_dir, "mmsera_best_val_acc.pt"))
                    mlflow.log_metric("test_acc", acc)
                if epoch % self.opt.save_freq == 0:
                    self.save_model(os.path.join(self.weight_dir, f"mmsera_epoch_{epoch}.pt"))
                    logger.info("Saving model to {}".format(os.path.join(self.weight_dir, f"mmsera_epoch_{epoch}.pt")))
