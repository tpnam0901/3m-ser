import logging
import os
from typing import Dict, Union

import torch
from torch import Tensor
from configs.base import Config
from models.networks import MSER, AudioOnly, MMSERA, SERVER, TextOnly
from utils.torch.trainer import TorchTrainer


class Trainer(TorchTrainer):
    def __init__(
        self,
        cfg: Config,
        network: Union[MMSERA, SERVER, AudioOnly, TextOnly],
        criterion: torch.nn.CrossEntropyLoss = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.network = network
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.train()
        self.optimizer.zero_grad()

        # Prepare batch
        input_ids, audio, label = batch

        # Move inputs to cpu or gpu
        audio = audio.to(self.device)
        label = label.to(self.device)
        input_ids = input_ids.to(self.device)

        # Forward pass
        output = self.network(input_ids, audio)
        loss = self.criterion(output, label)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Calculate accuracy
        _, preds = torch.max(output[0], 1)
        accuracy = torch.mean((preds == label).float())
        return {
            "loss": loss.detach().cpu().item(),
            "acc": accuracy.detach().cpu().item(),
        }

    def test_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.eval()
        # Prepare batch
        input_ids, audio, label = batch

        # Move inputs to cpu or gpu
        audio = audio.to(self.device)
        label = label.to(self.device)
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            # Forward pass
            output = self.network(input_ids, audio)
            loss = self.criterion(output, label)
            # Calculate accuracy
            _, preds = torch.max(output[0], 1)
            accuracy = torch.mean((preds == label).float())
        return {
            "loss": loss.detach().cpu().item(),
            "acc": accuracy.detach().cpu().item(),
        }


class MarginTrainer(TorchTrainer):
    def __init__(
        self,
        cfg: Config,
        network: Union[MMSERA, SERVER, AudioOnly, TextOnly],
        criterion: torch.nn.CrossEntropyLoss = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.network = network
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

        self.opt_criterion = torch.optim.Adam(
            params=[{"params": self.criterion.parameters()}],
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta_1, cfg.adam_beta_2),
            eps=cfg.adam_eps,
            weight_decay=cfg.adam_weight_decay,
        )

        self.scheduler_criterion = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.opt_criterion,
            lr_lambda=lambda epoch: (
                ((epoch + 1) / (4 + 1)) ** 2
                if epoch < -1
                else 0.1 ** len([m for m in [8, 14, 20, 25] if m - 1 <= epoch])
            ),
        )

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.train()
        self.optimizer.zero_grad()
        self.opt_criterion.zero_grad()

        # Prepare batch
        input_ids, audio, label = batch

        # Move inputs to cpu or gpu
        audio = audio.to(self.device)
        label = label.to(self.device)
        input_ids = input_ids.to(self.device)

        # Forward pass
        output = self.network(input_ids, audio)
        loss, logits = self.criterion(output, label)

        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.opt_criterion.step()

        # Calculate accuracy
        _, preds = torch.max(logits, 1)
        accuracy = torch.mean((preds == label).float())
        return {
            "loss": loss.detach().cpu().item(),
            "acc": accuracy.detach().cpu().item(),
        }

    def test_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.eval()
        # Prepare batch
        input_ids, audio, label = batch

        # Move inputs to cpu or gpu
        audio = audio.to(self.device)
        label = label.to(self.device)
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            # Forward pass
            output = self.network(input_ids, audio)
            loss, logits = self.criterion(output, label)
            # Calculate accuracy
            _, preds = torch.max(logits, 1)
            accuracy = torch.mean((preds == label).float())
        return {
            "loss": loss.detach().cpu().item(),
            "acc": accuracy.detach().cpu().item(),
        }

    def lr_scheduler(self, step: int, epoch: int):
        if self.scheduler is not None:
            self.scheduler.step()
        self.scheduler_criterion.step()

    def save_all_states(self, path: str, global_epoch: int, global_step: int):
        checkpoint = {
            "epoch": global_epoch,
            "global_step": global_step,
            "state_dict_network": self.network.state_dict(),
            "state_optimizer": self.optimizer.state_dict(),
            "state_criterion": self.criterion.state_dict(),
            "state_lr_scheduler_criterion": self.scheduler_criterion.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint["state_lr_scheduler"] = self.scheduler.state_dict()

        ckpt_path = os.path.join(
            path, "checkpoint_{}_{}.pt".format(global_epoch, global_step)
        )
        torch.save(checkpoint, ckpt_path)
        return ckpt_path

    def load_all_states(self, path: str, device: str = "cpu"):
        dict_checkpoint = torch.load(os.path.join(path), map_location=device)

        self.start_epoch = dict_checkpoint["epoch"]
        self.global_step = dict_checkpoint["global_step"]
        self.network.load_state_dict(dict_checkpoint["state_dict_network"])
        self.optimizer.load_state_dict(dict_checkpoint["state_optimizer"])
        self.criterion.load_state_dict(dict_checkpoint["state_criterion"])
        self.scheduler_criterion.load_state_dict(
            dict_checkpoint["state_lr_scheduler_criterion"]
        )
        if self.scheduler is not None:
            self.scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])

        logging.info("Successfully loaded checkpoint from {}".format(path))
        logging.info("Resume training from epoch {}".format(self.start_epoch))


class MSER_Trainer(TorchTrainer):
    def __init__(
        self,
        cfg: Config,
        network: MSER,
        criterion: torch.nn.CrossEntropyLoss = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.network = network
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        self.temperature = 0.07

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.train()
        self.optimizer.zero_grad()

        # Prepare batch
        (
            (batch_text_inputids, batch_text_attention),
            (batch_audio, audio_length),
            (batch_label, target_labels),
        ) = batch

        # Move inputs to cpu or gpu
        batch_text_inputids = batch_text_inputids.long().to(self.device)
        batch_text_attention = batch_text_attention.to(self.device)
        batch_audio = batch_audio.to(self.device)
        audio_length = audio_length.to(self.device)
        batch_label = batch_label.to(self.device)
        target_labels = [_target.to(self.device) for _target in target_labels]
        emotion_labels = batch_label

        # Forward pass
        classification_feats_pooled, orgin_res_change, final_output = self.network(
            batch_text_inputids, batch_text_attention, batch_audio, audio_length
        )

        # Cross-entropy loss
        cel_loss = self.criterion([classification_feats_pooled], batch_label)

        # Self Contrastive loss
        l_pos_neg_self = torch.einsum(
            "nc,ck->nk", [orgin_res_change, orgin_res_change.T]
        )
        l_pos_neg_self = torch.log_softmax(l_pos_neg_self, dim=-1)
        l_pos_neg_self = l_pos_neg_self.view(-1)

        cl_self_labels = target_labels[emotion_labels[0]]

        for index in range(1, final_output.size(0)):
            cl_self_labels = torch.cat(
                (
                    cl_self_labels,
                    target_labels[emotion_labels[index]]
                    + index * emotion_labels.size(0),
                ),
                0,
            )

        l_pos_neg_self = l_pos_neg_self / self.temperature
        cl_self_loss = torch.gather(l_pos_neg_self, dim=0, index=cl_self_labels)
        cl_self_loss = -cl_self_loss.sum() / cl_self_labels.size(0)

        total_loss = cel_loss + cl_self_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        # Calculate accuracy
        _, preds = torch.max(classification_feats_pooled, 1)
        accuracy = torch.mean((preds == batch_label).float())
        return {
            "loss": total_loss.detach().cpu().item(),
            "acc": accuracy.detach().cpu().item(),
        }

    def test_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.eval()
        # Prepare batch
        (
            (batch_text_inputids, batch_text_attention),
            (batch_audio, audio_length),
            (batch_label, target_labels),
        ) = batch

        # Move inputs to cpu or gpu
        batch_text_inputids = batch_text_inputids.to(self.device)
        batch_text_attention = batch_text_attention.to(self.device)
        batch_audio = batch_audio.to(self.device)
        audio_length = audio_length.to(self.device)
        batch_label = batch_label.to(self.device)
        target_labels = [_target.to(self.device) for _target in target_labels]
        emotion_labels = batch_label

        with torch.no_grad():
            # Forward pass
            classification_feats_pooled, orgin_res_change, final_output = self.network(
                batch_text_inputids, batch_text_attention, batch_audio, audio_length
            )

            # Cross-entropy loss
            cel_loss = self.criterion([classification_feats_pooled], batch_label)

            # Self Contrastive loss
            l_pos_neg_self = torch.einsum(
                "nc,ck->nk", [orgin_res_change, orgin_res_change.T]
            )
            l_pos_neg_self = torch.log_softmax(l_pos_neg_self, dim=-1)
            l_pos_neg_self = l_pos_neg_self.view(-1)

            cl_self_labels = target_labels[emotion_labels[0]]

            for index in range(1, final_output.size(0)):
                cl_self_labels = torch.cat(
                    (
                        cl_self_labels,
                        target_labels[emotion_labels[index]]
                        + index * emotion_labels.size(0),
                    ),
                    0,
                )

            l_pos_neg_self = l_pos_neg_self / self.temperature
            cl_self_loss = torch.gather(l_pos_neg_self, dim=0, index=cl_self_labels)
            cl_self_loss = -cl_self_loss.sum() / cl_self_labels.size(0)

            total_loss = cel_loss + cl_self_loss
            # Calculate accuracy
            _, preds = torch.max(classification_feats_pooled, 1)
            accuracy = torch.mean((preds == batch_label).float())

        return {
            "loss": total_loss.detach().cpu().item(),
            "acc": accuracy.detach().cpu().item(),
        }
