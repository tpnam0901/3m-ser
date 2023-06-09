import logging
import os
from typing import Dict

from utils.torch.callbacks import Callback
from utils.torch.trainer import TorchTrainer


class CheckpointCallback(Callback):
    def __init__(
        self, checkpoint_dir: str, save_freq: int = None, save_weights_only: bool = True, save_best_only: bool = False
    ):
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq
        self.save_weights_only = save_weights_only
        self.save_best_only = save_best_only
        self.curr_epoch = 0
        if save_best_only:
            self.best_val = {}

    def __call__(
        self,
        trainer: TorchTrainer,
        global_step: int,
        global_epoch: int,
        logs: Dict,
        isValPhase: bool = False,
        logger: logging = None,
    ):
        """Abstract method to be implemented by the user.

        Args:
            trainer (trainer.Trainer): trainer.TorchTrainer module
            global_step (int): The global step of the training.
            global_epoch (int): The global epoch of the training.
            logs (Dict): The logs of the training which contains the loss and the metrics. For example:
                                                            {
                                                                "loss": 0.1,
                                                                "accuracy": 0.9
                                                                "some_custom_metric": 0.5
                                                            }
            isValPhase (bool, optional): Whether the callback is called during the validation phase. Defaults to False.
        """
        if self.save_best_only:
            if isValPhase:
                for k, v in logs:
                    if (
                        k not in self.best_val
                        or (v < self.best_val[k] and "loss" in k)
                        or (v > self.best_val[k] and not "loss" in k)
                    ):
                        logger.info("Model improved from {} to {}".format(self.best_val.get(k, None), v))
                        self.best_val[k] = v
                        if self.save_weights_only:
                            trainer.save_weights(self.checkpoint_dir, None)
                        else:
                            trainer.save(self.checkpoint_dir, None)
        elif self.save_freq is None:
            if global_epoch > self.curr_epoch:
                logger.info("Saving model at epoch {}".format(global_epoch))
                if self.save_weights_only:
                    trainer.save_weights(self.checkpoint_dir, global_epoch)
                else:
                    trainer.save(self.checkpoint_dir, global_epoch)
                self.curr_epoch = global_epoch
        elif global_step % self.save_freq == 0:
            logger.info("Saving model at step {}".format(global_step))
            if self.save_weights_only:
                trainer.save_weights(self.checkpoint_dir, global_step)
            else:
                trainer.save(self.checkpoint_dir, global_step)
