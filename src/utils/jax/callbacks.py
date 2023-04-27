import logging
from abc import ABC, abstractmethod
from typing import Dict

from flax.training import train_state


class Callback(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(
        self,
        trainer,
        state: train_state.TrainState,
        step: int,
        epoch: int,
        losses: Dict[str, float],
        metrics: Dict[str, float],
        logger: logging.Logger,
        validate=False,
    ):
        """

        Args:
            trainer (Trainer): BaseTrainer in trainer.py
            state (train_state.TrainState): flax train state
            step (int): current step
            epoch (int): current epoch
            losses (Dict[str, float]): The output of the loss function at the current step
            metrics (Dict[str, float]): The output of the metrics function at the current step
            logger (logging.Logger): The logger to use for logging
            validate (bool, optional): Whether the current step is a validation step. Defaults to False.
        These arguments are required for the callback to be able to do anything useful.

        Sample usage:
        """
        raise NotImplementedError("Please implement this method")


class CheckpointCallback(Callback):
    def __init__(self, save_freq: int = 1000, max_to_keep: int = 3):
        """

        Args:
            save_dir (str):
        """
        super(CheckpointCallback, self).__init__()
        self.save_freq = save_freq
        self.max_to_keep = max_to_keep

    def __call__(
        self,
        trainer,
        state: train_state.TrainState,
        step: int,
        epoch: int,
        losses: Dict[str, float],
        metrics: Dict[str, float],
        logger: logging.Logger,
        validate=False,
    ):
        # Ignore checkpoints during validation
        if validate:
            return
        # Save checkpoint every save_freq steps
        if step % self.save_freq == 0:
            logger.info(f"Saving checkpoint at step {step}")
            trainer.save(step, keep=self.max_to_keep)
