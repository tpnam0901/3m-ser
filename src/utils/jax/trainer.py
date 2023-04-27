import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import datetime
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlflow
import optax
import tensorflow as tf
import tqdm
from flax import linen as nn
from flax.training import checkpoints, train_state

import jax
import jax.numpy as jnp
import optimizers
from callbacks import Callback


def _get_log_text(
    values: Union[List, Dict, float],
    total_values: Dict[str, List],
    name="loss",
):
    """A helper function to log the values.

    Args:
        values (_type_): The values to log.
        total_values (_type_): All the values will be added to this dictionary.
        name (str, optional): The prefix of the values. Defaults to "loss".

    Returns:
        str: The log message.
    """
    log_text = ""
    if type(values) == list:
        for i, val in enumerate(values):
            log_text += "{}_{}: {:.4f} ".format(name, i, val)
            if name not in total_values.keys():
                total_values[name] = []
            total = total_values[name]
            total.append(val)
            mlflow.log_metric(f"{name}_{i}", val)
    elif type(values) == dict:
        for k, v in values.items():
            log_text += "{}: {:.4f} ".format(k, v)
            if k not in total_values.keys():
                total_values[k] = []
            total = total_values[k]
            total.append(v)
            mlflow.log_metric(k, v)
    else:
        log_text += "{}: {:.4f} ".format(name, values)
        if name not in total_values.keys():
            total_values[name] = []
        total = total_values[name]
        total.append(values)
        mlflow.log_metric(name, values)
    return log_text


class BaseTrainer(ABC):
    def __init__(
        self,
        backbone: nn.Module,
        dummy_input: Union[jnp.ndarray, List, Dict],
        checkpoints_dir: str = "checkpoints",
    ):
        """Base Trainer class for training Flax models which is based on the Keras API.

        Args:
            backbone (nn.Module): Flax model to train.
            dummy_input (Union[jnp.ndarray, List, Dict]): Dummy input to compile the model.
            log_dir (str, optional): Directory to save logs. Defaults to "_logs
        """
        self.backbone = backbone
        self.dummy_input = dummy_input
        self.is_compiled = False

        self.checkpoints_dir = os.path.join(checkpoints_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.log_dir = os.path.join(self.checkpoints_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.init_logger()

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

    def train_epoch(
        self,
        train_ds: Union[tf.data.Dataset, tf.keras.utils.Sequence],
        global_step: int,
        global_epoch: int,
        callbacks: List[Callable],
        verbose=1,
    ) -> Tuple[int, int]:
        """Train the model for one epoch.

        Args:
            train_ds (Union[tf.data.Dataset, tf.keras.utils.Sequence]): Training dataset.
            global_epoch (int): The current epoch.
            global_step (int): The current step.
            callbacks (List[Callable]): List of callbacks.
            verbose (int, optional): . There are three options: 0, 1, 2. Defaults to 1.
            0 - No logs will be written.
            1 - Show the progress by progress bar.
            2 - Show the progress by line.

        Returns:
            Tuple[int, int]: The updated global_epoch and global_step.
        """
        logger = logging.getLogger("training")
        if verbose == 1:
            pbar = tqdm.tqdm(total=len(train_ds))
            pbar.update(1)

        total_losses = {}
        total_metrics = {}
        # Start training model
        for batch in train_ds:
            losses, metrics = self.train_step(self.state, batch)

            # Log the losses and metrics
            log_text = _get_log_text(losses, total_losses, name="loss")
            if metrics is not None:
                log_text += _get_log_text(metrics, total_metrics, name="metric")
            if callbacks is not None:
                for callback in callbacks:
                    callback(self, self.state, global_step, global_epoch, losses, metrics, logger, validate=False)
            if verbose == 1:
                pbar.set_description(log_text)
                pbar.update(1)
            elif verbose == 2:
                logger.info(log_text)
            global_step += 1
            mlflow.log_metric("learning_rate", self.state.opt_state.hyperparams["learning_rate"])

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
        val_ds: Union[tf.data.Dataset, tf.keras.utils.Sequence],
    ) -> Tuple[Dict, Dict]:
        """Evaluate the model on the validation set.

        Args:
            val_ds (Union[tf.data.Dataset, tf.keras.utils.Sequence]): Validation dataset.

        Returns:
            Tuple[Dict, Dict]: The losses and metrics on the validation set.
        """
        logger = logging.getLogger("training")
        logger.info("Starting validation on validation set with {} samples".format(len(val_ds)))
        total_losses = {}
        total_metrics = {}

        # Start evaluating model
        with tqdm.tqdm(total=len(val_ds)) as pbar:
            pbar.update(1)
            for batch in val_ds:
                losses, metrics = self.eval_step(self.state, batch)
                log_text = _get_log_text(losses, total_losses, name="loss")
                if metrics is not None:
                    log_text += _get_log_text(metrics, total_metrics, name="metric")
                pbar.update(1)

        # Sumarize the validation with the mean of the losses, metrics
        log_summary = "Validation summary:\n\t\t\t\t"
        for k, v in total_losses.items():
            log_summary += "{}: {:.4f} ".format(k, sum(v) / len(v))
        log_summary += "\n\t\t\t\t"
        for k, v in total_metrics.items():
            log_summary += "{}: {:.4f} ".format(k, sum(v) / len(v))
        logger.info(log_summary)

        return total_losses, total_metrics

    def compile(
        self,
        loss_fn: Union[Callable, List[Callable], Dict[str, Callable]],
        metrics: Union[Callable, List[Callable], Dict[str, Callable]] = None,
        optimizer: Union[str, optax.GradientTransformation] = "sgd",
    ):
        """Compile the model with the loss function, metrics and optimizer.

        Args:
            loss_fn (Union[Callable, List[Callable], Dict[str, Callable]]): Loss function which is used in training.
            metrics (Union[Callable, List[Callable], Dict[str, Callable]], optional): Metrics which are used in training. Defaults to None.
            optimizer (Union[str, optax.GradientTransformation], optional): Optimizer which is used in training. Defaults to "sgd".
        """
        if type(optimizer) == str:
            available_optimizers = {
                "sgd": optimizers.sgd(learning_rate=0.01),
                "adam": optimizers.adam(learning_rate=0.01),
                "rmsprop": optimizers.rmsprop(learning_rate=0.01),
                "adagrad": optimizers.adagrad(learning_rate=0.01),
                "adafactor": optimizers.adafactor(learning_rate=0.01),
                "adamw": optimizers.adamw(learning_rate=0.01, weight_decay=0.01),
            }
            optimizer = available_optimizers.get(optimizer, None)
            if optimizer is None:
                raise NotImplementedError(
                    "{} is not found. List of available optimizers: {}".format(optimizer, list(available_optimizers.keys()))
                )

        # All callable functions will be wrapped in a list
        if callable(loss_fn):
            loss_fn = [loss_fn]
        self.loss_fn = loss_fn
        if callable(metrics):
            metrics = [metrics]
        self.metrics = metrics

        # Initialize the model with flax training state
        params = self.backbone.init(jax.random.PRNGKey(0), self.dummy_input)
        self.state = train_state.TrainState.create(apply_fn=self.backbone.apply, params=params, tx=optimizer)
        self.is_compiled = True

    def summary(self):
        """Summary of the model using tabulate."""
        print(self.backbone.tabulate(jax.random.PRNGKey(0), self.dummy_input))

    def fit(
        self,
        epochs: int,
        train_ds: Union[tf.data.Dataset, tf.keras.utils.Sequence],
        val_ds: Union[tf.data.Dataset, tf.keras.utils.Sequence] = None,
        verbose: Optional[int] = 1,
        callbacks: List[Callback] = None,
    ):
        assert self.is_compiled, "Please compile the model first"
        mlflow.set_tracking_uri(uri=f'file://{os.path.abspath(os.path.join(self.log_dir, "mlruns"))}')
        with mlflow.start_run():
            global_step = 1
            for epoch in range(epochs):
                logger = logging.getLogger("training")
                logger.info("Start training epoch {}/{}".format(epoch + 1, epochs))
                global_step, _ = self.train_epoch(train_ds, global_step, epoch + 1, callbacks=callbacks, verbose=verbose)
                if val_ds is not None:
                    total_losses, total_metrics = self.evaluate(val_ds)
                    if callbacks is not None:
                        for callback in callbacks:
                            callback(
                                self, self.state, global_step, epoch + 1, total_losses, total_metrics, logger, validate=True
                            )

    def save(self, step: int = 0, keep: int = 3):
        """Save the model to the checkpoint directory.

        Args:
            step (int, optional): The current step. Defaults to 0.
            keep (int, optional): Number of checkpoints to keep. Defaults to 3.
        """
        checkpoints.save_checkpoint(
            ckpt_dir=os.path.join(self.checkpoints_dir, "weights"), target=self.state, step=step, keep=keep
        )

    def load(self, ckpt_dir: str):
        """Load the model from the checkpoint directory.

        Args:
            ckpt_dir (str): Path to the checkpoint directory.
        """
        assert os.path.exists(ckpt_dir), "Checkpoint directory is not found at {}".format(ckpt_dir)
        checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=self.state)

    @abstractmethod
    def train_step(
        self,
        state: train_state.TrainState,
        batch: Any,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """A single training step which performs the forward pass, losses & metrics calculation, and backpropagation.
        The losses and metrics are based on the loss_fn and metrics provided in the compile function and
        assigned to the self.loss_fn and self.metrics.
        You need to implement this function in your subclass and control the loss and metrics calculation.

        Args:
            state (flax.training.train_state.TrainState): The current training state.
            batch (Any): The input of the model.

        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: losses and metrics. The losses is required.
            If no metrics are provided, return None.

        Sample code:
            def train_step(self, state, batch):
                @jax.jit
                def _train_step(state, params, batch):
                    def losses_fn(params, batch):
                        logits = state.apply_fn(params, batch)
                        total_loss = 0
                        for loss_fn in self.loss_fn:
                            total_loss += loss_fn(logits, jnp.zeros_like(logits))
                        return total_loss

                    loss, grads = jax.value_and_grad(losses_fn)(params, batch)
                    state = state.apply_gradients(grads=grads)
                    return loss, state

                losses, self.state = _train_step(state, state.params, batch)
                metrics = None
                return losses, metrics
        """
        raise NotImplementedError("Please implement this method")

    @abstractmethod
    def eval_step(
        self,
        state: train_state.TrainState,
        batch: Any,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """A single evaluation step which performs the forward pass, losses & metrics calculation.
        The losses and metrics are based on the loss_fn and metrics provided in the compile function and
        assigned to the self.loss_fn and self.metrics.
        You need to implement this function in your subclass and control the loss and metrics calculation.

        Args:
            state (flax.training.train_state.TrainState): The current training state.
            batch (Any): The input of the model.

        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: losses and metrics. The losses is required.
            If no metrics are provided, return None.

        Sample code:
            def eval_step(self, state, batch):
                @jax.jit
                def _eval_step(state, params, batch):
                    logits = state.apply_fn(params, batch)
                    total_loss = 0
                    for loss_fn in self.loss_fn:
                        total_loss += loss_fn(logits, jnp.zeros_like(logits))
                    return total_loss

                losses = _eval_step(state, state.params, batch)
                metrics = None
                return losses, metrics
        """
        raise NotImplementedError("Please implement this method")
