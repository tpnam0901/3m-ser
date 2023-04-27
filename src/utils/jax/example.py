from flax import linen as nn

import jax
import jax.numpy as jnp
from callbacks import CheckpointCallback
from trainer import BaseTrainer


class SimpleMLP(nn.Module):
    def setup(self):
        self.dense1 = nn.Dense(features=256)
        self.dense2 = nn.Dense(features=1)

    def __call__(self, input):
        x = self.dense1(input)
        x = nn.relu(x)
        return self.dense2(x)


class Trainer(BaseTrainer):
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


def loss_fn(logits, batch):
    return jnp.mean((logits - batch) ** 2)


trainer = Trainer(SimpleMLP(), jnp.ones((1, 10)), checkpoints_dir="checkpoints")
trainer.summary()
trainer.compile(loss_fn=loss_fn, metrics=["accuracy"], optimizer="sgd")
trainer.fit(10, jnp.ones((100, 10)), jnp.ones((50, 10)), verbose=1, callbacks=[CheckpointCallback(50, 3)])
