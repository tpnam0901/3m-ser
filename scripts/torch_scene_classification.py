# %% [markdown]
# ## Import library

# %%
import os
import sys

lib_path = os.path.abspath("").replace("notebooks", "src")
sys.path.append(lib_path)
import pandas
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

from data.dataloader import TorchDataset
from utils.torch.callbacks import CheckpointsCallback
from utils.torch.trainer import TorchTrainer

# %% [markdown]
# ## Hyperparameters

# %%
batch_size = 32
im_size = (224, 224)
epochs = 5

# %% [markdown]
# ## Dataset

# %% [markdown]
# You can download dataset from here: https://www.kaggle.com/datasets/nitishabharathi/scene-classification

# %%
train = pandas.read_csv("train-scene-classification/train.csv")
images_name = train["image_name"].tolist()
labels = train["label"].tolist()
images_name = ["train-scene-classification/train/" + x for x in images_name]
labels = [int(x) for x in labels]
num_classes = len(set(labels))

# %%
transform = transforms.Compose([transforms.ToTensor()])
train_data = TorchDataset(images_name[:-128], labels[:-128], im_size=im_size, transform=transform)
test_data = TorchDataset(images_name[-128:], labels[-128:], im_size=im_size, transform=transform)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

# %% [markdown]
# ## Model

# %%
loss_fn = torch.nn.CrossEntropyLoss()


class SimpleClassification(TorchTrainer):
    def __init__(self, num_classes, im_size=(224, 224), **kwargs):
        super(SimpleClassification, self).__init__(**kwargs)
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(im_size[0] // 8 * im_size[1] // 8 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)

    def train_step(self, batch):
        self.optimizer.zero_grad()

        inputs, labels = batch["inputs"], batch["labels"]

        logits = self(inputs)

        loss = loss_fn(logits, labels)

        loss.backward()

        self.optimizer.step()

        acc = (logits.argmax(dim=1) == labels).float().mean()

        return {"loss": loss.item(), "acc": acc.item()}

    def test_step(self, batch):
        inputs, labels = batch["inputs"], batch["labels"]

        logits = self(inputs)

        loss = loss_fn(logits, labels)

        acc = (logits.argmax(dim=1) == labels).float().mean()

        return {"loss": loss.item(), "acc": acc.item()}


model = SimpleClassification(num_classes, im_size=im_size)

# %% [markdown]
# ## Training

# %%
ckpt_callback = CheckpointsCallback("checkpoints", save_freq=1000, keep_one_only=True)
model.compile(optimizer="sgd")
model.fit(train_dataloader, epochs=epochs, callbacks=[ckpt_callback])

# %% [markdown]
# ## Testing

# %%
model = SimpleClassification.load("checkpoints/checkpoint_.pt")
y_true = []
y_pred = []
for batch in test_dataloader:
    inputs, labels = batch["inputs"], batch["labels"]
    logits = model.predict(inputs)
    y_true += labels.tolist()
    y_pred += logits.argmax(dim=1).tolist()

# %%
from sklearn.metrics import accuracy_score

print(accuracy_score(y_true, y_pred))
