import logging
import os
import sys

lib_path = os.path.abspath("").replace("scripts", "src")
sys.path.append(lib_path)

import argparse
import datetime
import random

import numpy as np
import torch
from torch import nn, optim
from transformers import BertTokenizer

from configs.base import Config
from data.dataloader import build_train_test_dataset
from models import networks
from trainer import Trainer
from utils.configs import get_options
from utils.torch.callbacks import CheckpointsCallback

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def main(opt: Config):
    # Model
    try:
        network = getattr(networks, opt.model_type)(
            num_classes=opt.num_classes,
            num_attention_head=opt.num_attention_head,
            dropout=opt.dropout,
        )
    except AttributeError:
        raise NotImplementedError("Model {} is not implemented".format(opt.model_type))

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Preapre the checkpoint directory
    opt.checkpoint_dir = checkpoint_dir = os.path.join(
        os.path.abspath(opt.checkpoint_dir), opt.model_type, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    log_dir = os.path.join(checkpoint_dir, "logs")
    weight_dir = os.path.join(checkpoint_dir, "weights")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    opt.save(opt)

    # Build dataset
    train_ds, test_ds = build_train_test_dataset(opt.data_root)

    trainer = Trainer(network=network, tokenizer=tokenizer, criterion=nn.CrossEntropyLoss(), log_dir=opt.checkpoint_dir)
    # Build optimizer and criterion
    optimizer = optim.Adam(params=trainer.network.parameters(), lr=opt.learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.learning_rate_step_size, gamma=opt.learning_rate_gamma)

    ckpt_callback = CheckpointsCallback(
        checkpoint_dir=weight_dir,
        save_freq=opt.save_freq,
        max_to_keep=3,
        save_best_val=True,
    )
    trainer.compile(optimizer=optimizer, scheduler=lr_scheduler)
    trainer.fit(train_ds, opt.num_epochs, test_ds, callbacks=[ckpt_callback])


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, default="../src/configs/base.py")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    opt = get_options(args.config)
    main(opt)
