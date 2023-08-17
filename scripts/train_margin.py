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
from transformers import BertTokenizer, RobertaTokenizer

from configs.base import Config
from data.dataloader import build_train_test_dataset
from models import losses, networks
from trainer import MarginTrainer as Trainer
from utils.configs import get_options
from utils.torch.callbacks import CheckpointsCallback

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(opt: Config):
    logging.info("Initializing model...")
    # Model
    try:
        network = getattr(networks, opt.model_type)(
            num_classes=opt.num_classes,
            num_attention_head=opt.num_attention_head,
            dropout=opt.dropout,
            text_encoder_type=opt.text_encoder_type,
            text_encoder_dim=opt.text_encoder_dim,
            text_unfreeze=opt.text_unfreeze,
            audio_encoder_type=opt.audio_encoder_type,
            audio_encoder_dim=opt.audio_encoder_dim,
            audio_unfreeze=opt.audio_unfreeze,
            audio_norm_type=opt.audio_norm_type,
            fusion_head_output_type=opt.fusion_head_output_type,
        )
        network.to(device)
    except AttributeError:
        raise NotImplementedError("Model {} is not implemented".format(opt.model_type))

    logging.info("Initializing checkpoint directory and dataset...")
    if opt.text_encoder_type == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif opt.text_encoder_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    else:
        raise NotImplementedError("Tokenizer {} is not implemented".format(opt.text_encoder_type))

    # Preapre the checkpoint directory
    opt.checkpoint_dir = checkpoint_dir = os.path.join(
        os.path.abspath(opt.checkpoint_dir),
        opt.name,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    log_dir = os.path.join(checkpoint_dir, "logs")
    weight_dir = os.path.join(checkpoint_dir, "weights")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    opt.save(opt)

    # Build dataset
    train_ds, test_ds = build_train_test_dataset(
        opt.data_root,
        opt.batch_size,
        tokenizer,
        opt.audio_max_length,
        text_max_length=opt.text_max_length,
        audio_encoder_type=opt.audio_encoder_type,
    )

    logging.info("Initializing trainer...")
    criterion = losses.CombinedMarginLoss(
        in_features=opt.feat_dim,
        out_features=opt.num_classes,
        s=opt.margin_loss_scale,
        m1=opt.margin_loss_m1,
        m2=opt.margin_loss_m2,
        m3=opt.margin_loss_m2,
    )
    criterion.to(device)

    trainer = Trainer(
        network=network,
        criterion=criterion,
        log_dir=opt.checkpoint_dir,
    )
    logging.info("Start training...")
    # Build optimizer and criterion
    optimizer = optim.Adam(params=trainer.network.parameters(), lr=opt.learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.learning_rate_step_size, gamma=opt.learning_rate_gamma)

    ckpt_callback = CheckpointsCallback(
        checkpoint_dir=weight_dir,
        save_freq=opt.save_freq,
        max_to_keep=opt.max_to_keep,
        save_best_val=opt.save_best_val,
        save_all_states=opt.save_all_states,
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
