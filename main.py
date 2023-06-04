import argparse
import logging
import random

import numpy as np
import torch

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

from src.utils.configs import get_options
from src.utils.torch.trainer import Trainer

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main(args):
    opt = get_options(args.config)
    if args.mode == "train":
        trainer = Trainer(opt)
        trainer.train()
    elif args.mode == "infer":
        raise NotImplementedError
    elif args.mode == "eval":
        raise NotImplementedError
    elif args.mode == "visualize":
        raise NotImplementedError


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["train", "infer", "eval", "visualize"])
    parser.add_argument("-cfg", "--config", type=str, default="src/configs/base.py")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
