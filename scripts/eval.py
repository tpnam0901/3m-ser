import logging
import os
import sys

lib_path = os.path.abspath("").replace("scripts", "src")
sys.path.append(lib_path)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
import csv
import argparse
import torch
from sklearn import svm
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
)
from data.dataloader import build_train_test_dataset
from tqdm.auto import tqdm
from models import networks
from configs.base import Config

from collections import Counter


def calculate_accuracy(y_pred, y_true):
    class_weights = {cls: 1.0 / count for cls, count in Counter(y_true).items()}
    wa = balanced_accuracy_score(
        y_true, y_pred, sample_weight=[class_weights[cls] for cls in y_true]
    )
    ua = accuracy_score(y_true, y_pred)
    return ua, wa


def eval(cfg, checkpoint_path, all_state_dict=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = getattr(networks, cfg.model_type)(cfg)
    network.to(device)

    # Build dataset
    _, test_ds = build_train_test_dataset(cfg)
    weight = torch.load(checkpoint_path, map_location=torch.device(device))
    if all_state_dict:
        weight = weight["state_dict_network"]
    else:
        weight = weight.state_dict()

    network.load_state_dict(weight)
    network.eval()
    network.to(device)

    y_actu = []
    y_pred = []

    for every_test_list in tqdm(test_ds):
        input_ids, audio, label = every_test_list
        input_ids = input_ids.to(device)
        audio = audio.to(device)
        label = label.to(device)
        with torch.no_grad():
            output = network(input_ids, audio)[0]
            _, preds = torch.max(output, 1)
            y_actu.append(label.detach().cpu().numpy()[0])
            y_pred.append(preds.detach().cpu().numpy()[0])
    bacc = balanced_accuracy_score(y_actu, y_pred)
    ua, wa = calculate_accuracy(y_actu, y_pred)
    return bacc, ua, wa


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ckpt", "--checkpoint_path", type=str, help="path to checkpoint folder"
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="whether to travel child folder or not",
    )
    return parser.parse_args()


def find_checkpoint_folder(path):
    candidate = os.listdir(path)
    if "logs" in candidate and "weights" in candidate and "cfg.log" in candidate:
        return [path]
    list_candidates = []
    for c in candidate:
        list_candidates += find_checkpoint_folder(os.path.join(path, c))
    return list_candidates


def main(root_path):
    logging.info("Finding checkpoints")
    list_checkpoints = find_checkpoint_folder(root_path)
    csv_path = os.path.basename(root_path) + ".csv"
    # field names
    fields = ["Model", "Settings", "Time", "BACC", "UA", "WA"]
    with open(csv_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for ckpt in list_checkpoints:
            meta_info = ckpt.split("/")
            time = meta_info[-1]
            settings = meta_info[-2]
            model_name = meta_info[-3]
            logging.info("Evaluating: {}/{}/{}".format(model_name, settings, time))
            cfg_path = os.path.join(ckpt, "cfg.log")
            ckpt_path = os.path.join(ckpt, "weights/best_acc/checkpoint_0_0.pt")
            cfg = Config()
            cfg.load(cfg_path)
            # Change to test set
            cfg.data_valid = "test.pkl"
            bacc, ua, wa = eval(cfg, ckpt_path)
            writer.writerows(
                [
                    {
                        "Model": model_name,
                        "Settings": settings,
                        "Time": time,
                        "BACC": round(bacc, 3),
                        "UA": round(ua, 3),
                        "WA": round(wa, 3),
                    }
                ]
            )
            logging.info("BACC {:.3f} | UA {:.3f} | WA {:.3f}".format(bacc, ua, wa))


if __name__ == "__main__":
    args = arg_parser()
    if not args.recursive:
        cfg_path = os.path.join(args.checkpoint_path, "cfg.log")
        ckpt_path = os.path.join(
            args.checkpoint_path, "weights/best_acc/checkpoint_0_0.pt"
        )
        cfg = Config()
        cfg.load(cfg_path)
        # Change to test set
        cfg.data_valid = "test.pkl"
        bacc, ua, wa = eval(cfg, ckpt_path)
        logging.info("BACC {:.3f} | UA {:.3f} | WA {:.3f}".format(bacc, ua, wa))
    else:
        main(args.checkpoint_path)
