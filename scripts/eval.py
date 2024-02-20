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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import svm
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from data.dataloader import build_train_test_dataset
from tqdm.auto import tqdm
from models import networks
from configs.base import Config
from collections import Counter


def calculate_accuracy(y_true, y_pred):
    class_weights = {cls: 1.0 / count for cls, count in Counter(y_true).items()}
    bacc = balanced_accuracy_score(
        y_true, y_pred, sample_weight=[class_weights[cls] for cls in y_true]
    )
    acc = accuracy_score(y_true, y_pred)
    return bacc, acc


def eval(cfg, checkpoint_path, all_state_dict=True, cm=False):
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
    bacc, acc = calculate_accuracy(y_actu, y_pred)
    if cm:
        cm = confusion_matrix(y_actu, y_pred)
        print("Confusion Matrix: \n", cm)
        cmn = (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]) * 100

        ax = plt.subplots(figsize=(8, 5.5))[1]
        sns.heatmap(
            cmn,
            cmap="YlOrBr",
            annot=True,
            square=True,
            linecolor="black",
            linewidths=0.75,
            ax=ax,
            fmt=".2f",
            annot_kws={"size": 16},
        )
        ax.set_xlabel("Predicted", fontsize=18, fontweight="bold")
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.set_ticklabels(
            ["Anger", "Happiness", "Sadness", "Neutral"], fontsize=16
        )
        ax.set_ylabel("Ground Truth", fontsize=18, fontweight="bold")
        ax.yaxis.set_ticklabels(
            ["Anger", "Happiness", "Sadness", "Neutral"], fontsize=16
        )
        plt.tight_layout()
        plt.savefig(cfg.name + ".png", format="png", dpi=1200)

    return bacc, acc


def find_checkpoint_folder(path):
    candidate = os.listdir(path)
    if "logs" in candidate and "weights" in candidate and "cfg.log" in candidate:
        return [path]
    list_candidates = []
    for c in candidate:
        list_candidates += find_checkpoint_folder(os.path.join(path, c))
    return list_candidates


def main(root_path, cm):
    logging.info("Finding checkpoints")
    list_checkpoints = find_checkpoint_folder(root_path)
    test_set = args.test_set if args.test_set is not None else "test.pkl"
    csv_path = os.path.basename(root_path) + "{}.csv".format(test_set)
    # field names
    fields = ["BACC", "ACC", "Time", "Model", "Settings"]
    with open(csv_path, "a") as csvfile:
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
            cfg.data_valid = test_set
            bacc, acc = eval(cfg, ckpt_path, cm=cm)
            writer.writerows(
                [
                    {
                        "BACC": round(bacc * 100, 2),
                        "ACC": round(acc * 100, 2),
                        "Time": time,
                        "Model": model_name,
                        "Settings": settings,
                    }
                ]
            )
            logging.info("BACC {:.3f} | ACC {:.3f}".format(bacc, acc))


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

    parser.add_argument(
        "-t",
        "--test_set",
        type=str,
        default=None,
        help="name of testing set. Ex: test.pkl",
    )

    parser.add_argument(
        "-cm",
        "--confusion_matrix",
        action="store_true",
        help="whether to export consution matrix or not",
    )

    return parser.parse_args()


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
        bacc, acc = eval(cfg, ckpt_path, cm=args.confusion_matrix)
        logging.info("BACC {:.3f} | ACC {:.3f}".format(bacc, acc))
    else:
        main(args.checkpoint_path, args.confusion_matrix)
