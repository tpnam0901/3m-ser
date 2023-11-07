import argparse
import glob
import logging
import os
import pickle
import random

import pandas as pd
import soundfile as sf
import tqdm
from moviepy.editor import VideoFileClip
from sklearn.model_selection import train_test_split

LABEL_MAP = {
    "ang": 0,
    "hap": 1,
    "sad": 2,
    "neu": 3,
}

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def export_mp4_to_audio(
    mp4_file: str,
    wav_file: str,
    verbose: bool = False,
):
    """Convert mp4 file to wav file

    Args:
        mp4_file (str): Path to mp4 input file
        wav_file (str): Path to wav output file
        verbose (bool, optional): Whether to print ffmpeg output. Defaults to False.
    """
    try:
        video = VideoFileClip(mp4_file)
    except:
        logging.warning(f"Failed to load {mp4_file}")
        return 0
    audio = video.audio
    audio.write_audiofile(wav_file, verbose=verbose)
    return 1


def preprocess_IEMOCAP(args):
    data_root = args.data_root
    ignore_length = args.ignore_length

    session_id = list(range(1, 6))

    samples = []
    labels = []
    iemocap2label = LABEL_MAP
    iemocap2label.update({"exc": 1})

    for sess_id in tqdm.tqdm(session_id):
        sess_path = os.path.join(data_root, "Session{}".format(sess_id))
        sess_autio_root = os.path.join(sess_path, "sentences/wav")
        sess_label_root = os.path.join(sess_path, "dialog/EmoEvaluation")
        label_paths = glob.glob(os.path.join(sess_label_root, "*.txt"))
        for l_path in label_paths:
            with open(l_path, "r") as f:
                label = f.read().split("\n")
                for l in label:
                    if str(l).startswith("["):
                        data = l.split()
                        wav_folder = data[3][:-5]
                        wav_name = data[3] + ".wav"
                        emo = data[4]
                        wav_path = os.path.join(sess_autio_root, wav_folder, wav_name)
                        wav_data, _ = sf.read(wav_path, dtype="int16")
                        # Ignore samples with length < ignore_length
                        if len(wav_data) < ignore_length:
                            logging.warning(f"Ignoring sample {wav_path} with length {len(wav_data)}")
                            continue
                        emo = iemocap2label.get(emo, None)
                        if emo is not None:
                            samples.append((wav_path, emo))
                            labels.append(emo)

    # Shuffle and split
    temp = list(zip(samples, labels))
    random.Random(args.seed).shuffle(temp)
    samples, labels = zip(*temp)
    train_samples, test_samples, _, _ = train_test_split(samples, labels, test_size=0.2, random_state=args.seed)

    # Save data
    os.makedirs(args.dataset + "_preprocessed", exist_ok=True)
    with open(os.path.join(args.dataset + "_preprocessed", "train.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(args.dataset + "_preprocessed", "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    logging.info(f"Train samples: {len(train_samples)}")
    logging.info(f"Test samples: {len(test_samples)}")
    logging.info(f"Saved to {args.dataset + '_preprocessed'}")
    logging.info("Preprocessing finished successfully")


def preprocess_MELD(args):
    meld2label = {
        "sadness": "sad",
        "neutral": "neu",
        "joy": "hap",
        "anger": "ang",
    }
    data_root = args.data_root
    ignore_length = args.ignore_length

    train_df = pd.read_csv(os.path.join(data_root, "train_sent_emo.csv"))
    dev_df = pd.read_csv(os.path.join(data_root, "dev_sent_emo.csv"))
    test_df = pd.read_csv(os.path.join(data_root, "test_sent_emo.csv"))

    logging.info("Preprocessing train set")
    train_samples = []
    for row in tqdm.tqdm(train_df.iterrows()):
        emotion = row[1]["Emotion"]
        target = meld2label.get(emotion, None)
        if target is not None:
            dia = row[1]["Dialogue_ID"]
            utt = row[1]["Utterance_ID"]
            inp_path = os.path.join(data_root, "train_splits", f"dia{dia}_utt{utt}.mp4")
            out_path = os.path.join(data_root, "train_splits", f"dia{dia}_utt{utt}.wav")
            if export_mp4_to_audio(inp_path, out_path):
                wav_data, sr = sf.read(out_path, dtype="int16")
                # Ignore samples with length < ignore_length
                if len(wav_data) < ignore_length:
                    logging.warning(f"Ignoring sample {out_path} with length {len(wav_data)}")
                    continue
                train_samples.append((os.path.abspath(out_path), row[1]["Utterance"], LABEL_MAP[target]))

    logging.info("Preprocessing dev set")
    dev_samples = []
    for row in tqdm.tqdm(dev_df.iterrows()):
        emotion = row[1]["Emotion"]
        target = meld2label.get(emotion, None)
        if target is not None:
            dia = row[1]["Dialogue_ID"]
            utt = row[1]["Utterance_ID"]
            inp_path = os.path.join(data_root, "dev_splits_complete", f"dia{dia}_utt{utt}.mp4")
            out_path = os.path.join(data_root, "dev_splits_complete", f"dia{dia}_utt{utt}.wav")
            if export_mp4_to_audio(inp_path, out_path):
                wav_data, sr = sf.read(out_path, dtype="int16")
                # Ignore samples with length < ignore_length
                if len(wav_data) < ignore_length:
                    logging.warning(f"Ignoring sample {out_path} with length {len(wav_data)}")
                    continue
                dev_samples.append((os.path.abspath(out_path), row[1]["Utterance"], LABEL_MAP[target]))

    logging.info("Preprocessing test set")
    test_samples = []
    for row in tqdm.tqdm(test_df.iterrows()):
        emotion = row[1]["Emotion"]
        target = meld2label.get(emotion, None)
        if target is not None:
            dia = row[1]["Dialogue_ID"]
            utt = row[1]["Utterance_ID"]
            inp_path = os.path.join(data_root, "output_repeated_splits_test", f"dia{dia}_utt{utt}.mp4")
            out_path = os.path.join(data_root, "output_repeated_splits_test", f"dia{dia}_utt{utt}.wav")
            if export_mp4_to_audio(inp_path, out_path):
                wav_data, sr = sf.read(out_path, dtype="int16")
                # Ignore samples with length < ignore_length
                if len(wav_data) < ignore_length:
                    logging.warning(f"Ignoring sample {out_path} with length {len(wav_data)}")
                    continue
                test_samples.append((os.path.abspath(out_path), row[1]["Utterance"], LABEL_MAP[target]))

    # Save data
    os.makedirs(args.dataset + "_preprocessed", exist_ok=True)
    with open(os.path.join(args.dataset + "_preprocessed", "train.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(args.dataset + "_preprocessed", "dev.pkl"), "wb") as f:
        pickle.dump(dev_samples, f)
    with open(os.path.join(args.dataset + "_preprocessed", "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    logging.info(f"Train samples: {len(train_samples)}")
    logging.info(f"Dev samples: {len(dev_samples)}")
    logging.info(f"Test samples: {len(test_samples)}")
    logging.info(f"Saved to {args.dataset + '_preprocessed'}")
    logging.info("Preprocessing finished successfully")


def preprocess_ESD(args):
    esd2label = {
        "Angry": "ang",
        "Happy": "hap",
        "Neutral": "neu",
        "Sad": "sad",
    }

    directory = glob.glob(args.data_root + "/*")
    samples = []
    labels = []

    # Loop through all folders
    for dir in tqdm.tqdm(directory):
        # Read label file
        label_path = os.path.join(dir, dir.split("/")[-1] + ".txt")
        with open(label_path, "r") as f:
            label = f.read().strip().splitlines()
        # Extract samples from label file
        for l in label:
            filename, transcript, emotion = l.split("\t")
            target = esd2label.get(emotion, None)
            if target is not None:
                samples.append((os.path.abspath(os.path.join(dir, emotion, filename + ".wav")), transcript, LABEL_MAP[target]))
                # Labels are use for splitting
                labels.append(LABEL_MAP[target])

    # Shuffle and split
    temp = list(zip(samples, labels))
    random.Random(args.seed).shuffle(temp)
    samples, labels = zip(*temp)
    train_samples, test_samples, _, _ = train_test_split(samples, labels, test_size=0.2, random_state=args.seed)

    # Save data
    os.makedirs(args.dataset + "_preprocessed", exist_ok=True)
    with open(os.path.join(args.dataset + "_preprocessed", "train.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(args.dataset + "_preprocessed", "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    logging.info(f"Train samples: {len(train_samples)}")
    logging.info(f"Test samples: {len(test_samples)}")
    logging.info(f"Saved to {args.dataset + '_preprocessed'}")
    logging.info("Preprocessing finished successfully")


def main(args):
    preprocess_fn = {
        "IEMOCAP": preprocess_IEMOCAP,
        "ESD": preprocess_ESD,
        "MELD": preprocess_MELD,
    }

    preprocess_fn[args.dataset](args)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataset", type=str, default="ESD", choices=["IEMOCAP", "ESD", "MELD"])
    parser.add_argument("-dr", "--data_root", type=str, help="Path to folder containing IEMOCAP data", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore_length", type=int, default=0, help="Ignore samples with length < ignore_length")

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
