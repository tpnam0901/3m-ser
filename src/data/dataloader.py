import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


class IEMOCAPDataset(Dataset):
    def __init__(
        self,
        path: str = "path/to/data.pkl",
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased"),
        audio_max_length: int = 546220,
        text_max_length: int = 100,
    ):
        """Dataset for IEMOCAP

        Args:
            path (str, optional): Path to data.pkl. Defaults to "path/to/data.pkl".
            tokenizer (BertTokenizer, optional): Tokenizer for text. Defaults to BertTokenizer.from_pretrained("bert-base-uncased").
            audio_max_length (int, optional): The maximum length of audio. Defaults to 546220. None for no padding and truncation.
            text_max_length (int, optional): The maximum length of text. Defaults to 100. None for no padding and truncation.
        """
        super(IEMOCAPDataset, self).__init__()
        with open(path, "rb") as train_file:
            self.data_list = pickle.load(train_file)
        self.audio_max_length = audio_max_length
        self.text_max_length = text_max_length
        self.tokenizer = tokenizer

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        fileName, text, sprectrome, label = self.data_list[index].values()
        wav_data, sr = sf.read(fileName, dtype="int16")
        samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        if self.audio_max_length is not None and samples.shape[0] < self.audio_max_length:
            samples = np.pad(samples, (0, self.audio_max_length - samples.shape[0]), "constant")
        elif self.audio_max_length is not None:
            samples = samples[: self.audio_max_length]

        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        if self.text_max_length is not None and len(input_ids) < self.text_max_length:
            input_ids = np.pad(
                input_ids,
                (0, self.text_max_length - len(input_ids)),
                "constant",
                constant_values=self.tokenizer.pad_token_id,
            )
        elif self.text_max_length is not None:
            input_ids = input_ids[: self.text_max_length]

        return torch.from_numpy(np.asarray(input_ids)), torch.from_numpy(samples.astype(np.float32)), torch.tensor(label)

    def __len__(self):
        return len(self.data_list)


def build_train_test_dataset(root: str = "data/") -> Tuple[List, List]:
    """Read train and test data from pickle files

    Args:
        root (str, optional): Path to data directory. Defaults to "data/".
        Your data directory should contain train_data.pkl and test_data.pkl

    Returns:
        Tuple[List, List]: Tuple of train and test data
    """
    with open(os.path.join(root, "train_data.pkl"), "rb") as train_file:
        train_list = pickle.load(train_file)
    with open(os.path.join(root, "test_data.pkl"), "rb") as test_file:
        test_list = pickle.load(test_file)
    return (train_list, test_list)


def build_batch_train_test_dataset(
    root: str = "data/",
    batch_size: int = 64,
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased"),
    audio_max_length: int = 546220,
    text_max_length: int = 100,
) -> Tuple[DataLoader, DataLoader]:
    """Read train and test data from pickle files

    Args:
        root (str, optional): Path to data directory. Defaults to "data/".
        Your data directory should contain train_data.pkl and test_data.pkl

    Returns:
        Tuple[List, List]: Tuple of train and test data
    """
    if batch_size == 1:
        audio_max_length = None
        text_max_length = None
    training_data = IEMOCAPDataset(os.path.join(root, "train_data.pkl"), tokenizer, audio_max_length, text_max_length)
    test_data = IEMOCAPDataset(os.path.join(root, "test_data.pkl"), tokenizer, None, None)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    return (train_dataloader, test_dataloader)
