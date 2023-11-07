import os
import pickle
from typing import Dict, List, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, RobertaTokenizer

from configs.base import Config
from torchvggish.vggish_input import waveform_to_examples


class BaseDataset(Dataset):
    def __init__(
        self,
        path: str = "path/to/data.pkl",
        tokenizer: Union[
            BertTokenizer, RobertaTokenizer
        ] = BertTokenizer.from_pretrained("bert-base-uncased"),
        audio_max_length: int = 546220,
        text_max_length: int = 100,
        audio_encoder_type: str = "vggish",
    ):
        """Dataset for IEMOCAP

        Args:
            path (str, optional): Path to data.pkl. Defaults to "path/to/data.pkl".
            tokenizer (BertTokenizer, optional): Tokenizer for text. Defaults to BertTokenizer.from_pretrained("bert-base-uncased").
            audio_max_length (int, optional): The maximum length of audio. Defaults to 546220. None for no padding and truncation.
            text_max_length (int, optional): The maximum length of text. Defaults to 100. None for no padding and truncation.
        """
        super(BaseDataset, self).__init__()
        with open(path, "rb") as train_file:
            self.data_list = pickle.load(train_file)
        self.audio_max_length = audio_max_length
        self.text_max_length = text_max_length
        self.tokenizer = tokenizer
        self.audio_encoder_type = audio_encoder_type

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        audio_path, text, label = self.data_list[index]
        samples = self.__paudio__(audio_path)
        input_ids = self.__ptext__(text)
        label = self.__plabel__(label)

        return input_ids, samples, label

    def __paudio__(self, file_path: int) -> torch.Tensor:
        wav_data, sr = sf.read(file_path, dtype="int16")
        samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        if (
            self.audio_max_length is not None
            and samples.shape[0] < self.audio_max_length
        ):
            samples = np.pad(
                samples, (0, self.audio_max_length - samples.shape[0]), "constant"
            )
        elif self.audio_max_length is not None:
            samples = samples[: self.audio_max_length]

        if self.audio_encoder_type == "vggish" or self.audio_encoder_type == "lstm_mel":
            samples = waveform_to_examples(
                samples, sr, return_tensor=False
            )  # num_samples, 96, 64
            samples = np.expand_dims(samples, axis=1)  # num_samples, 1, 96, 64
        elif self.audio_encoder_type != "panns":
            samples = torchaudio.functional.resample(samples, sr, 16000)

        return torch.from_numpy(samples.astype(np.float32))

    def __ptext__(self, text: str) -> torch.Tensor:
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
        return torch.from_numpy(np.asarray(input_ids))

    def __plabel__(self, label: int) -> torch.Tensor:
        return torch.tensor(label)

    def __len__(self):
        return len(self.data_list)


class IEMOCAPDataset(BaseDataset):
    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        audio_path, text, _, label = self.data_list[index].values()
        samples = self.__paudio__(audio_path)
        input_ids = self.__ptext__(text)
        label = self.__plabel__(label)

        return input_ids, samples, label


class IEMOCAPAudioDataset(BaseDataset):
    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        audio_path, label = self.data_list[index]
        samples = self.__paudio__(audio_path)
        label = self.__plabel__(label)
        # Dummy input_ids for text encoder
        input_ids = torch.zeros(1, dtype=torch.long)
        return input_ids, samples, label


def build_train_test_dataset(opt: Config):
    DATASET_MAP = {
        "IEMOCAP": IEMOCAPDataset,
        "IEMOCAPAudio": IEMOCAPAudioDataset,
        "ESD": BaseDataset,
        "MELD": BaseDataset,
    }

    if opt.text_encoder_type == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif opt.text_encoder_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    else:
        raise NotImplementedError(
            "Tokenizer {} is not implemented".format(opt.text_encoder_type)
        )

    dataset = DATASET_MAP.get(opt.data_name, None)
    if dataset is None:
        raise NotImplementedError(
            "Dataset {} is not implemented, list of available datasets: {}".format(
                opt.data_name, DATASET_MAP.keys()
            )
        )

    audio_max_length = opt.audio_max_length
    text_max_length = opt.text_max_length
    if opt.batch_size == 1:
        audio_max_length = None
        text_max_length = None

    training_data = dataset(
        os.path.join(opt.data_root, "train.pkl"),
        tokenizer,
        audio_max_length,
        text_max_length,
        opt.audio_encoder_type,
    )
    test_data = dataset(
        os.path.join(opt.data_root, "test.pkl"),
        tokenizer,
        None,
        None,
        opt.audio_encoder_type,
    )

    train_dataloader = DataLoader(
        training_data, batch_size=opt.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    return (train_dataloader, test_dataloader)
