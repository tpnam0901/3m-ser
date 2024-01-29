import os
import pickle
import re
from typing import Dict, List, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer,
    RobertaTokenizer,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

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

    def _text_preprocessing(self, text):
        """
        - Remove entity mentions (eg. '@united')
        - Correct errors (eg. '&amp;' to '&')
        @param    text (str): a string to be processed.
        @return   text (Str): the processed string.
        """
        # Remove '@name'
        text = re.sub("[\(\[].*?[\)\]]", "", text)

        # Replace '&amp;' with '&'
        text = re.sub(" +", " ", text).strip()

        # Normalize and clean up text; order matters!
        try:
            text = " ".join(text.split())  # clean up whitespaces
        except:
            text = "NULL"

        # Convert empty string to NULL
        if not text.strip():
            text = "NULL"

        return text

    def __ptext__(self, text: str) -> torch.Tensor:
        text = self._text_preprocessing(text)
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


#### For MSER ####
class IEMOCAPDatasetMSER(BaseDataset):
    def __init__(self, audio_processor, **kwargs):
        super(IEMOCAPDatasetMSER, self).__init__(**kwargs)
        self.audio_length = 3000
        self.feature_name = "fbank"
        self.feature_dim = 40
        self.audio_processor = audio_processor

        vocabulary_chars_str = "".join(
            t for t in self.audio_processor.tokenizer.get_vocab().keys() if len(t) == 1
        )
        self.vocabulary_text_cleaner = re.compile(  # remove characters not in vocabulary
            f"[^\s{re.escape(vocabulary_chars_str)}]",  # allow space in addition to chars in vocabulary
            flags=re.IGNORECASE if self.audio_processor.tokenizer.do_lower_case else 0,
        )

    def get_full_path(self, filename):
        filename = filename.split("_")
        session = {
            "Ses01F": "Session1",
            "Ses01M": "Session1",
            "Ses02F": "Session2",
            "Ses02M": "Session2",
            "Ses03F": "Session3",
            "Ses03M": "Session3",
            "Ses04F": "Session4",
            "Ses04M": "Session4",
            "Ses05F": "Session5",
            "Ses05M": "Session5",
        }[filename[0]]

        full_path = os.path.join(
            "IEMOCAP_full_release",
            session,
            "sentences/wav",
            str("_").join(filename[:-1]),
            str("_").join(filename) + ".wav",
        )
        return full_path

    def text_preprocessing(self, text):
        """
        - Remove entity mentions (eg. '@united')
        - Correct errors (eg. '&amp;' to '&')
        @param    text (str): a string to be processed.
        @return   text (Str): the processed string.
        """
        # Remove '@name'
        text = re.sub("[\(\[].*?[\)\]]", "", text)

        # Replace '&amp;' with '&'
        text = re.sub(" +", " ", text).strip()

        # Normalize and clean up text; order matters!
        try:
            text = " ".join(text.split())  # clean up whitespaces
        except:
            text = "NULL"

        # Convert empty string to NULL
        if not text.strip():
            text = "NULL"

        return text

    def prepare_example(self, text, vocabulary_text_cleaner):
        # Normalize and clean up text; order matters!
        try:
            text = " ".join(text.split())  # clean up whitespaces
        except:
            text = "NULL"
        updated_text = text
        updated_text = vocabulary_text_cleaner.sub("", updated_text)
        if updated_text != text:
            return re.sub(" +", " ", updated_text).strip()
        else:
            return re.sub(" +", " ", text).strip()

    def label2idx(label):
        label2idx = {"hap": 0, "ang": 1, "neu": 2, "sad": 3, "exc": 0}
        return label2idx[label]

    def __getitem__(self, index):
        audio_path, bert_text, label = self.data_list[index]

        # audio_name = data["fileName"]
        # bert_text = data["text"]
        # label = data["label"]
        # ------------- extract the audio features -------------#
        # audio_path = self.get_full_path(audio_path)
        # wave, sr = librosa.core.load(audio_path, sr=None)

        # # precautionary measure to fit in a 24GB gpu, feel free to comment the next 2 lines
        # if len(wave) > 210000:
        #     wave = wave[:210000]
        wave = self.__paudio__(audio_path)

        audio_length = len(wave)
        bert_text = self.text_preprocessing(bert_text)

        # # ------------- labels -------------#
        # label = self.label2idx(label)

        # ------------- wrap up all the output info the dict format -------------#
        return {
            "audio_input": wave,
            "text_input": bert_text,
            "audio_length": audio_length,
            "label": label,
        }


def build_mser_dataset(cfg: Config):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-base"
    )
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        "facebook/wav2vec2-base",
        do_lower_case=False,
        word_delimiter_token="|",
    )
    audio_processor = Wav2Vec2Processor(feature_extractor, tokenizer)

    def collate(sample_list):
        batch_audio = np.stack([x["audio_input"] for x in sample_list])
        batch_bert_text = [x["text_input"] for x in sample_list]

        # ----------------tokenize and pad the audio----------------------#
        batch_audio = audio_processor(batch_audio, sampling_rate=16000).input_values

        batch_audio = [{"input_values": audio} for audio in batch_audio]
        batch_audio = audio_processor.pad(
            batch_audio,
            padding=True,
            return_tensors="pt",
        )

        # ----------------tokenize and pad the text----------------------#
        batch_text = tokenizer(
            batch_bert_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        batch_text_inputids = batch_text["input_ids"]
        batch_text_attention = batch_text["attention_mask"]

        # ----------------tokenize and pad the extras----------------------#
        audio_length = torch.LongTensor([x["audio_length"] for x in sample_list])
        batch_label = torch.tensor([x["label"] for x in sample_list], dtype=torch.long)

        target_labels = []

        for label_idx in range(4):
            temp_labels = []
            for idx, _label in enumerate(batch_label):
                if _label == label_idx:
                    temp_labels.append(idx)

            target_labels.append(torch.LongTensor(temp_labels[:]))

        return (
            (batch_text_inputids, batch_text_attention),
            (batch_audio["input_values"], audio_length),
            (batch_label, target_labels),
        )

    audio_max_length = cfg.audio_max_length
    text_max_length = cfg.text_max_length

    train_dataset = IEMOCAPDatasetMSER(
        audio_processor,
        path=os.path.join(cfg.data_root, "train.pkl"),
        tokenizer=tokenizer,
        audio_max_length=audio_max_length,
        text_max_length=text_max_length,
        audio_encoder_type=cfg.audio_encoder_type,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    test_set = cfg.data_valid if cfg.data_valid is not None else "test.pkl"
    valid_dataset = IEMOCAPDatasetMSER(
        audio_processor,
        path=os.path.join(cfg.data_root, test_set),
        tokenizer=tokenizer,
        audio_max_length=audio_max_length,
        text_max_length=text_max_length,
        audio_encoder_type=cfg.audio_encoder_type,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        collate_fn=collate,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    return (train_loader, valid_loader)


def build_train_test_dataset(cfg: Config):
    DATASET_MAP = {
        "IEMOCAP": BaseDataset,
        "IEMOCAPAudio": IEMOCAPAudioDataset,
        "ESD": BaseDataset,
        "MELD": BaseDataset,
        "IEMOCAP_MSER": build_mser_dataset,
        "MELD_MSER": build_mser_dataset,
    }

    dataset = DATASET_MAP.get(cfg.data_name, None)
    if dataset is None:
        raise NotImplementedError(
            "Dataset {} is not implemented, list of available datasets: {}".format(
                cfg.data_name, DATASET_MAP.keys()
            )
        )
    if cfg.data_name in ["IEMOCAP_MSER", "MELD_MSER"]:
        return dataset(cfg)

    if cfg.text_encoder_type == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif cfg.text_encoder_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    else:
        raise NotImplementedError(
            "Tokenizer {} is not implemented".format(cfg.text_encoder_type)
        )

    audio_max_length = cfg.audio_max_length
    text_max_length = cfg.text_max_length
    if cfg.batch_size == 1:
        audio_max_length = None
        text_max_length = None

    training_data = dataset(
        path=os.path.join(cfg.data_root, "train.pkl"),
        tokenizer=tokenizer,
        audio_max_length=audio_max_length,
        text_max_length=text_max_length,
        audio_encoder_type=cfg.audio_encoder_type,
    )

    test_set = cfg.data_valid if cfg.data_valid is not None else "test.pkl"
    test_data = dataset(
        path=os.path.join(cfg.data_root, test_set),
        tokenizer=tokenizer,
        audio_max_length=None,
        text_max_length=None,
        audio_encoder_type=cfg.audio_encoder_type,
    )

    train_dataloader = DataLoader(
        training_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return (train_dataloader, test_dataloader)
