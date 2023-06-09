import os
import pickle
from typing import List, Tuple


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
