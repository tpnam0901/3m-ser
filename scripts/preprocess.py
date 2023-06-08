import argparse
import csv
import logging
import os
import pickle
import random
import sys
import time

import pandas as pd
import torch
from torchvggish import vggish_input

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def file_search(dirname, ret, list_avoid_dir=[]):
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)

        if os.path.isdir(full_filename):
            if full_filename.split("/")[-1] in list_avoid_dir:
                continue
            else:
                file_search(full_filename, ret, list_avoid_dir)
        else:
            ret.append(full_filename)


def extract_trans(list_in_file, out_file):
    lines = []
    for in_file in list_in_file:
        cnt = 0
        with open(in_file, "r") as f1:
            lines = f1.readlines()
        with open(out_file, "a") as f2:
            csv_writer = csv.writer(f2)
            lines = sorted(lines)  # sort based on first element
            for line in lines:
                name = line.split(":")[0].split(" ")[0].strip()
                # unwanted case
                if name[:3] != "Ses":  # noise transcription such as reply  M: sorry
                    continue
                elif name[-3:-1] == "XX":  # we don't have matching pair in label
                    continue
                trans = line.split(":")[1].strip()
                cnt += 1
                csv_writer.writerow([name, trans])


def find_category(lines):
    is_target = True

    id = ""
    c_label = ""
    list_ret = []

    list_category = ["ang", "hap", "sad", "neu", "fru", "exc", "fea", "sur", "dis", "oth", "xxx"]

    category = {}
    for c_type in list_category:
        if category.__contains__(c_type):
            pass
        else:
            category[c_type] = len(category)

    for line in lines:
        if is_target == True:
            try:
                id = line.split("\t")[1].strip()  #  extract ID
                c_label = line.split("\t")[2].strip()  #  extract category
                if not category.__contains__(c_label):
                    print("ERROR nokey ", c_label)
                    sys.exit()

                list_ret.append([id, c_label])
                is_target = False

            except:
                print("ERROR ", line)
                sys.exit()

        else:
            if line == "\n":
                is_target = True

    return list_ret


def extract_labels(list_in_file, out_file):
    id = ""
    lines = []
    list_ret = []

    for in_file in list_in_file:
        with open(in_file, "r") as f1:
            lines = f1.readlines()
            lines = lines[2:]  # remove head
            list_ret = find_category(lines)

        list_ret = sorted(list_ret)  # sort based on first element

        with open(out_file, "a") as f2:
            csv_writer = csv.writer(f2)
            csv_writer.writerows(list_ret)


def main(args):
    list_files = []
    output_dir = args.output_dir
    data_root = args.data_root
    os.makedirs(output_dir, exist_ok=True)

    assert os.path.exists(data_root), "data root does not exist"

    for x in range(5):
        sess_name = "Session" + str(x + 1)

        path = os.path.join(data_root, sess_name, "dialog", "transcriptions") + "/"
        file_search(path, list_files)
        list_files = sorted(list_files)

        print(sess_name + ", #sum files: " + str(len(list_files)))

    extract_trans(list_files, os.path.join(output_dir, "processed_trans.csv"))

    # read contents of csv file
    file = pd.read_csv(os.path.join(output_dir, "processed_trans.csv"))

    # adding header
    headerList = ["sessionID", "text"]

    # converting data frame to csv
    file.to_csv(os.path.join(output_dir, "processed_trans_head.csv"), header=headerList, index=False)

    list_category = ["ang", "hap", "sad", "neu", "fru", "exc", "fea", "sur", "dis", "oth", "xxx"]

    category = {}
    for c_type in list_category:
        if category.__contains__(c_type):
            pass
        else:
            category[c_type] = len(category)

    # [schema] ID, label [csv]

    list_files = []
    list_avoid_dir = ["Attribute", "Categorical", "Self-evaluation"]

    for x in range(5):
        sess_name = "Session" + str(x + 1)

        path = os.path.join(data_root, sess_name, "dialog", "EmoEvaluation") + "/"
        file_search(path, list_files, list_avoid_dir)
        list_files = sorted(list_files)

        print(sess_name + ", #sum files: " + str(len(list_files)))

    extract_labels(list_files, os.path.join(output_dir, "processed_labels.csv"))

    # read contents of csv file
    file = pd.read_csv(os.path.join(output_dir, "processed_labels.csv"))

    # adding header
    headerList = ["sessionID", "label"]

    # converting data frame to csv
    file.to_csv(os.path.join(output_dir, "processed_labels_head.csv"), header=headerList, index=False)

    dfl = pd.read_csv(os.path.join(output_dir, "processed_labels_head.csv"))
    dfl.loc[dfl["label"] == "ang", "label"] = 0
    dfl.loc[dfl["label"] == "hap", "label"] = 1
    dfl.loc[dfl["label"] == "exc", "label"] = 1
    dfl.loc[dfl["label"] == "sad", "label"] = 2
    dfl.loc[dfl["label"] == "neu", "label"] = 3
    dfl.loc[dfl["label"] == "fru", "label"] = -1
    dfl.loc[dfl["label"] == "fea", "label"] = -1
    dfl.loc[dfl["label"] == "sur", "label"] = -1
    dfl.loc[dfl["label"] == "dis", "label"] = -1
    dfl.loc[dfl["label"] == "oth", "label"] = -1
    dfl.loc[dfl["label"] == "xxx", "label"] = -1
    dfl.head(10)

    dfl.to_csv(os.path.join(output_dir, "processed_digital_labels_head.csv"), index=False)

    # reading two csv files
    data1 = pd.read_csv(os.path.join(output_dir, "processed_trans_head.csv"))
    data2 = pd.read_csv(os.path.join(output_dir, "processed_digital_labels_head.csv"))

    # using merge function by setting how='inner'
    translabels = pd.merge(data1, data2, on="sessionID", how="inner")

    translabels.to_csv(os.path.join(output_dir, "processed_trans_labels_head.csv"), index=False)

    list_files = []
    for x in range(5):
        sess_name = "Session" + str(x + 1)
        path = os.path.join(data_root, sess_name, "sentences", "wav") + "/"
        file_search(path, list_files)
        list_files = sorted(list_files)
        print(sess_name + ", #sum files: " + str(len(list_files)))

    df = pd.read_csv(os.path.join(output_dir, "processed_trans_labels_head.csv"))

    no_rows = len(list_files)
    # cnt = 0
    index = 0
    sprectrogram_shape = []
    docs = []
    bookmark = 0
    extraLabel = 0
    for everyFile in list_files:
        if everyFile.split("/")[-1].endswith(".wav"):
            filename = everyFile.split("/")[-1].strip(".wav")
            print(filename)
            lable = df.loc[df["sessionID"] == filename]["label"].values[0]
            text = df.loc[df["sessionID"] == filename]["text"].values[0]
            # print('label',lable)
            if lable != -1:
                input_batch = vggish_input.wavfile_to_examples(everyFile)
                # print(input_batch.size())
                if (len(input_batch.size()) < 4) or (input_batch.size(dim=0) <= 1):
                    # print("Wrong", input_batch.size())
                    continue
                elif (len(input_batch.size()) == 4) and (input_batch.size(dim=0) > 1):
                    # print("Correct", input_batch.size())
                    docs.append(
                        {
                            "fileName": everyFile.split("/")[-1].strip(".wav"),
                            "text": text,
                            "sprectrome": input_batch,
                            "label": lable,
                        }
                    )
                    index += 1
                    # print('index',index)
                    # cnt+=1
                    # if cnt > 100:
                    # break
            else:
                extraLabel = extraLabel + 1
                # print('extraLabel',extraLabel)

    random.shuffle(docs)
    random.shuffle(docs)
    random.shuffle(docs)
    total_length = len(docs)
    train_length = int(0.8 * total_length)
    train_list = docs[0:train_length]
    test_list = docs[train_length:]
    print("no of items for train ", len(train_list))
    print("no of items for test ", len(test_list))
    # no of items for train  4424
    # no of items for test  1107

    # Write data
    train_file = open(os.path.join(output_dir, "train_data.pkl"), "wb")

    pickle.dump(train_list, train_file)

    train_file.close()

    test_file = open(os.path.join(output_dir, "test_data.pkl"), "wb")

    pickle.dump(test_list, test_file)

    test_file.close()


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, help="Path to folder containing IEMOCAP data", default="IEMOCAP_full_release")
    parser.add_argument("--output_dir", type=str, help="Path to folder to save processed data", default="data")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
