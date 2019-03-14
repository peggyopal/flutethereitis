"""
File Name: process_data.py

Authors: Peggy Anderson & Kyle Seidenthal

Date: 27-02-2019

Last Modified: Wed 27 Feb 2019 06:22:03 PM CST

Description: A file to process the data to prepare it for evaluation with the
             models

"""

import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp

import json
import os
import tensorflow as tf
import tqdm         # progress bar
import unittest


# Pooling stuff
NUM_WORKERS = mp.cpu_count()

# TensorFlow Data Stuff
PATH_TO_DATA_FOLDER = os.path.abspath("data/")
PATH_TO_CLEAN_DATA_FOLDER = os.path.abspath("data/clean_data")
PATH_TO_UNBAL_TRAIN_FOLDER = PATH_TO_CLEAN_DATA_FOLDER + "/unbal_train/flute_didgeridoo"
PATH_TO_BAL_TRAIN_FOLDER = PATH_TO_CLEAN_DATA_FOLDER + "/bal_train/flute_didgeridoo"
PATH_TO_EVAL_FOLDER = PATH_TO_CLEAN_DATA_FOLDER + "/eval/flute_didgeridoo"

# Labels Stuff
LABELS_CSV = os.path.join(PATH_TO_DATA_FOLDER, "class_labels_indices.csv")


def _lookup_label_by_index(label_int):
    """
    """
    labels = pd.read_csv(LABELS_CSV)
    return labels["display_name"].loc[labels["index"] == label_int].item()


def _extract_sequence(tf_data):
    """
    """
    sequence = tf.train.SequenceExample()
    sequence.ParseFromString(tf_data)
    return sequence


def _extract_audio_embedding(ae_features):
    """
    Extract Audio Embedding as a List
    """
    audio_embeddings = []

    sess = tf.InteractiveSession()      # Need to start this for .eval()
    for second in range(0, len(ae_features)):
        raw_embedding = tf.decode_raw(ae_features[second].bytes_list.value[0],tf.uint8)
        float_embedding = tf.cast(raw_embedding, tf.float32).eval().tolist()
        # print(type(float_embedding[1]))
        audio_embeddings.append(float_embedding)
    sess.close()

    return audio_embeddings


def _extract_labels(protobuf):
    """
    """
    labels = []
    for i in range(0, len(protobuf)):
        labels.append(protobuf[i])
    return labels


def _process_tensor_file(tf_file_path):
    """
    """
    data = {}

    raw_data = tf.python_io.tf_record_iterator(path=tf_file_path)

    for record in raw_data:
        sequence = _extract_sequence(record)
        video_id = sequence.context.feature["video_id"].bytes_list.value

        labels_protobuf = sequence.context.feature["labels"].int64_list.value
        labels = _extract_labels(labels_protobuf)

        audio_embedding_features = sequence.feature_lists.feature_list["audio_embedding"].feature
        audio_embedding_list = _extract_audio_embedding(audio_embedding_features)
        # print(type(audio_embedding_list[0]))
        data[video_id[0]] = {
                            "labels": labels,
                            "audio_embeddings": audio_embedding_list
                        }

        # print(data)
        # break
    #data_json = json.dumps(data, ensure_ascii=True)
    # print(data_json)
    # return data_json
    return data

def _process_data(dir_path):
    """
    Process the tensorflow records in data_dir and compute statistics about the features contained in them

    :param data_dir: The path to the tensorflow records to process
    :param outname: The path to the directory to save the computations in
    :returns: None
    """

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Get the features from the data
    data = {}
    all_tf_files = os.listdir(dir_path)
    pool = mp.Pool(NUM_WORKERS)
    # with tqdm.tqdm(total=len(all_tf_files), unit="files") as pbar:
        # for tf_file in all_tf_files:
        #     tf_file_path = os.path.join(dir_path, tf_file)
        #     data.update(_process_tensor_file(tf_file_path))
        # pbar.set_postfix(file=tf_file, refresh=False)
        # pbar.update(1)
    result = list(tqdm.tqdm(pool.imap(_process_tensor_file, [os.path.join(dir_path, x) for x in all_tf_files]), total=len(all_tf_files), unit="files"))
        # for tf_file in all_tf_files:
        #     tf_file_path = os.path.join(dir_path, tf_file)

    return result


def get_bal_train():
    """
    """
    return _process_data(PATH_TO_BAL_TRAIN_FOLDER)


def get_unbal_train():
    """
    """
    return _process_data(PATH_TO_UNBAL_TRAIN_FOLDER)


def get_eval():
    """
    """
    return _process_data(PATH_TO_EVAL_FOLDER)


# TESTING STUFF
# if __name__ == "__main__":
    # data = os.path.join(PATH_TO_BAL_TRAIN_FOLDER, "_5.tfrecord")
    #
    # dict = _process_data(data)
    # print(dict)
    # print(type(dict[b'_5w5TVK5B90']['audio_embeddings']))

    # dict = get_bal_train()
    # print(dict)
