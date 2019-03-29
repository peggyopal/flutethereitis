"""
File Name: process_data.py

Authors: Peggy Anderson & Kyle Seidenthal

Date: 11-03-2019

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
    Given a integer value for a label lookup the string value

    :param label_int: A label as an integer value
    :returns: String representation of a label if found, if index is not valid
              the program will continue as normal, but a value in the list is
              yelling at you
    """
    label = pd.read_csv(LABELS_CSV)
    try:
        return label["display_name"].loc[label["index"] == label_int].item()
    except:
        return "CAN'T FIND LABEL INDEX"


def _extract_sequence(tf_data):
    """
    Extract the SequenceExample as a string from a given TensorFlow record

    :param data_dir: A TensorFlow record
    :returns: String value of SequenceExample of TensorFlow record
    """
    sequence = tf.train.SequenceExample()
    sequence.ParseFromString(tf_data)
    return sequence


def _extract_audio_embedding(ae_features):
    """
    Extract Audio Embedding as a List

    :param ae_features: A TensorFlow feature list
    :returns: A list of the audio embeddings as float values
    """
    audio_embeddings = []

    sess = tf.InteractiveSession()      # Need to start this for .eval()
    for second in range(0, len(ae_features)):
        raw_embedding = tf.decode_raw(ae_features[second].bytes_list.value[0],tf.uint8)
        float_embedding = tf.cast(raw_embedding, tf.float32).eval().tolist()
        audio_embeddings.append(float_embedding)
    sess.close()

    return audio_embeddings


def _convert_labels(protobuf):
    """
    Convert labels from google.protobuf.pyext._message.RepeatedScalarContainer
    to a list of the labels as the string values representation

    :param protobuf: google.protobuf.pyext._message.RepeatedScalarContainer of
                     labels extracted from TensorFlow SequenceExample
    :returns: a list of integer values for labels
    """
    labels = []
    for i in range(0, len(protobuf)):
        label_as_string = _lookup_label_by_index(protobuf[i])
        labels.append(label_as_string)
    return labels


def _process_tensor_file(tf_file_path):
    """
    Process the tensorflow records tf_file_path to represent in a dictionary to
    enable easy process of data

    :param tf_file_path: The path to the tensorflow records to process
    :returns: A dictionary keyed by video_ids where the values are the labels
              and audio embeddings of the video_id
    """
    data = {}

    raw_data = tf.python_io.tf_record_iterator(path=tf_file_path)

    for record in raw_data:
        sequence = _extract_sequence(record)
        video_id = sequence.context.feature["video_id"].bytes_list.value

        labels_protobuf = sequence.context.feature["labels"].int64_list.value
        labels = _convert_labels(labels_protobuf)

        audio_embedding_features = sequence.feature_lists.feature_list["audio_embedding"].feature
        audio_embedding_list = _extract_audio_embedding(audio_embedding_features)

        data[video_id[0]] = {
                            "labels": labels,
                            "audio_embedding": audio_embedding_list
                        }

    return data

def _process_data(dir_path):
    """
    Process the tensorflow records in data_dir and compute statistics about the features contained in them

    :param data_dir: The path to the tensorflow records to process
    :returns:
    """
    data = {}
    all_tf_files = os.listdir(dir_path)
    pool = mp.Pool(NUM_WORKERS)
    result = list(tqdm.tqdm(pool.imap(_process_tensor_file, [os.path.join(dir_path, x) for x in all_tf_files]), total=len(all_tf_files), unit="files"))

    return result


def get_bal_train():
    """
    Process the balanced training set TensorFlow records to use in a ML model

    :returns: A dictionary representation of the data set
    """
    return _process_data(PATH_TO_BAL_TRAIN_FOLDER)


def get_unbal_train():
    """
    Process the unbalanced training set TensorFlow records to use in a ML model

    :returns: A dictionary representation of the data set
    """
    return _process_data(PATH_TO_UNBAL_TRAIN_FOLDER)


def get_eval():
    """
    Process the evaluation set TensorFlow records to use in a ML model

    :returns: A dictionary representation of the data set
    """
    return _process_data(PATH_TO_EVAL_FOLDER)


# # TESTING STUFF
# if __name__ == "__main__":
#     label = _lookup_label_by_index(100000)
#     print(label)
#     data = os.path.join(PATH_TO_BAL_TRAIN_FOLDER, "_5.tfrecord")
#
#     dict = _process_data(data)
#     print(dict)
#     print(type(dict[b'_5w5TVK5B90']['audio_embeddings']))
#
#     dict = get_bal_train()
#     print(dict)
