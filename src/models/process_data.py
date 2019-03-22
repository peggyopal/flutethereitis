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
import tensorflow as tf

import json
import os
import sys
import tqdm         # progress bar
import unittest

module_path = os.path.dirname(os.path.abspath("src/helpers.py"))
sys.path.insert(0, module_path + '/../../')
import src.helpers as help


# Pooling stuff
NUM_WORKERS = mp.cpu_count()

# TensorFlow Data Stuff
PATH_TO_DATA_FOLDER = os.path.abspath("data/")
PATH_TO_CLEAN_DATA_FOLDER = os.path.abspath("data/clean_data")
PATH_TO_UNBAL_TRAIN_FOLDER = PATH_TO_CLEAN_DATA_FOLDER + "/unbal_train/flute_didgeridoo"
PATH_TO_BAL_TRAIN_FOLDER = PATH_TO_CLEAN_DATA_FOLDER + "/bal_train/flute_didgeridoo"
# PATH_TO_EVAL_FOLDER = PATH_TO_DATA_FOLDER + "/eval"
PATH_TO_EVAL_FOLDER = PATH_TO_CLEAN_DATA_FOLDER + "/eval/flute_didgeridoo"



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
        label_as_string = help.lookup_label_by_index(protobuf[i])
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
        sequence = help.extract_sequence(record)
        video_id = sequence.context.feature["video_id"].bytes_list.value

        labels_protobuf = sequence.context.feature["labels"].int64_list.value
        labels = help.convert_feature_key_to_string(labels_protobuf)

        audio_embedding_features = sequence.feature_lists.feature_list["audio_embedding"].feature
        audio_embedding_list = help.extract_audio_embedding(audio_embedding_features)

        data[video_id[0]] = {
                            "labels": labels,
                            "audio_embeddings": audio_embedding_list
                        }

    return data

def _process_data(dir_path):
    """
    Process the tensorflow records in data_dir and compute statistics about the features contained in them

    :param data_dir: The path to the tensorflow records to process
    :returns:
    """

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    data = {}
    print(dir_path)
    all_tf_files = os.listdir(dir_path)
    all_tf_files = [os.path.join(dir_path, x) for x in all_tf_files if not x.startswith(".")]
    pool = mp.Pool(NUM_WORKERS)
    # print(len([os.path.join(dir_path, x) for x in all_tf_files]))
    result = list(tqdm.tqdm(pool.imap(_process_tensor_file, all_tf_files), total=len(all_tf_files), unit="files"))

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
