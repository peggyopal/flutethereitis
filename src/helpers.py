"""
File Name: helpers.py

Authors: Peggy Anderson & Kyle Seidenthal

Date: 11-03-2019

Description: A file of shared methods used in models and data characterization

"""

import tensorflow as tf
import pandas as pd

import os


# Labels Stuff
PATH_TO_DATA_FOLDER = os.path.abspath("data/")
LABELS_CSV = os.path.join(PATH_TO_DATA_FOLDER, "class_labels_indices.csv")


def lookup_label_by_index(label_int):
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


def convert_feature_key_to_string(protobuf):
    features = []
    for i in range(0, len(protobuf)):
        feature_as_string = lookup_label_by_index(protobuf[i])
        features.append(feature_as_string)
    return features


def extract_sequence(tf_data):
    """
    Extract the SequenceExample as a string from a given TensorFlow record

    :param data_dir: A TensorFlow record
    :returns: String value of SequenceExample of TensorFlow record
    """
    sequence = tf.train.SequenceExample()
    sequence.ParseFromString(tf_data)
    return sequence
