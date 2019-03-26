"""
File Name: helpers.py

Authors: Peggy Anderson & Kyle Seidenthal

Date: 11-03-2019

Description: A file of shared methods used in models and data characterization

"""

import tensorflow as tf
import pandas as pd

import os
import tqdm


# Labels Stuff
PATH_TO_DATA_FOLDER = os.path.abspath("data/")
PATH_TO_CLEAN_DATA_FOLDER = os.path.abspath("data/clean_data")
LABELS_CSV = os.path.join(PATH_TO_DATA_FOLDER, "class_labels_indices.csv")
CLEAN_LABELS_CSV = os.path.join(PATH_TO_CLEAN_DATA_FOLDER, "class_labels_indices_cleaned.csv")


def extract_label_codes_from_cleaned_labels_csv():
    clean_labels = pd.read_csv(CLEAN_LABELS_CSV)
    label_string_codes = []

    for index, row in tqdm.tqdm(clean_labels.iterrows(), total=clean_labels.shape[0]):
        label_string_codes.append(row["mid"])

    return label_string_codes


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


def extract_example(tf_data):

    example = tf.train.Example.FromString(tf_data)
    return example

def extract_sequence(tf_data):
    """
    Extract the SequenceExample as a string from a given TensorFlow record

    :param data_dir: A TensorFlow record
    :returns: String value of SequenceExample of TensorFlow record
    """
    sequence = tf.train.SequenceExample.FromString(tf_data)
    return sequence


def extract_audio_embedding(ae_features):
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


def create_segment_dataframe(path_to_segment):
    """
    Create a dataframe for the segments in the given file
    :param path_to_segment: The path to the file to load
    :return: A pandas dataframe containing the data
    """
    return pd.read_csv(path_to_segment, sep=',\s+', header=2, engine="python")
