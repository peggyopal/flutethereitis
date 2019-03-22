"""
File Name: clean_embeddings.py

Authors: Peggy Anderson & Kyle Seidenthal

Date: 12-02-2019

Last Modified: Thu 14 Feb 2019 08:48:47 AM CST

Description: A script to sort the audio embedding feature data into ones we want and ones we dont

"""
import numpy as np
import pandas as pd
import argparse
import os
import tensorflow as tf
import tqdm
import sys
from shutil import copyfile
import multiprocessing as mp

module_path = os.path.dirname(os.path.abspath("src/helpers.py"))
sys.path.insert(0, module_path + '/../')
import src.helpers as help


# Suppress TQDM warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Data stuff
PATH_TO_DATA_FOLDER = os.path.abspath("data/")
PATH_TO_CLEAN_DATA_FOLDER = os.path.abspath("data/clean_data")

# Original data stuff
PATH_TO_UNBAL_TRAIN_FOLDER = PATH_TO_DATA_FOLDER + "/unbal_train"
PATH_TO_BAL_TRAIN_FOLDER = PATH_TO_DATA_FOLDER + "/bal_train"
PATH_TO_EVAL_FOLDER = PATH_TO_DATA_FOLDER + "/eval"

# Cleaned data stuff
PATH_TO_CLEAN_BAL_CSV = PATH_TO_CLEAN_DATA_FOLDER + "/balanced_train_segments_cleaned.csv"
PATH_TO_CLEAN_UNBAL_CSV = PATH_TO_CLEAN_DATA_FOLDER + "/unbalanced_train_segments_cleaned.csv"
PATH_TO_CLEAN_EVAL_CSV = PATH_TO_CLEAN_DATA_FOLDER + "/eval_segments_cleaned.csv"
PATH_TO_CLEAN_UNBAL_TRAIN_FOLDER = PATH_TO_CLEAN_DATA_FOLDER + "/unbal_train/flute_didgeridoo"
PATH_TO_CLEAN_BAL_TRAIN_FOLDER = PATH_TO_CLEAN_DATA_FOLDER + "/bal_train/flute_didgeridoo"
PATH_TO_CLEAN_EVAL_FOLDER = PATH_TO_CLEAN_DATA_FOLDER + "/eval/flute_didgeridoo"
CLEAN_LABELS_CSV = os.path.join(PATH_TO_DATA_FOLDER, "class_labels_indices_cleaned.csv")
CLEANED_CSV = pd.DataFrame()


NUM_WORKERS = mp.cpu_count()


def _set_cleaned_csv(cleaned_csv_path):
    global CLEANED_CSV
    try:
        CLEANED_CSV = pd.read_csv(cleaned_csv_path)
    except Exception as e:
        raise


def _extract_cleaned_records(cleaned_labels, sequence):
    cleaned_records = []

    sess = tf.InteractiveSession()
    video_id = sequence.context.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8')
    sess.close()

    start_time_seconds = sequence.context.feature['start_time_seconds']
    end_time_seconds = sequence.context.feature['end_time_seconds']

    audio_embedding_features = sequence.feature_lists.feature_list["audio_embedding"].feature
    audio_embedding_list = help.extract_audio_embedding(audio_embedding_features)

    hits = CLEANED_CSV[CLEANED_CSV["# YTID"].str.contains(video_id)]

    if not hits.empty:
        example_label = list(np.asarray(sequence.context.feature['labels'].int64_list.value))

        labels_to_keep = []
        for label in example_label:
            label_index_hits = cleaned_labels[cleaned_labels["index"] == label]

            if not label_index_hits.empty:
                labels_to_keep.append(label)

        updated_record = tf.train.SequenceExample(
            context=
                tf.train.Features(feature={
                    'video_id': tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[bytes(video_id, encoding="utf-8")]
                        )
                    ),
                    'start_time_seconds': start_time_seconds,
                    'end_time_seconds': end_time_seconds,
                    'labels': tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=labels_to_keep
                        )
                    )
                    }
                ),
              feature_lists=sequence.feature_lists

        )
        cleaned_records.append(updated_record)

    return cleaned_records


def _overwrite_tfrecord(cleaned_dir_path, tfrecord, cleaned_examples):
    # Remove the old file
    os.remove(cleaned_dir_path)
    # Save it
    with tf.python_io.TFRecordWriter(cleaned_dir_path) as writer:
        for examp in cleaned_examples:
            writer.write(examp.SerializeToString())
    return


def _process_tfrecords(tf_file_path):
    """
    Process the given tensorflow record file to remove unecessary labels

    :param file: The file to process
    :returns: None, but the file will be updated to only include relevant labels
    """
    cleaned_labels = pd.read_csv(CLEAN_LABELS_CSV)
    raw_data = tf.python_io.tf_record_iterator(path=tf_file_path)

    for record in raw_data:
        sequence = help.extract_sequence(record)
        updated_record = _extract_cleaned_records(cleaned_labels, sequence)


    _overwrite_tfrecord(tf_file_path, record, updated_record)
    # # Remove the old file
    # os.remove(os.path.join(cleaned_dir_path, tfrecord))
    # # Save it
    # with tf.python_io.TFRecordWriter(os.path.join(cleaned_dir_path, tfrecord)) as writer:
    #     for examp in cleaned_examples:
    #         writer.write(examp.SerializeToString())


def _make_clean_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _copy_all_desired_tfrecords(data_dir_path, cleaned_csv, cleaned_dir_path):
    for filename in tqdm.tqdm(os.listdir(data_dir_path), unit="files"):
        if filename[-9:] != ".tfrecord" and filename.startswith("."):
            continue

        YouTubeID_shard = filename[:2]

        hits = cleaned_csv[cleaned_csv["# YTID"].str.startswith(YouTubeID_shard)]

        if not hits.empty:
            cleaned_file_path = os.path.join(cleaned_dir_path, filename)
            copyfile(os.path.join(data_dir_path, filename), cleaned_file_path)
            # print(os.path.join(data_dir_path, filename), cleaned_file_path)


def _segregate_tfrecords(data_dir_path, cleaned_csv_path, cleaned_dir_path):
    _make_clean_dir(cleaned_dir_path)

    print("\nFlowing Tensors...")
    _set_cleaned_csv(cleaned_csv_path)
    _copy_all_desired_tfrecords(data_dir_path, CLEANED_CSV, cleaned_dir_path)

    print("\nTensoring Flows...")

    audio_embeddings_dict = {}
    audio_labels_dict = {}

    clean_labels = pd.read_csv(CLEAN_LABELS_CSV)

    cleaned_tfrd = os.listdir(cleaned_dir_path)
    cleaned_tfrd_path = [os.path.join(cleaned_dir_path, x) for x in cleaned_tfrd if not x.startswith(".")]

    pool = mp.Pool(NUM_WORKERS)
    result = list(tqdm.tqdm(pool.imap(_process_tfrecords, cleaned_tfrd_path), total=len(cleaned_tfrd), unit="files"))

    return



def clean_bal_train_records():
    """
    Process the balanced training set TensorFlow records to use in a ML model

    :returns: A dictionary representation of the data set
    """
    return _segregate_tfrecords(PATH_TO_BAL_TRAIN_FOLDER, PATH_TO_CLEAN_BAL_CSV, PATH_TO_CLEAN_BAL_TRAIN_FOLDER)


def clean_unbal_train_records():
    """
    Process the unbalanced training set TensorFlow records to use in a ML model

    :returns: A dictionary representation of the data set
    """
    return _segregate_tfrecords(PATH_TO_UNBAL_TRAIN_FOLDER, PATH_TO_CLEAN_UNBAL_CSV, PATH_TO_CLEAN_UNBAL_TRAIN_FOLDER)


def clean_eval_records():
    """
    Process the evaluation set TensorFlow records to use in a ML model

    :returns: A dictionary representation of the data set
    """
    return _segregate_tfrecords(PATH_TO_EVAL_FOLDER, PATH_TO_CLEAN_EVAL_CSV, PATH_TO_CLEAN_EVAL_FOLDER)
