"""
File Name: clean_segment.py

Authors: Peggy Anderson & Kyle Seidenthal

Date: 12-02-2019

Last Modified: Wed 13 Feb 2019 06:49:40 PM CST

Description: A script to clean the datasets containing the labels

"""
import pandas as pd
import tqdm

import argparse
import os
import sys
import utils

module_path = os.path.dirname(os.path.abspath("src/helpers.py"))
sys.path.insert(0, module_path + '/../')
import src.helpers as help


# Data stuff
PATH_TO_DATA_FOLDER = os.path.abspath("data/")
PATH_TO_CLEAN_DATA_FOLDER = os.path.abspath("data/clean_data")
CLEAN_LABELS_CSV = os.path.join(PATH_TO_DATA_FOLDER, "class_labels_indices.csv")


def _csv_filename_to_path(csv_filename):
    return os.path.join(PATH_TO_DATA_FOLDER, csv_filename)


def _get_csv_filename():
    # Parse command line args to get file name
    parser = argparse.ArgumentParser(description="Clean a given set of segments.")

    parser.add_argument('dataset_csv', help="The path to the csv of the data to clean.")

    args = parser.parse_args()

    if args.dataset_csv[-4:] != ".csv":
        print("Must enter a csv")
        sys.exit(1)

    # Get the path to the file
    dataset_csv = args.dataset_csv
    return dataset_csv


def read_from_clean_labels_csv():
    # Read the cleaned version of the labels
    clean_labels = pd.read_csv(CLEAN_LABELS_CSV)

    # Get a list of the codes for the labels that we want to keep data for
    string_codes = []

    for index, row in tqdm.tqdm(clean_labels.iterrows(), total=clean_labels.shape[0]):
        string_codes.append(row["mid"])


def _isolate_videos_with_desired_labels(csv_path, clean_label_codes):
    all_segments = utils.create_segment_dataframe(csv_path)

    clean_segments = pd.DataFrame()

    # Keep matches for the labels that we are keeping
    for string in tqdm.tqdm(clean_label_codes):
        hits = all_segments[all_segments["positive_labels"].str.lower().str.contains(string.lower())]
        clean_segments = clean_segments.append(hits)

    return clean_segments


def _remove_undesired_labels_from_dataframe(dataframe, clean_label_codes):
    for index, row in dataframe.iterrows():
        positive_labels = row["positive_labels"].lstrip("\"").rstrip("\"").split(",")
        positive_labels = [x for x in positive_labels if x in clean_label_codes]
        dataframe.at[index, "positive_labels"] = ",".join(positive_labels)

    return dataframe


def _save_csv(dataframe, dataset_csv):
    cleaned_segments_filename = PATH_TO_CLEAN_DATA_FOLDER + "/" + dataset_csv.lstrip('../../data/')[:-4] + "_cleaned.csv"
    dataframe.to_csv(cleaned_segments_filename)


def _create_cleaned_segments_csv(dataset_csv):
    # print(csv_name)
    clean_label_codes = help.extract_label_codes_from_cleaned_labels_csv()
    # dataset_csv = _get_csv_filename()
    dataset_path = _csv_filename_to_path(dataset_csv)
    desired_records = _isolate_videos_with_desired_labels(dataset_path, clean_label_codes)
    desired_records_without_undesired_labels = _remove_undesired_labels_from_dataframe(desired_records, clean_label_codes)
    _save_csv(desired_records_without_undesired_labels, dataset_csv)


def clean_bal_train_segments():
    """
    Process the balanced training set TensorFlow records to use in a ML model

    :returns: A dictionary representation of the data set
    """
    return _create_cleaned_segments_csv("balanced_train_segments.csv")


def clean_unbal_train_segments():
    """
    Process the unbalanced training set TensorFlow records to use in a ML model

    :returns: A dictionary representation of the data set
    """
    return _create_cleaned_segments_csv("balanced_train_segments.csv")


def clean_eval_segments():
    """
    Process the evaluation set TensorFlow records to use in a ML model

    :returns: A dictionary representation of the data set
    """
    return _create_cleaned_segments_csv("eval_segments.csv")
