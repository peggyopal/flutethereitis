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
import os

# TensorFlow Data Stuff
PATH_TO_DATA_FOLDER = "../../data"
PATH_TO_CLEAN_DATA_FOLDER = "../../data/clean_data/flute_didgeridoo"
PATH_TO_UNBAL_TRAIN_FOLDER = PATH_TO_CLEAN_DATA_FOLDER + "/unbal_train/flute_didgeridoo"
PATH_TO_BAL_TRAIN_FOLDER = PATH_TO_CLEAN_DATA_FOLDER + "/bal_train/flute_didgeridoo"
PATH_TO_EVAL_FOLDER = PATH_TO_CLEAN_DATA_FOLDER + "/eval/flute_didgeridoo"

# Labels Stuff
LABELS_CSV = os.path.join(PATH_TO_DATA_FOLDER, "class_labels_indicies.csv")


def _lookup_label(label_int):
    """
    """
    labels = pd.read_csv(LABELS_CSV)
    return labels["display_name"].loc[labels["index"] == label_int].item()


def _prepare_data():
    """
    """
    return


def get_bal_train():
    """
    """
    return


def get_unbal_train():
    """
    """
    return


def get_eval():
    """
    """
    return


if __name__ == "__main__":
    print(_lookup_label(169))
