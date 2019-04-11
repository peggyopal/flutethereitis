"""
File Name: run.py

Authors: Peggy Anderson & Kyle Seidenthal

Date: 29-03-2019

Description: A file to run the classifier with HMM or RNN

"""

from src.models.hidden_markov import two_classifiers
from src.models.process_data import get_eval, get_bal_train, get_unbal_train


import json
import os


TRAIN_JSON = "data/clean_data/train_data_json.txt"
EVAL_JSON = "data/clean_data/eval_data_json.txt"


def check_for_old_data():
    train = os.path.isfile(TRAIN_JSON)
    eval = os.path.isfile(EVAL_JSON)

    if train and eval:
        return True

    return False


def get_new_datasets(training_set):
    if training_set == "unbal":
        training = get_unbal_train()
    else:
        training = get_bal_train()

    eval = get_eval()

    json.dump(training, open(TRAIN_JSON, "w"))
    json.dump(eval, open(EVAL_JSON, "w"))


def load_datasets():
    training = json.load(open(TRAIN_JSON))
    eval = json.load(open(EVAL_JSON))
    return training, eval


if __name__ == "__main__":
    print("-------------------------------------------------------------------")
    print("wait. SETUP REQUIRED!")

    model = str(input("Enter model (either HMM or RNN): ")).upper()
    while model not in ["RNN", "HMM"]:
        model = str(input("Enter either HMM or RNN: ")).upper()

    if model == 'RNN':
        print("There is another way to run RNN, please refer to the documentation")
        exit()

    prev_data_exists = check_for_old_data()
    if prev_data_exists:
        new_data = str(input("Do you want to load new data? (This would over write any previous data) [y/n] ")).lower()
        while new_data not in ["y", "n"]:
            new_data = str(input("It's a yes [y] or no [n] question: ")).lower()

    if not prev_data_exists or new_data == "y":
        training_set = str(input("Enter training set (either bal or unbal): ")).lower()
        while training_set not in ["bal", "unbal"]:
            training_set = str(input("Enter training set (either bal or unbal): ")).lower()

    if new_data == "y" or not prev_data_exists:
        get_new_datasets(training_set)

    training, eval = load_datasets()
    print("-------------------------------------------------------------------")

    if model == 'HMM':
        print("Running HMM Classifier.... \n")
        two_classifiers.hmm_run(training, eval)
