"""
File Name: run.py

Authors: Peggy Anderson & Kyle Seidenthal

Date: 29-03-2019

Description: A file to run the classifier with HMM or RNN

"""

from src.models.hidden_markov import test
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
    print("So you want to determine if there is a flute?")

    model = str(input("Enter model (either HMM or RNN): ")).upper()
    while model not in ["RNN", "HMM"]:
        model = str(input("Enter either HMM or RNN: ")).upper()

    if model == 'RNN':
        print("There is another way to run RNN, please refer to the documentation")
        exit()

    training_set = str(input("Enter training set (either bal or unbal): ")).lower()
    while training_set not in ["bal", "unbal"]:
        training_set = str(input("Enter training set (either bal or unbal): ")).lower()

    prev_data_exists = check_for_old_data()
    new_data = str(input("Do we want new data? Or use the previous data: [y/n] ")).lower()
    while new_data not in ["y", "n"]:
        training_set = str(input("It's a yes [y] or no [n] question: ")).lower()

    if new_data == "y" or not prev_data_exists:
        get_new_datasets(training_set)

    training, eval = load_datasets()


    if model == 'HMM':
        print("Running HMM Classifier.... \n")
        test.hmm_run(training, eval)
