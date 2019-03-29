"""
File Name: run.py

Authors: Peggy Anderson & Kyle Seidenthal

Date: 29-03-2019

Description: A file to run the classifier with HMM or RNN

"""

from src.models.hidden_markov import test


if __name__ == "__main__":
    print("So you want to determine if there is a flute?")
    allowed_models = ["RNN", "HMM"]
    model = str(input("Enter model (either HMM or RNN): ")).upper()
    while model not in allowed_models:
        model = str(input("Enter either HMM or RNN: ")).upper()

    if model == 'RNN':
        print("There is another way to run RNN, please refer to the documentation")
        exit()

    allowed_trains = ["bal", "unbal"]
    training_set = str(input("Enter training set (either bal or unbal): ")).lower()
    while training_set not in allowed_trains:
        training_set = str(input("Enter training set (either bal or unbal): ")).lower()

    if model == 'HMM':
        print("Running HMM Classifier.... \n")
        test.hmm_run()
