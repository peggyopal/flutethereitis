"""
File Name: run.py

Authors: Peggy Anderson & Kyle Seidenthal

Date: 29-03-2019

Description: A file to run the classifier with HMM or RNN

"""

from src.models.hidden_markov import test


def get_model_from_command_line():
    allowed_models = ["RNN", "HMM"]
    print("So you want to determine if there is a flute?")
    model = str(input("Enter either HMM or RNN: ")).upper()
    while model not in allowed_models:
        model = str(input("Enter either HMM or RNN: ")).upper()
    return model


if __name__ == "__main__":
    model = get_model_from_command_line()

    if model == 'HMM':
        print("Running HMM Classifier.... \n")
        test.hmm_run()
