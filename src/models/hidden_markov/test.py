"""
File Name:

Authors: Peggy Anderson & Kyle Seidenthal

Date: 11-03-2019

Description:

"""

import sys, os
import numpy as np

from hmmlearn import hmm

module_path = os.path.dirname(os.path.abspath("src/models"))
sys.path.insert(0, module_path + '/../')
from src.models.process_data import get_eval, get_bal_train, get_unbal_train

import pandas as pd
import matplotlib.pyplot as plt


# Suppress TQDM warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_lists_of_audio_embeddings(dataset):
    flute_audio_embeddings = []
    didgerigoo_audio_embeddings = []

    for video_id_features in dataset:
        for video_id in video_id_features:
            if 'Flute' in video_id_features[video_id]['labels']:
                flute_audio_embeddings.append(video_id_features[video_id]['audio_embedding'])

            if 'Didgeridoo' in video_id_features[video_id]['labels']:
                didgerigoo_audio_embeddings.append(video_id_features[video_id]['audio_embedding'])

    return flute_audio_embeddings, didgerigoo_audio_embeddings


def get_datasets(training_set):
    if training_set == "unbal":
        training = get_unbal_train()
    else:
        training = get_bal_train()
    eval = get_eval()
    return training, eval


def hmm_run(training_set="bal"):
    print("Get Datasets... ")
    train_data, eval_data = get_datasets(training_set)
    print("Datasets Got! \n")

    model = hmm.GaussianHMM(n_components=2)

    flute_aes_train, didgerigoo_aes_train = create_lists_of_audio_embeddings(train_data)


    print(len(didgerigoo_aes_train))
    print(len(flute_aes_train))
