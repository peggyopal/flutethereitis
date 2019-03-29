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


def hmm_run():
    print("Get Datasets... ")
    # unbal_train_data = get_unbal_train()
    bal_train_data = get_bal_train()
    eval_data = get_eval()
    print("Datasets Got! \n")
    #
    # print(len(bal_train_data))
    # print(len(eval_data))

    # np.random.seed(42)
    #
    model = hmm.GaussianHMM(n_components=2)

    flute_aes_bal, didgerigoo_aes_bal = create_lists_of_audio_embeddings(bal_train_data)


    print(len(didgerigoo_aes_bal))
    print(len(flute_aes_bal))

# if __name__ == "__main__":
#     print("Get Datasets... ")
#     # unbal_train_data = get_unbal_train()
#     bal_train_data = get_bal_train()
#     eval_data = get_eval()
#     print("Datasets Got! \n")
#     #
#     # print(len(bal_train_data))
#     # print(len(eval_data))
#
#     # np.random.seed(42)
#     #
#     model = hmm.GaussianHMM(n_components=2)
#
#     flute_aes_bal, didgerigoo_aes_bal = create_lists_of_audio_embeddings(bal_train_data)
#
#
#     print(len(didgerigoo_aes_bal))
#     print(len(flute_aes_bal))
#
#     # X, Z = model.sample(100)
#     #
#     # print(Z)
