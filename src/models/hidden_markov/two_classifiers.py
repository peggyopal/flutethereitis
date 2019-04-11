"""
File Name:

Authors: Peggy Anderson & Kyle Seidenthal

Date: 11-03-2019

Description:

"""

import sys
import os


from hmmlearn import hmm
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings


# Suppress TQDM warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _create_arrays_of_audio_embeddings(dataset):
    flute_audio_embeddings = []
    didgerigoo_audio_embeddings = []

    for video_id_features in dataset:
        for video_id in video_id_features:
            audio_embeddings = sum(video_id_features[video_id]['audio_embedding'], [])
            if len(audio_embeddings) != 1280:
                audio_embeddings = audio_embeddings + ([0] * (1280 - len(audio_embeddings)))

            if 'Flute' in video_id_features[video_id]['labels']:
                flute_audio_embeddings.append(audio_embeddings)

            if 'Didgeridoo' in video_id_features[video_id]['labels']:
                didgerigoo_audio_embeddings.append(audio_embeddings)#

    flute_audio_embeddings = np.array(flute_audio_embeddings)
    didgerigoo_audio_embeddings = np.array(didgerigoo_audio_embeddings)

    return flute_audio_embeddings, didgerigoo_audio_embeddings


def _fit_classifier(training_set):
    model = hmm.GaussianHMM(n_components=2, algorithm='map')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        model.fit(training_set)

    return model


def hmm_run(train_data, eval_data):
    # print("Get Datasets... ")
    # train_data, eval_data = get_datasets(training_set)
    # print("Datasets Got! \n")

    flute_aes, didgeridoo_aes = _create_arrays_of_audio_embeddings(train_data)
    flute_eval, didgeridoo_eval = _create_arrays_of_audio_embeddings(eval_data)

    flute_train, flute_test = train_test_split(flute_aes)
    didgeridoo_train, didgeridoo_test = train_test_split(didgeridoo_aes)

    print("train: ", len(didgeridoo_train))
    print("test: ", len(didgeridoo_test))
    print(len(didgeridoo_aes))

    flute_hmm = _fit_classifier(flute_train)
    didgeridoo_hmm = _fit_classifier(didgeridoo_train)

    predictions = flute_hmm.predict_proba(flute_test)
    print("predictions: ", predictions)

    predictions = didgeridoo_hmm.predict_proba(didgeridoo_test)
    print("predictions: ", predictions)

    score = flute_hmm.score(flute_eval)
    print("flute_hmm flute_eval score: ", score)
    score = flute_hmm.score(didgeridoo_eval)
    print("flute_hmm didgeridoo_eval score: ", score)

    score = didgeridoo_hmm.score(flute_eval)
    print("didgeridoo_hmm flute_eval score: ", score)
    score = didgeridoo_hmm.score(didgeridoo_eval)
    print("didgeridoo_hmm didgeridoo_eval score: ", score)
    # # use hmm.predict_proba(X)

# import json
# TRAIN_JSON = "data/clean_data/train_data_json.txt"
# EVAL_JSON = "data/clean_data/eval_data_json.txt"
# training = json.load(open(TRAIN_JSON))
# eval = json.load(open(EVAL_JSON))
# hmm_run(training, eval)
