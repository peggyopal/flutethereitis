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
            if 'Flute' in video_id_features[video_id]['labels']:
                flute_audio_embeddings.append(video_id_features[video_id]['audio_embedding'])

            if 'Didgeridoo' in video_id_features[video_id]['labels']:
                didgerigoo_audio_embeddings.append(video_id_features[video_id]['audio_embedding'])

    flute_audio_embeddings = np.array(flute_audio_embeddings)
    nsamples, nx, ny = flute_audio_embeddings.shape
    flute_audio_embeddings = flute_audio_embeddings.reshape((nsamples,nx*ny))

    didgerigoo_audio_embeddings = np.array(didgerigoo_audio_embeddings)
    nsamples, nx, ny = didgerigoo_audio_embeddings.shape
    didgerigoo_audio_embeddings = didgerigoo_audio_embeddings.reshape((nsamples,nx*ny))

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

    flute_aes, didgerigoo_aes = _create_arrays_of_audio_embeddings(train_data)

    flute_train, flute_test = train_test_split(flute_aes)
    didgeridoo_train, didgeridoo_test = train_test_split(didgerigoo_aes)

    print("train: ", len(didgeridoo_train))
    print("test: ", len(didgeridoo_test))
    print(len(didgerigoo_aes))

    flute_hmm = _fit_classifier(flute_train)
    didgeridoo_hmm = _fit_classifier(didgeridoo_train)

    predictions = flute_hmm.predict_proba(flute_test)
    print("predictions: ", predictions)

    score = flute_hmm.score(flute_test)
    print("flute hmm flute test score: ", score)

    score = flute_hmm.score(didgeridoo_test)
    print("flute hmm didgeridoo test score: ", score)
    # use hmm.predict_proba(X)

# hmm_run()
