"""
File Name:

Authors: Peggy Anderson & Kyle Seidenthal

Date: 11-03-2019

Description:

"""

import sys
import os


from hmmlearn import hmm
from sklearn.model_selection import train_test_split, KFold

import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import warnings


# Suppress TQDM warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _check_ae_array_length(audio_embedding):
    while len(audio_embedding) != 10:
        audio_embedding.append(np.zeros(128))
    return audio_embedding


def _create_arrays_of_audio_embeddings(dataset):
    flute_audio_embeddings = []
    didgerigoo_audio_embeddings = []

    for video_id_features in dataset:
        for video_id in video_id_features:
            audio_embeddings = video_id_features[video_id]['audio_embedding']
            _check_ae_array_length(audio_embeddings)

            if 'Flute' in video_id_features[video_id]['labels']:
                flute_audio_embeddings.append(audio_embeddings)

            if 'Didgeridoo' in video_id_features[video_id]['labels']:
                didgerigoo_audio_embeddings.append(audio_embeddings)#

    flute_audio_embeddings = np.array(flute_audio_embeddings)
    didgerigoo_audio_embeddings = np.array(didgerigoo_audio_embeddings)

    return flute_audio_embeddings, didgerigoo_audio_embeddings


def _fit_classifier(training_set):
    model = hmm.GaussianHMM(n_components=2, algorithm='viterbi')
    # model = hmm.GMMHMM(n_components=2, init_params="st")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        for audio_embedding in training_set:
            model.fit(audio_embedding)

    return model


def _classify(dataset, fHMM, dHMM):
    true_positive = 0
    false_negative = 0
    true_negative = 0
    false_positive = 0
    idk = 0

    count = 0

    for video_id_features in dataset:
        for video_id in video_id_features:
            audio_embedding = video_id_features[video_id]['audio_embedding']
            _check_ae_array_length(audio_embedding)

            label = video_id_features[video_id]['labels']

            # Check if there is more than one label associated with the data
            if len(label) > 1:
                for l in label:
                    if l == 'Flute':
                        label = 'Flute'
                        break

            flute_proba = fHMM.predict(audio_embedding)
            didgeridoo_proba = dHMM.predict(audio_embedding)

            if sum(flute_proba) > sum(didgeridoo_proba):
                if label[0] == 'Flute':
                    true_positive += 1
                else:
                    false_negative += 1
            elif sum(flute_proba) < sum(didgeridoo_proba):
                if label[0] == 'Didgeridoo':
                    true_negative += 1
                else:
                    false_positive += 1
            else:
                idk += 1

    print(count)
    return (true_positive, false_negative, true_negative, false_positive, idk)


def _compute_precision_recall_accuracy(classifications):
    true_positive = classifications[0]
    false_negative = classifications[1]
    true_negative = classifications[2]
    false_positive = classifications[3]

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    accuracy = true_positive / (true_negative + false_negative + false_positive + true_positive)

    return precision, recall, accuracy


def hmm_run(train_data, eval_data):
    flute_train, didgeridoo_train = _create_arrays_of_audio_embeddings(train_data)

    flute_hmm = _fit_classifier(flute_train)
    didgeridoo_hmm = _fit_classifier(didgeridoo_train)

    classifications = _classify(eval_data, flute_hmm, didgeridoo_hmm)
    # test_precision, test_recall, test_accuracy = _compute_precision_recall_accuracy(classifications)

    print("classifications: {}".format(classifications))
    # print("precision: {}".format(test_precision))
    # print("recall: {}".format(test_recall))
    # print("accuracy: {}".format(test_accuracy))
    ## use hmm.predict_proba(X)

import json
TRAIN_JSON = "data/clean_data/train_data_json.txt"
EVAL_JSON = "data/clean_data/eval_data_json.txt"
training = json.load(open(TRAIN_JSON))
eval = json.load(open(EVAL_JSON))
hmm_run(training, eval)
