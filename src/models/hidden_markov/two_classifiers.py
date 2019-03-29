"""
File Name:

Authors: Peggy Anderson & Kyle Seidenthal

Date: 11-03-2019

Description:

"""

from hmmlearn import hmm


import warnings


def _fit_classifier(training_set):
    model = hmm.GaussianHMM(n_components=2, algorithm='map')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        model.fit(training_set)

    return model


def get_two_classifiers(flute_train, didgeridoo_train):
    flute_model = _fit_classifier(flute_train)
    return flute_model
