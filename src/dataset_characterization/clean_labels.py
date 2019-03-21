"""
File Name: clean_labels.py

Authors: Peggy Anderson & Kyle Seidenthal

Date: 12-02-2019

Last Modified: Thu 14 Feb 2019 08:53:30 AM CST

Description: A script to remove labels that are not related to musical instruments

"""

import os

import pandas as pd
from tqdm import tqdm


PATH_TO_DATA_FOLDER = os.path.abspath("data/")
PATH_TO_CLEAN_DATA_FOLDER = os.path.abspath("data/clean_data")

LABELS_CSV =  os.path.join(PATH_TO_DATA_FOLDER, "class_labels_indices.csv")


def clean_all_labels():

# labels_to_keep = ["Musical", "Instrument", "Plucked", "String",  "Drum", "Singing","Bass", "Acoustic", "Hi-Hat",
# "Cymbal", "Bagpipes", "Digeridoo", "Cello", "Flute", "Glockenspiel", "Clarinet", "Organ", "Percussion", "Trombone",
# "Banjo", "Mandolin", "Guitar", "Strum", "Harp", "Clapping", "Piano", "Trumpet", "Cowbell", "Harmonica", "Saxophone",
# "French Horn", "Theremin", "Timpani", "Rattle", "Jingle Bell", "Zither", "Rimshot","Harpsichord", "Maraca", "Yodeling",
# "Tubular Bells", "Gong", "Violin", "Fiddle"]

aan_labels.to_csv(os.path.join(PATH_TO_CLEAN_DATA_FOLDER, "class_labels_indicies_cleaned.csv"))
