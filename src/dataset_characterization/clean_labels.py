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

module_path = os.path.dirname(os.path.abspath("src/helpers.py"))
sys.path.insert(0, module_path + '/../')
import src.helpers as help


PATH_TO_DATA_FOLDER = os.path.abspath("data/")
PATH_TO_CLEAN_DATA_FOLDER = os.path.abspath("data/clean_data")

LABELS_CSV =  os.path.join(PATH_TO_DATA_FOLDER, "class_labels_indices.csv")


# labels_to_keep = ["Musical", "Instrument", "Plucked", "String",  "Drum", "Singing","Bass", "Acoustic", "Hi-Hat",
# "Cymbal", "Bagpipes", "Digeridoo", "Cello", "Flute", "Glockenspiel", "Clarinet", "Organ", "Percussion", "Trombone",
# "Banjo", "Mandolin", "Guitar", "Strum", "Harp", "Clapping", "Piano", "Trumpet", "Cowbell", "Harmonica", "Saxophone",
# "French Horn", "Theremin", "Timpani", "Rattle", "Jingle Bell", "Zither", "Rimshot","Harpsichord", "Maraca", "Yodeling",
# "Tubular Bells", "Gong", "Violin", "Fiddle"]

labels_to_keep = ["flute", "didgeridoo"]

labels_df = pd.read_csv(LABELS_CSV)

clean_labels = pd.DataFrame()

for label in tqdm(labels_to_keep):
    hits = labels_df[labels_df["display_name"].str.lower().str.contains(label.lower())]

    clean_labels = clean_labels.append(hits)

clean_labels = clean_labels.drop_duplicates(subset="display_name")

clean_labels.to_csv(os.path.join(PATH_TO_CLEAN_DATA_FOLDER, "class_labels_indicies_cleaned.csv"))
