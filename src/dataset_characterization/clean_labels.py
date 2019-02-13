"""
File Name: clean_labels.py

Authors: Peggy Anderson & Kyle Seidenthal 

Date: 12-02-2019

Last Modified: Tue 12 Feb 2019 05:58:07 PM CST

Description: A script to remove labels that are not related to musical instruments

"""
import pandas as pd
import os
from tqdm import tqdm 

PATH_TO_DATA_FOLDER = "../../data"

LABELS_CSV =  os.path.join(PATH_TO_DATA_FOLDER, "class_labels_indices.csv")


labels_to_keep = ["Musical", "Instrument", "Plucked", "String",  "Drum", "Singing","Bass", "Acoustic", "Hi-Hat",
"Cymbal", "Bagpipes", "Digeridoo", "Cello", "Flute", "Glockenspiel", "Clarinet", "Organ", "Percussion", "Trombone",
"Banjo", "Mandolin", "Guitar", "Strum", "Harp", "Clapping", "Piano", "Trumpet", "Cowbell", "Harmonica", "Saxophone",
"French Horn", "Theremin", "Timpani", "Rattle", "Jingle Bell", "Zither", "Rimshot","Harpsichord", "Maraca", "Yodeling",
"Tubular Bells", "Gong", "Violin", "Fiddle"]

labels_df = pd.read_csv(LABELS_CSV)
   
clean_labels = pd.DataFrame()
     
for label in tqdm(labels_to_keep):
    hits = labels_df[labels_df["display_name"].str.lower().str.contains(label.lower())]

    clean_labels = clean_labels.append(hits)

clean_labels = clean_labels.drop_duplicates(subset="display_name")
  
clean_labels.to_csv(os.path.join(PATH_TO_DATA_FOLDER, "clean_labels.csv"))
    

