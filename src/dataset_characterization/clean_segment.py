"""
File Name: clean_segment.py

Authors: Peggy Anderson & Kyle Seidenthal

Date: 12-02-2019

Last Modified: Wed 13 Feb 2019 06:49:40 PM CST

Description: A script to clean the datasets containing the labels

"""
import pandas as pd
import os
import utils
from tqdm import tqdm
import argparse
import sys

PATH_TO_CLEAN_DATA_FOLDER = "../../data/clean_data"

# Parse command line args to get file name
parser = argparse.ArgumentParser(description="Clean a given set of segments.")

parser.add_argument('segment_csv', help="The path to the csv of the data to clean.")

args = parser.parse_args()

if args.segment_csv[-4:] != ".csv":
    print("Must enter a csv")
    sys.exit(1)

# Get the path to the file
segment_path = args.segment_csv

# Read the cleaned version of the labels
clean_labels = pd.read_csv(os.path.join(PATH_TO_CLEAN_DATA_FOLDER, "class_labels_indicies_cleaned.csv"))

# Get a list of the codes for the labels that we want to keep data for
string_codes = []

for index, row in tqdm(clean_labels.iterrows(), total=clean_labels.shape[0]):
    string_codes.append(row["mid"])

# Load the segment data that was specified
segments = utils.create_segment_dataframe(segment_path)

clean_segments = pd.DataFrame()

# Keep matches for the labels that we are keeping
for string in tqdm(string_codes):
    hits = segments[segments["positive_labels"].str.lower().str.contains(string.lower())]
    clean_segments = clean_segments.append(hits)

for index, row in clean_segments.iterrows():
    positive_labels = row["positive_labels"].lstrip("\"").rstrip("\"").split(",")
    positive_labels = [x for x in positive_labels if x in string_codes]
    clean_segments.at[index, "positive_labels"] = ",".join(positive_labels)

# Save it
outname = PATH_TO_CLEAN_DATA_FOLDER + "/" + args.segment_csv.lstrip('../../data/')[:-4] + "_cleaned.csv"

clean_segments.to_csv(outname)
