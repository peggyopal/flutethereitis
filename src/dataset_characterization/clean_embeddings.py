"""
File Name: clean_embeddings.py

Authors: Peggy Anderson & Kyle Seidenthal 

Date: 12-02-2019

Last Modified: Tue 12 Feb 2019 07:13:35 PM CST

Description: A script to sort the audio embedding feature data into ones we want and ones we dont

"""
import pandas as pd
import argparse
import os
from tqdm import tqdm

PATH_TO_DATA_FOLDER = "../../data"

# Parse command line args to get file name
parser = argparse.ArgumentParser(description="Clean a given set of segments.")

parser.add_argument('data_dir', help="The path to the directory containing the data set.")
parser.add_argument('clean_csv', help="The path to the csv to use to get the labels from.")
args = parser.parse_args()

segment = pd.read_csv(args.clean_csv)

clean_outdir = os.path.join(args.data_dir, "clean")
dirty_outdir = os.path.join(args.data_dir, "dirty")

if not os.path.exists(clean_outdir):
    os.makedirs(clean_outdir)

if not os.path.exists(dirty_outdir):
    os.makedirs(dirty_outdir)


for filename in tqdm(os.listdir(args.data_dir)):
    if filename[-9:] != ".tfrecord":
        continue

    YTID_shard = filename[:2]
    
    hits = segment[segment["# YTID"].str.startswith(YTID_shard)]
    

    if hits.empty:
        dirty_outfile = os.path.join(dirty_outdir, filename)
        os.rename(os.path.join(args.data_dir, filename), dirty_outfile)

    else:
        clean_outfile = os.path.join(clean_outdir, filename)
        os.rename(os.path.join(args.data_dir, filename), clean_outfile)


# TODO: So, the files actually contain many sound samples, so we will have to load up the tensorflow data and remove
# entries that we don't have labels for.  We might want to skip the above step in that case, but this may speed that
# process up
