"""
File Name: clean_embeddings.py

Authors: Peggy Anderson & Kyle Seidenthal 

Date: 12-02-2019

Last Modified: Thu 14 Feb 2019 08:48:47 AM CST

Description: A script to sort the audio embedding feature data into ones we want and ones we dont

"""
import numpy as np
import pandas as pd
import argparse
import os
import tensorflow as tf
from tqdm import tqdm
from shutil import copyfile
import multiprocessing as mp

PATH_TO_DATA_FOLDER = "../../data"
LABELS_TO_KEEP = os.path.join(PATH_TO_DATA_FOLDER,"clean_data" , "class_labels_indicies_cleaned.csv")

# Parse command line args to get file name
parser = argparse.ArgumentParser(description="Clean a given set of segments.")

MAIN_DATA_DIR = "../../data/"
NUM_WORKERS = mp.cpu_count()

parser.add_argument('data_dir', help="The path to the directory containing the data set.")
parser.add_argument('clean_csv', help="The path to the csv to use to get the labels from.")
parser.add_argument('outname', help="The name of the folder to store the cleaned data in")
args = parser.parse_args()

segment = pd.read_csv(args.clean_csv)

original_folder = os.path.split(args.data_dir)[-1]

clean_outdir = os.path.join(MAIN_DATA_DIR, "clean_data", original_folder, args.outname)

if not os.path.exists(clean_outdir):
    os.makedirs(clean_outdir)

print("\nFlowing Tensors...")

# Copy relevant files that match the beginnings of video codes for the labels we have selected to a clean location
for filename in tqdm(os.listdir(args.data_dir)):
    if filename[-9:] != ".tfrecord":
        continue

    YTID_shard = filename[:2]
    
    hits = segment[segment["# YTID"].str.startswith(YTID_shard)]
    

    if not hits.empty:
        clean_outfile = os.path.join(clean_outdir, filename)
        copyfile(os.path.join(args.data_dir, filename), clean_outfile)

print("\nTensoring Flows...")

audio_embeddings_dict = {}
audio_labels_dict = {}
sess = tf.Session() 

clean_labels = pd.read_csv(LABELS_TO_KEEP)


def process_tfrecords(tfrecord):
    """
    Process the given tensorflow record file to remove unecessary labels
    
    :param file: The file to process
    :returns: None, but the file will be updated to only include relevant labels
    """
    cleaned_examples = []
    
    # Get each example in the record and check its labels
    for example in tf.python_io.tf_record_iterator(os.path.join(clean_outdir, tfrecord)):
        tf_example = tf.train.Example.FromString(example)
        vid_id = tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8')
        start_time_seconds = tf_example.features.feature['start_time_seconds']
        end_time_seconds = tf_example.features.feature['end_time_seconds']
        
        hits = segment[segment["# YTID"].str.contains(vid_id)]
        
        
        if not hits.empty:
            example_label = list(np.asarray(tf_example.features.feature['labels'].int64_list.value))
            tf_seq_example = tf.train.SequenceExample.FromString(example)
    
            
            labels_to_keep = []
            for label in example_label:
                label_index_hits = clean_labels[clean_labels["index"] == label]

                if not label_index_hits.empty:
                    labels_to_keep.append(label)
            
           
            example = tf.train.SequenceExample(
                context=
                    tf.train.Features(feature={
                        'video_id': tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[bytes(vid_id, encoding="utf-8")]
                            )
                        ),
                        'start_time_seconds': start_time_seconds,
                        'end_time_seconds': end_time_seconds,
                        'labels': tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=labels_to_keep
                            ) 
                        )
                        }
                    ),
                  feature_lists=tf_seq_example.feature_lists                
                  
            )  
            cleaned_examples.append(example)
    
    # Remove the old file
    os.remove(os.path.join(clean_outdir, tfrecord))    
    # Save it
    with tf.python_io.TFRecordWriter(os.path.join(clean_outdir, tfrecord)) as writer:
        for examp in cleaned_examples:
            writer.write(examp.SerializeToString())


# Run through the tensor records and remove any instances of labels that we do not want
pool = mp.Pool(NUM_WORKERS)
result = list(tqdm(pool.imap(process_tfrecords, os.listdir(clean_outdir)), total=len(os.listdir(clean_outdir))))
    

