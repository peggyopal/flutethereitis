"""
File Name: feature_charactarization.py

Authors: Peggy Anderson & Kyle Seidenthal 

Date: 15-02-2019

Last Modified: Sat 16 Feb 2019 02:51:31 PM CST

Description: A script to calculate various properties about the 128-dimensional feature vectors

"""

import numpy as np
import pandas as pd
import argparse
import os
import tensorflow as tf
from tqdm import tqdm
from shutil import copyfile
import multiprocessing as mp
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates, andrews_curves

PATH_TO_DATA_FOLDER = "../../data"
NUM_WORKERS = mp.cpu_count()
LABELS_CSV = os.path.join(PATH_TO_DATA_FOLDER,"clean_data" , "class_labels_indicies_cleaned.csv")

def process_tensor_file(tfrecord):
    """
    Process one tensorflow record file and get all of its feature vectors
    
    :param tfrecord: The path to the record file to process
    :returns: A dictionary, keyed by the class label.  Each entry contains a 2D numpy array where the rows are the 128
    8-bit audio features for the class in the dictionary.
    """
    feature_vectors = {}
    
     # Get each example in the record and process its features
    for example in tf.python_io.tf_record_iterator(tfrecord):
        tf_example = tf.train.Example.FromString(example)
        
        example_label = list(np.asarray(tf_example.features.feature['labels'].int64_list.value))
        tf_seq_example = tf.train.SequenceExample.FromString(example)
        n_frames = len(tf_seq_example.feature_lists.feature_list['audio_embedding'].feature)
        
        audio_frame = []
        
        sess = tf.InteractiveSession() 

        # iterate through frames
        for i in range(n_frames):
            audio_frame.append(tf.cast(tf.decode_raw(tf_seq_example.feature_lists.feature_list['audio_embedding'].feature[i].bytes_list.value[0],tf.uint8)
                       ,tf.float32).eval()) 
            
        for label in example_label:
            if label not in feature_vectors.keys():
                feature_vectors[label] = []

            for frame in audio_frame:
                feature_vectors[label].append(frame)  
        sess.close() 
                               
    return feature_vectors

def process(data_dir, outname, title):
    """
    Process the tensorflow records in data_dir and compute statistics about the features contained in them
    
    :param data_dir: The path to the tensorflow records to process
    :param outname: The path to the directory to save the computations in
    :returns: None
    """
     
    if not os.path.exists(outname):
        os.makedirs(outname)


    # Get the features from the data in parallel
    pool = mp.Pool(NUM_WORKERS)
    result = list(tqdm(pool.imap(process_tensor_file, [os.path.join(data_dir, x) for x in os.listdir(data_dir)]), total=len(os.listdir(data_dir))))

    # Put all the data in one place, indexed by class name
    master_dict = {}

    for res in result:
        for key in res.keys():
            
            if key not in master_dict.keys():
                master_dict[key] = np.array(res[key])

            else:
                master_dict[key] = np.concatenate((master_dict[key], res[key]), axis=0)
    
    # Compute average feature for each class
    avgs = {}
    std_devs = {}

    feature_dataframes = {}

    for key in master_dict.keys():
        avgs[key] = np.mean(master_dict[key], axis=0)
        std_devs[key] = np.std(master_dict[key], axis=0)
        
        feature_dataframes[key] = pd.DataFrame(master_dict[key], columns=[x for x in range(128)])
        feature_dataframes[key]['class'] = index_to_english(key)
        
    # Create a dataframe from the features that we can use to make nice plots
    master_dataframe = pd.concat([feature_dataframes[key] for key in feature_dataframes.keys()])


    # Histogram of the features
    bins = np.arange(128)
   
    plt.figure()
    

    for key in avgs.keys():
        
        if index_to_english(key) == "Flute":
            color_name = 'r'
        else:
            color_name = 'b'

        avg = avgs[key]
        plt.bar(bins, avg, alpha=0.5, label=index_to_english(key), color=color_name)
        

    plt.legend(loc="upper right")
    if title is None:
        plt.title("Avg. Flute vs Didgeridoo Feature Vector")
    else:
        plt.title("Avg. " + title + " Flute vs Didgeridoo Feature Vector")
    
    plt.xlabel("Feature Number")
    plt.ylabel("Value")
    plt.savefig(os.path.join(outname, 'avg_feature_vector.png'))
    #plt.show()
    
    # Andrews curves help visualize high dimensional features
    plt.figure()
    if title is None:
        plt.title("Andews Curves")
    else:
        plt.title("Andrews Curves for " + title) 

    ax = andrews_curves(master_dataframe, 'class', colormap='viridis')

    # Correct the colours so they are always the same
    colors = {l.get_label():l.get_color() for l in ax.lines}
    
    for line, klass in zip(ax.lines, master_dataframe["class"]):
        if klass == "Flute":
            line.set_color('r')
        else:
            line.set_color('b')
    leg = ax.get_legend()
    hl_dict = {handle.get_label(): handle for handle in leg.legendHandles}
    hl_dict['Flute'].set_color('red')
    hl_dict['Didgeridoo'].set_color('blue')
    plt.savefig(os.path.join(outname, 'andrews_curves.png'))
    #plt.show()
    
    # Get a latex table of the description for the data
    description_latex = master_dataframe.describe().to_latex()
    
    master_dataframe.to_csv(os.path.join(outname, "description.csv"))
    
    with open(os.path.join(outname, "latex_description.tex"), 'w') as f:
        f.write(description_latex)   

     
def index_to_english(index):
    """
    Convert a given index to its english class label
    
    :param index: The index to translate
    :returns: A string representing the label name in english
    """
    labels = pd.read_csv(LABELS_CSV)

    row = labels.loc[labels["index"] == index]
    return row["display_name"].item()


if __name__ == "__main__":
    # Parse command line args to get file name
    parser = argparse.ArgumentParser(description="Compute properties of the audio features")

    
    parser.add_argument('data_dir', help="The path to the directory containing the data set.")
    parser.add_argument('outname', help="The path of the folder to store the cleaned data in")
    parser.add_argument('-data_title', help="The name to use in figure titles")
    args = parser.parse_args()

    title = None
    if args.data_title is not None:
        title = args.data_title

    process(args.data_dir, args.outname, title) 

