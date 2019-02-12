import pandas as pd
import matplotlib.pyplot as plt
import os



PATH_TO_DATA_FOLDER = "../../data"
BALANCED_TRAIN_SEGMENTS = os.path.join(PATH_TO_DATA_FOLDER, "balanced_train_segments.csv")
EVAL_SEGMENTS = os.path.join(PATH_TO_DATA_FOLDER, "eval_segments.csv")
UNBALANCED_TRAIN_SEGMENTS = os.path.join(PATH_TO_DATA_FOLDER, "unbalanced_train_segments.csv")



def create_segment_dataframe(path_to_segment):
    """
    Create a dataframe for the segments in the given file
    :param path_to_segment: The path to the file to load
    :return: A pandas dataframe containing the data
    """
    return pd.read_csv(path_to_segment, sep=',\s+', header=2, engine="python")

def count_positive_labels(segment):
    """
    Create a histogram of the positive labels for this segment
    Note this is tricky because there can be more than one label in the column, separated by commas
     
    :param segments: A dataframe containing the data
    :returns: A histogram of the occurances of each of the positive labels 
    """
    histogram = {}
        
    for index, row in segment.iterrows():
        labels = row["positive_labels"]

        labels = [str(x.strip("\"")) for x in labels.split(',')]
        
        for label in labels:
            if label in histogram.keys():
                histogram[label] += 1
            else:
                histogram[label] = 1

    return histogram 

def map_positive_labels_to_display_names(histogram):
    """
    Replace the jenky label strings with english display names for the given histogram
    
    :param histogram: A dictionary with positive label strings as keys, and a count of their occurances
    :returns: A new histogram, with the english labels as the key
    """

    class_labels_df = pd.read_csv(os.path.join(PATH_TO_DATA_FOLDER, "class_labels_indices.csv")) 
    new_hist = {}
    for key, value in histogram.items():
        row = class_labels_df.loc[class_labels_df['mid'] == key]
        
        new_key = str(row["display_name"].values[0])
        new_hist[new_key] = value

    return new_hist

def process_label_data(path_to_data):
    """
    Calculate various properties for the label data
    :param path_to_data: The path to the data file
    :return: None
    """

    segment = create_segment_dataframe(path_to_data)
    hist = count_positive_labels(segment)
    mapped_hist = map_positive_labels_to_display_names(hist)

    # TODO: DO stuff

def main():

    process_label_data(UNBALANCED_TRAIN_SEGMENTS)
    process_label_data(BALANCED_TRAIN_SEGMENTS)
    process_label_data(EVAL_SEGMENTS)

if __name__ == "__main__":
    main()

