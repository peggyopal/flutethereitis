import pandas as pd


def create_segment_dataframe(path_to_segment):
    """
    Create a dataframe for the segments in the given file
    :param path_to_segment: The path to the file to load
    :return: A pandas dataframe containing the data
    """
    return pd.read_csv(path_to_segment, sep=',\s+')

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

        labels = [x for x in labels.split(',')]

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
    pass

def main():
    balanced_train_segments = create_segment_dataframe("../../data/balanced_train_segments.csv")
    balance_train_hist = count_positive_labels(balanced_train_segments)
    print(balance_train_hist)

if __name__ == "__main__":
    main()

