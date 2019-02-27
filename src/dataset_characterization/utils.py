"""
File Name: 

Authors: Peggy Anderson & Kyle Seidenthal 

Date: 12-02-2019

Last Modified: Tue 12 Feb 2019 06:22:03 PM CST

Description: A file containing useful functions for the project

"""
import pandas as pd

def create_segment_dataframe(path_to_segment):
    """
    Create a dataframe for the segments in the given file
    :param path_to_segment: The path to the file to load
    :return: A pandas dataframe containing the data
    """
    return pd.read_csv(path_to_segment, sep=',\s+', header=2, engine="python")



