# Data Characterization
We are using Google's AudioSet Dataset to train, test, and evaluate our CMPT820
machine learning class project. This dataset contains 2.084,320 10-second
annotated YouTube videos. Since our project is focused on the identification of
musical instruments, there is a lot of data in Google's Dataset that we do not
require, such as snoring, alarms, vehicle noise, etc. Therefore, there are
scripts in this directory that are designed to clean the data to only include
relevant audio samples.

## How to clean the data
1. run `./clean_it_all`
    - this will perform the following:
        - `clean_labels.py`: compares a predetermined (by us) list of strings
          ['flute'] with the list of labels included
          in dataset to create a new `class_labels_indicies_cleaned.csv` file of
          labels only containing words that match our allowed strings. Such as,
          'flute', or any additional strings.
        - `clean_segment.py`: extract the identification code associated to each
          label from class_labels_indicies_cleaned.csv and copy all rows from
          Google's CSV of samples that contains the relevant label
          identification codes into a `*_cleaned.csv`
        - `'clean_embeddings.py'`: makes a copy of all samples from Google's
          dataset and saves it into the clean_data directory
