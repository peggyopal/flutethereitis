"""
File Name: train_flute_didgeridoo_LSTM.py

Authors: Peggy Anderson & Kyle Seidenthal

Date: 29-03-2019

Description: A script to train the LSTM on the didgeridoo and flute labels

"""

from src.models.rnn import flute_didgeridoo_LSTM as FDLSTM
from src.models import process_data as pdata
import tensorflow as tf



def convert_data_to_rnn_structure(data):
    """
    Converts the dictionary inputs of the data to the (list(num_samples, 10, 128), list(num_samples)) shape required by the RNN
    
    :param data: The dictionary of data to convert
    :returns: A tuple of shape (list of shape(num_samples, 10, 128), list(num_samples))
    """

    samples = []
    labels = []
    for d in data:
        for video in d.keys():
            samples.append(d[video]['audio_embedding'])

            labels.append(d[video]['label_indices'][0])
    
    return (samples, labels)

def main():
    
    bal_train_data = pdata.get_bal_train()
    eval_data = pdata.get_eval()

    converted_bal_train = convert_data_to_rnn_structure(bal_train_data)
    converted_eval = convert_data_to_rnn_structure(eval_data)

    rnn = FDLSTM.RNN(converted_bal_train, converted_bal_train, converted_eval)
    rnn.train()

if __name__ == "__main__":

    main()
    
