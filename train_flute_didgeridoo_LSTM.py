"""
File Name: train_flute_didgeridoo_LSTM.py

Authors: Peggy Anderson & Kyle Seidenthal

Date: 29-03-2019

Description: A script to train the LSTM on the didgeridoo and flute labels

"""

from src.models.rnn import flute_didgeridoo_LSTM as FDLSTM
from src.models import process_data as pdata
import random
import numpy as np

# The percent of training vs testing data to split into
TRAIN_PROB_PERCENT = 70

# The number of time samples in the data samples
TIME_STEPS = 10

# The batch size to use
BATCH_SIZE = 5

# The number of different classes to classify (flute and didgeridoo)
NUM_CLASSES = 2

# For ensuring that we shift the time series properly with the time steps
SKIP_STEP=10

# The length of each feature vector
FEATURE_LENGTH = 128

# The number of epochs to train for
NUM_EPOCHS = 5

def train_test_split(data):

    
    train = {}
    test = {}

    for record in data:
        for video in record.keys():
            if random.randint(0, 100) < TRAIN_PROB_PERCENT:
                train[video] = record[video]
            else:
                test[video] = record[video]

    return train, test

def convert_list_of_dicts_to_dict(data):

    output = {}
    for record in data:
        for video in record.keys():
            output[video] = record[video]

    return output

# split data into train and valid
#train_data = pdata.get_unbal_train()
train_data = pdata.get_bal_train()

train_data, valid_data = train_test_split(train_data)

eval_data = pdata.get_eval()

train_data_generator = FDLSTM.FluteDidgeridooBatchGenerator(train_data, TIME_STEPS, BATCH_SIZE, NUM_CLASSES,
        skip_step=SKIP_STEP)
valid_data_generator = FDLSTM.FluteDidgeridooBatchGenerator(valid_data, TIME_STEPS, BATCH_SIZE, NUM_CLASSES,
        skip_step=SKIP_STEP)

model = FDLSTM.FDLSTM(NUM_CLASSES, FEATURE_LENGTH, TIME_STEPS)

model.fit(train_data_generator, valid_data_generator, NUM_EPOCHS)

eval_data = convert_list_of_dicts_to_dict(eval_data)
eval_data_generator = FDLSTM.FluteDidgeridooBatchGenerator(eval_data, TIME_STEPS, 1, NUM_CLASSES, skip_step=10)

num_correct = 0

for i in range(len(eval_data_generator.data[0])):
    data = next(eval_data_generator.generate())
    prediction = model.predict(data[0])
    

    if np.array_equal(prediction, data[1]):
        num_correct += 1

print("Evaluation Accuracy: %.2f%%" % (num_correct/len(eval_data_generator.data[0]))) 

# TODO: Evaluate the trained model
