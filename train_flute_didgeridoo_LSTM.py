"""
File Name: train_flute_didgeridoo_LSTM.py

Authors: Peggy Anderson & Kyle Seidenthal

Date: 29-03-2019

Description: A script to train the LSTM on the didgeridoo and flute labels

"""
from sklearn.model_selection import KFold
from src.models.rnn import flute_didgeridoo_LSTM as FDLSTM
from src.models import process_data as pdata
import random
import numpy as np
import datetime
import argparse
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

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
NUM_EPOCHS = 20


INTERMEDIATE_OUTPUT_FOLDER = os.path.join("intermediate_results", "rnn")

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


def _get_train_data(dataset):

    if dataset == "bal_train":
        return pdata.get_bal_train()
        
    elif dataset == "unbal_train":
        return pdata.get_unbal_train()
         
    else:
        raise ValueError("Invalid dataset %s" % dataset)

def _get_train_and_test_dicts(train_index, test_index, data):
        
    train_keys = []
    test_keys = []
        
    for index in train_index:
        train_keys.append(list(data.keys())[index])

    for index in test_index:
        test_keys.append(list(data.keys())[index])

    train_dict = {}
    test_dict = {}

    for key in train_keys:
        train_dict[key] = data[key]

    for key in test_keys:
        test_dict[key] = data[key]

    return train_dict, test_dict

def _create_out_dir(dataset):
    # Create output dir
    date = datetime.datetime.now()
    save_folder = date.strftime('%Y-%m-%d_%H:%M:%S')
    out_path = os.path.join(INTERMEDIATE_OUTPUT_FOLDER, dataset, save_folder)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    return out_path

def _train_10_fold_cross_val(input_train_data, out_path):
    
    # Use a 10-fold cross-evaluation generator
    kf = KFold(n_splits=10)

    trained_models = []
    
    histories = []


    model_name = 0


    # Train 10 models
    for train_index, test_index in kf.split(input_train_data):
        
        # Split the data into the training and testing sets
        train_dict, test_dict = _get_train_and_test_dicts(train_index, test_index, input_train_data)

        train_data_generator = FDLSTM.FluteDidgeridooBatchGenerator(train_dict, TIME_STEPS, BATCH_SIZE,
            NUM_CLASSES, skip_step=SKIP_STEP)

        test_data_generator = FDLSTM.FluteDidgeridooBatchGenerator(test_dict, TIME_STEPS, BATCH_SIZE,
            NUM_CLASSES, skip_step=SKIP_STEP)

        model = FDLSTM.FDLSTM(NUM_CLASSES, FEATURE_LENGTH, TIME_STEPS, use_dropout=True)

        history = model.fit(train_data_generator, test_data_generator, NUM_EPOCHS)
       
        histories.append(history.history)
        model_json = model.to_json()
 
        

        with open(os.path.join(out_path, "model_" + str(model_name) +".json"), 'w') as file:
            file.write(model_json)
         
        
        trained_models.append(model)
        model_name += 1

    _save_average_histories(histories, out_path)

    return trained_models

def _save_average_histories(histories, out_dir):

    avg_histories = {}

    losses = []
    accuracies = []
    val_losses = []
    val_accuracies = []

    for history in histories:
        losses.append(history['loss'])
        accuracies.append(history['acc'])
        val_losses.append(history['val_loss'])
        val_accuracies.append(history['val_acc'])
 
    avg_histories['loss'] = np.mean(losses, axis=0).tolist()
    avg_histories['acc'] = np.mean(accuracies, axis=0).tolist()
    avg_histories['val_loss'] = np.mean(val_losses, axis=0).tolist()
    avg_histories['val_acc'] = np.mean(val_accuracies, axis=0).tolist()

    with open(os.path.join(out_dir, "avg_history.json"), 'w') as fp:
        json.dump(avg_histories, fp)

    plt.figure()
    plt.plot(avg_histories['loss'])
    plt.plot(avg_histories['val_loss'])
    plt.title("Model Train vs Validation Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(os.path.join(out_dir, "Loss_Plot.png"))

    plt.figure()
    plt.plot(avg_histories['acc'])
    plt.plot(avg_histories['val_acc'])
    plt.title("Model Train vs Validation Accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(os.path.join(out_dir, "Accuracy_Plot.png"))



def _evaluate_model(trained_models, eval_data, out_path):
    
    # Prepare eval data
    eval_data = convert_list_of_dicts_to_dict(eval_data)
    eval_data_generator = FDLSTM.FluteDidgeridooBatchGenerator(eval_data, TIME_STEPS, 1, NUM_CLASSES, skip_step=10)

    num_correct = 0

    correct_ys = [eval_data_generator.convert_label_string_to_id(x[0]) for x in eval_data_generator.data[1]]

    predicted_ys = []

    for model in trained_models:
        predicted_ys.append(model.predict_generator(eval_data_generator,
            len(eval_data_generator.data[0])/eval_data_generator.batch_size))
   
   
    avg_pred_probs = np.mean(predicted_ys, axis=0)
    avg_pred = np.around(avg_pred_probs)

    conf_matrix = confusion_matrix(correct_ys, avg_pred)
    
    for i in range(len(correct_ys)):
        if avg_pred[i] == correct_ys[i]:
            num_correct += 1

    accuracy = num_correct/len(correct_ys)
    print("Accuracy: %.2f" % accuracy)
    print("Confusion Matrix: ")
    print(conf_matrix)
   
    intermediate_results = {"predictions": avg_pred, "prediction_probs": avg_pred_probs, "confusion_matrix": conf_matrix}

    _save_intermediate_results(intermediate_results, out_path)

def _save_intermediate_results(dictionary, out_path):

    for key in dictionary.keys():
        np.save(os.path.join(out_path, key), dictionary[key])


def train(dataset):
    
    # Grab the training data
    raw_input_train_data = _get_train_data(dataset)
    input_train_data = convert_list_of_dicts_to_dict(raw_input_train_data)
    
    # Grab Evaluation data
    eval_data = pdata.get_eval()
   
    out_dir = _create_out_dir(dataset)
    
    # Train the models using 10-fold cross-validation
    trained_models = _train_10_fold_cross_val(input_train_data, out_dir)

    _evaluate_model(trained_models, eval_data, out_dir)

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the FDLSTM')
    parser.add_argument('dataset')

    args = parser.parse_args()

    train(args.dataset)
    

