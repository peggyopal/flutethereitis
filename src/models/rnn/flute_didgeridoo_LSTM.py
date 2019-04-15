"""
File Name: flute_didgeridoo_LSTM.py

Authors: Peggy Anderson & Kyle Seidenthal

Date: 29-03-2019

Description: A LSTM model to classify flutes and didgeridoos from audio features taken from Google Audioset

"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical


class FluteDidgeridooBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, classes, skip_step=5):
        """
        Initialize a generator for our data.  This will allow us to shape the data in a way that the LSTM network will
        accept later
        
        :param data: The data dictionary that we want to use in the model
        :param num_steps: The number of time steps we will put into the network
        :param batch_size: The size of batches to use
        :param classes: The number of possible outputs of the network
        :param skip_step=5: 
        :returns: None
        """
        
       

        self.data = self._mush_data_into_list(data)

        if len(self.data[0]) < batch_size:
            raise ValueError("Batch size %d too large for data size %d" % (batch_size, len(self.data[0])))

        self.num_steps = num_steps
        self.batch_size = batch_size
        self.classes = classes

        self.current_idx = 0
        
        # TODO: I think we will want to make this equal to num_steps in our case
        self.skip_step = skip_step

    def _mush_data_into_list(self, data):

        out_data = ([], [])

        for video in data.keys():
            embeddings = data[video]["audio_embedding"]
            labels = data[video]["labels"]
            
            out_data[0].append(embeddings)
            out_data[1].append(labels)

        return out_data

    def convert_label_string_to_id(self, string):

        if string.lower() == "flute":
            return 0
        else:
            return 1

    def generate(self):
        """
        Generates the next batch of values for the input data
        
        :returns: Yields the data and targets as an array
        """
        x = np.zeros((self.batch_size, self.num_steps, 128))
        y = np.zeros((self.batch_size, 1))
        
        while True:
            for i in range(self.batch_size):
                
                current_sample = np.asarray(self.data[0][self.current_idx])
                num_timesteps = current_sample.shape[0] 

                if num_timesteps < self.num_steps:
                    x[i, :num_timesteps] = current_sample
                    x[i, num_timesteps:] = np.zeros((self.num_steps - num_timesteps, 128))
                else:
                    x[i] = current_sample

                self.current_idx += 1

                if self.current_idx > len(self.data[0]) - 1:
                    self.current_idx = 0
                
                y[i] = self.convert_label_string_to_id(self.data[1][self.current_idx][0])

            yield x, y



class FDLSTM(object):

    def __init__(self, classes, hidden_size, num_steps, use_dropout=False, checkpoints=False, checkpoint_path=None):

        self.classes = classes
        self.hidden_size = hidden_size
        self.num_steps = num_steps
        self.use_dropout = use_dropout
        self.checkpoints = checkpoints
        
        if checkpoints:
            
            if checkpoint_path == None:
                raise ValueError("Checkpoint path cannot be empty if you want to use checkpoints")

            self.checkpointer = ModelCheckpoint(filepath=checkpoint_path + "/model-{epoch:02d}.hdf5", verbose=1)

        self.model = self._build_model()

    def _build_model(self):

        model = Sequential()
        model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(self.num_steps, self.hidden_size)))
       
        if self.use_dropout:
            model.add(Dropout(0.5))


        model.add(Bidirectional(LSTM(10, return_sequences=True)))
        
        if self.use_dropout:
            model.add(Dropout(0.5))

        model.add(Bidirectional(LSTM(10)))
        
        #model.add(Flatten())
        
        if self.use_dropout:
            model.add(Dropout(0.5))
        
        model.add(Dense(1, activation="sigmoid"))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary() 
        return model

    def fit(self, training_data_generator, validation_data_generator, num_epochs):

        if self.checkpoints:
            return self.model.fit_generator(training_data_generator.generate(),
                steps_per_epoch=len(training_data_generator.data[0])//(training_data_generator.batch_size),
                num_epochs=num_epochs,
                validation_data=validation_data_generator.generate(), 
                validation_steps=len(validation_data_generator.data[0])//(validation_data_generator.batch_size),
                callbacks=[self.checkpointer])
        else:
            return self.model.fit_generator(training_data_generator.generate(),
                len(training_data_generator.data[0])//(training_data_generator.batch_size),
                num_epochs,
                validation_data=validation_data_generator.generate(), 
                validation_steps=len(validation_data_generator.data[0])//(validation_data_generator.batch_size)
                )
    
    def evaluate_generator(self, generator, length):
        return self.model.evaluate_generator(generator.generate(), length)

    def predict(self, input):

        return self.model.predict(input)

    def from_json(self, path):
        raise ValueError("Kyle you need to implement this")

    def to_json(self):
        return self.model.to_json()

    def predict_generator(self, generator, length):

        return self.model.predict_generator(generator.generate(), length)
