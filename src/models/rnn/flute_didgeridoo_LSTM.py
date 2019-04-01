"""
File Name: flute_didgeridoo_LSTM.py

Authors: Peggy Anderson & Kyle Seidenthal

Date: 29-03-2019

Description: A LSTM model to classify flutes and didgeridoos from audio features taken from Google Audioset

"""

import tensorflow as tf

class RNN():

    def __init__(self, train_data, test_data, eval_data):
        """
        Creates a RNN for classifying flutes and didgeridoos
        
        :param train_data: The training data as (data, label) tuple, where data is in in the form of a 3-dimensional list (num_samples, 10, 128), where num
                           samples is the number of samples in the dataset.  Each of the samples will have a list of 10 timesteps, each
                           containing a list of 128 features
        :param test_data: The testing data, in the same form as train_data
        :param eval_data: The evaluation data, in the same form as train_data
        :returns: None
        """
        self.train_data = train_data
        self.test_data = test_data
        self.eval_data = eval_data
        self.num_features = 128
        self.num_timesteps = 10
        self.num_classes = 2
        self.lstm_size = 128
        self.num_epochs = 10
        self.batch_size = 1
        self.model = self._create_model()

    def _create_model(self):
        
        # Create a layer with two outputs
        layer = {'weights':tf.Variable(tf.random_normal([self.lstm_size, self.num_classes])),
                'biases':tf.Variable(tf.random_normal([self.num_classes]))}
        
        # Create a graph input for the data of size [none, 10, 128]
        x = tf.placeholder(tf.float32, [self.num_timesteps, self.num_features])
        
        # Create an LSTM cell
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
        
        state = lstm_cell.zero_state(self.num_timesteps, dtype=tf.float32)

        outputs, state = lstm_cell(x, state)

        output = tf.matmul(outputs, layer['weights']) + layer['biases']

        return output

    def train(self):
        x = tf.placeholder(tf.float32, [ self.num_timesteps, self.num_features])
        y = tf.placeholder(tf.float32)

        cost = tf.reduce_sum(tf.square(y - self.model, name="cost"))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for epoch in range(self.num_epochs):
                epoch_loss = 0

                # TODO: Batches
                for i in range(len(self.train_data[0])):
                    epoch_x = self.train_data[0][i]
                    epoch_y = self.train_data[1][i]
                    
                    print(len(epoch_x), epoch_y)
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

                    epoch_loss += c

                print("Epoch, ", epoch, " completed out of ", self.num_epochs, " loss: ", epoch_loss)

            correct = tf.equal(tf.argmax(self.model, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print("Accuracy: ", accuracy.eval({x:self.eval_data[0], y:self.eval_data[1]}))

