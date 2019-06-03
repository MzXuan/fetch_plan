import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras import backend as K
import numpy as np


epochs = 100  # Number of epochs to train for.


class MyRNN():
    #todo: how to set initial state?
    #todo: add masking layer to support dynamic rnn of this model
    def __init__(self, in_dim, out_dim, num_units, num_layers):
        '''
        initialize my rnn model
        :param in_dim: feature dimension of input data
        :param out_dim: feature dimension of output data
        :param num_units: cell units number of rnn layers
        :param num_layers: number of stacked layers of rnn layers
        '''
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_units = num_units
        self.num_layers = num_layers


    def _build_model(self):
        '''
        build the rnn model
        :return:
        '''
        input_x = Input(shape=(None, self.in_dim))

        state_input_h = Input(shape=(self.num_units,))
        state_input_c = Input(shape=(self.num_units,))
        rnn_layers = []

        for i in range(0, self.num_layers):
            if i == 0:
                rnn_layers.append(LSTM(self.num_units, return_sequences=True, return_state=True, name='0lstm')(
                    input_x))
            else:
                rnn_layers.append(LSTM(self.num_units, return_sequences=True,
                                       return_state=True, name=str(i + 1) + 'lstm')(rnn_layers[i - 1][0]))

        train_lstm = rnn_layers[-1][0]
        output_train = Dense(self.out_dim, activation='linear')(train_lstm)

        self.rnn_layers = rnn_layers
        self.model = Model(inputs=input_x, outputs=output_train)

    def run_training(self, x, y):
        '''
        Train the rnn model
        :param x: input
        :param y: output
        '''
        self.model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy')
        self.model.fit(x, y)

        # get intermediate value
        get_lstm_state = K.function([self.model.layers[0].input],
                                    [self.model.get_layer('0lstm').output[1], self.model.get_layer('1lstm').output[1]])
        print("lstm output: ")
        output = get_lstm_state([x])
        print(output)
        print("shape of the state: ")
        print(output[0].shape)

    def run_inference(self, x, y):
        '''
        The inference step
        Firstly iterate the RNN on all ground truth x;
        then predict y by the output from last step;
        :param x:
        :param y:
        :return:
        '''
        # get intermediate value
        get_lstm_state = K.function([self.model.layers[0].input], [self.model.layers[1].output[1]])
        print("lstm output: ")
        output = get_lstm_state([x])
        print(output)
        print("shape of the state: ")
        print(output[0].shape)

        #set initial state of rnn layers


# def MyRNNModel():
#     # model: train model without last state as input;
#     # #inference model with last last state and also input
#     input_x= Input(shape=(None,in_dim))
#
#     state_input_h = Input(shape=(num_units,))
#     state_input_c = Input(shape=(num_units,))
#     states_inputs = [state_input_h, state_input_c]
#
#     for i in range(0, num_layers):
#         if i == 0:
#             rnn_lstm, state_h, state_c= LSTM(num_units, return_sequences=True, return_state=True, name='0lstm')(input_x)
#             # print("haha")
#         else:
#             rnn_lstm, state_h, state_c = LSTM(num_units, return_sequences=True,
#                                           return_state=True, name=str(i+1)+'lstm')(rnn_lstm)
#
#     train_lstm = rnn_lstm
#
#
#
#     output_train = Dense(out_dim, activation='linear')(train_lstm)
#
#     return Model(inputs=input_x, outputs=output_train)

    # train_lstm, state_h, state_c = rnn_lstm(input_x)
    # inference_lstm, state_h, state_c = rnn_lstm(input_x,  initial_state=states_inputs)
    #
    # states_outputs =[state_h, state_c]
    #
    # output_train = Dense(out_dim, activation='linear')(train_lstm)
    # output_inference = Dense(out_dim, activation='linear')(inference_lstm)
    #
    # return Model(inputs = input_x, outputs = [states_outputs] + output_train), \
    #        Model(inputs = [states_inputs]+input_x, outputs = [states_outputs] + output_inference)

def CreateSeqs():
    '''
    Prepare random sequences for test usage
    :return: sequences dataset
    '''
    data1 = np.random.random(size=(1, 100, 3))  # batch_size = 1, timespan = 100
    x=data1
    y=data1
    # print([data1][0].shape) # (1, 20)
    return x,y


def main():

    batch_size=64
    in_dim = 3
    out_dim = 3
    num_units = 64
    num_layers = 2
    my_rnn = MyRNN(in_dim, out_dim, num_units, num_layers)
    x, y = CreateSeqs()
    my_rnn.run_training(x, y)
    return 0

if __name__ == '__main__':
    main()
# padding sequence and prepare to cut it