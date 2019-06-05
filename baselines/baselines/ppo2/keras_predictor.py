import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Masking
from tensorflow.keras import backend as K
import numpy as np


epochs = 100  # Number of epochs to train for.



class MyRNN():
    #todo: how to set initial state?
    #todo: add masking layer to support dynamic rnn of this model
    #todo: plot training and validate error
    def __init__(self, in_dim, out_dim, num_units, num_layers, batch_size):
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
        self.batch_size = batch_size
        self._build_model()

    def _build_model(self):
        '''
        build the rnn model
        :return:
        '''
        #todo: add multiple model

        input_x = Input(shape=(None, self.in_dim))
        input_state_h = Input(batch_shape=(self.batch_size, self.num_units))
        input_state_c = Input(batch_shape=(self.batch_size, self.num_units))
        masked_x = Masking(mask_value=0.0, input_shape=(None, self.in_dim))(input_x)

        rnn_layers = []


        for i in range(0, self.num_layers):
            if i == 0:
                rnn_layers.append(LSTM(self.num_units, return_sequences=True, return_state=True,name='0lstm')(
                    inputs = masked_x, initial_state=[input_state_h, input_state_c]))
            else:
                rnn_layers.append(LSTM(self.num_units, return_sequences=True,return_state=True,name=str(i) + 'lstm')(
                    inputs = rnn_layers[i - 1][0], initial_state=[input_state_h, input_state_c]))


        train_lstm = rnn_layers[-1][0]
        output_layer = Dense(self.out_dim, activation='linear', name='output_layer')(train_lstm)
        self.rnn_layers = rnn_layers


        self.model = Model(inputs=[input_x, input_state_h, input_state_c], outputs=output_layer)
        # self.model = Model(inputs=input_x, outputs=output_layer)
        # self.inference_model

    def run_training(self, x, y):
        '''
        Train the rnn model
        :param x: input
        :param y: output
        '''
        self.model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy')
        state_h, state_c = self._create_initial_state(x.shape[0])
        self.model.fit([x, state_h, state_c], y)
        print("training done")

    def _create_initial_state(self, length):
        state_h = np.zeros((length,self.num_units), dtype=np.float32)
        state_c = np.zeros((length, self.num_units), dtype=np.float32)
        return state_h, state_c

    def run_inference(self, x, y = None):
        '''
        The inference step
        Firstly iterate the RNN on all ground truth x;
        then predict y by the output from last step;
        :param x:
        :param y:
        :return:
        '''
        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy')
        output = self.model.predict(x)

        # get intermediate value
        h_list = [rnn_states[1] for rnn_states in self.rnn_layers]
        c_list = [rnn_states[2] for rnn_states in self.rnn_layers]


        get_lstm_state_h = K.function([self.model.layers[0].input], h_list)
        get_lstm_state_c = K.function([self.model.layers[0].input], c_list)

        state_h = get_lstm_state_h([x])  # h1,h2,h3..., hn * batch_size * num_units
        state_c = get_lstm_state_c([x])  # c1,c2,c3..., cn * batch_size * num_units

        print("lstm output: ")
        print(state_h)  # output: list [? * output]
        print("shape of the state: ")
        print(state_h[0].shape)        # get intermediate value


        # state_list = []
        # for layer_state  in self.rnn_layers:
        #     state_list.append(layer_state[1])
        #     state_list.append(layer_state[2])
        #
        # get_lstm_state = K.function([self.model.layers[0].input], state_list)
        # state = get_lstm_state([x]) #h1, c1; h2, c2;... hn;cn * batch_size * num_units
        #
        # print("lstm output: ")
        # print(state)  # output: list [? * output]
        # print("shape of the state: ")
        # print(state[0].shape)

        for _ in range(50):
            x, state_h, state_c = self._inference_function(inputs = x, initial_states = [state_h, state_c])

    def _inference_function(self, inputs, initial_states):
        # masked_x = K.function([self.model.layers[0].input], [self.model.layers[0]])([inputs])
        new_states_h = []
        new_states_c = []

        inputs = inputs.astype(np.float32)
        print("shape of input")
        print(inputs.shape)

        for batch_idx in range(inputs.shape[0]):

            for idx in range(self.num_layers):
                stacked_lstm = self.model.get_layer(str(idx)+"lstm")

                state = np.stack((initial_states[0][idx][batch_idx], initial_states[1][idx][batch_idx]), axis=0)
                current_state = K.variable(value=state)
                print("current state shape: ")
                print(current_state.shape)
                # current_state = current_state.tolist()

                if idx == 0:
                    out, state_h, state_c = \
                        stacked_lstm.call(inputs = np.expand_dims(inputs[batch_idx], axis=0),
                                          initial_state=current_state)
                else:
                    out, state_h, state_c = \
                        stacked_lstm.call(inputs = out, initial_states=current_state)
                print("iteration: ", idx)
                new_states_h.append(state_h)
                new_states_c.append(state_c)
                outputs = self.model.get_layer("output_layer").call(inputs=out)

        return outputs, new_states_h, new_states_c




def CreateSeqs(batch_size):
    '''
    Prepare random sequences for test usage
    :return: sequences dataset
    '''
    data1 = np.random.random(size=(batch_size, 100, 3))  # batch_size = 1, timespan = 100
    x=data1
    y=data1
    # print([data1][0].shape) # (1, 20)
    return x,y


def main():

    batch_size=1
    in_dim = 3
    out_dim = 3
    num_units = 64
    num_layers = 1
    my_rnn = MyRNN(in_dim, out_dim, num_units, num_layers, batch_size)
    x, y = CreateSeqs(batch_size)
    my_rnn.run_training(x,y)
    # my_rnn.run_inference(x, y)
    return 0

if __name__ == '__main__':
    main()
# padding sequence and prepare to cut it