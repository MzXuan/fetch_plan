import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
in_dim = 3
out_dim = 3

num_units = 64
num_layers = 3

def MyRNNModel():
    # model: train model without last state as input;
    # #inference model with last last state and also input
    input_x= Input(shape=(None,in_dim))

    state_input_h = Input(shape=(num_units,))
    state_input_c = Input(shape=(num_units,))
    states_inputs = [state_input_h, state_input_c]

    rnn_lstm = LSTM(num_units, return_sequences=True, return_state=True)(input_x)
    for i in range(0, num_layers):
        rnn_lstm, state_h, state_c = LSTM(num_units, return_sequences=True, return_state=True)(rnn_lstm)

    train_lstm = rnn_lstm

    #todo: check how to get intermidiate value of this model

    output_train = Dense(out_dim, activation='linear')(train_lstm)

    return Model(inputs=input_x, outputs=output_train)

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


def BuildModel():
    #build my model
    model_train = MyRNNModel()
    # print(model_train.layers[0])


def TrainModel():
    '''
    Train the rnn model
    :return:
    '''
    model_train = MyRNNModel()
    model_train.compile(optimizer='rmsprop',
                        loss='categorical_crossentropy')

    x, y = CreateSeqs()
    model_train.fit(x,y)

def CreateSeqs():
    '''
    Prepare random sequences for test usage
    :return: sequences dataset
    '''
    data1 = np.random.random(size=(1, 100, 3))  # batch_size = 1, timespan = 100
    x=data1
    y=data1
    print([data1][0].shape) # (1, 20)
    return x,y


def main():
    TrainModel()
    return 0

if __name__ == '__main__':
    main()
# padding sequence and prepare to cut it