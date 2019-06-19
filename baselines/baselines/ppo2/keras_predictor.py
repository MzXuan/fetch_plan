import os, sys, time, glob
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Masking, TimeDistributed, GRU, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import regularizers
from datetime import datetime


epochs = 100  # Number of epochs to train for.
LSTM_ACT = 'tanh'
REC_ACT = 'hard_sigmoid'
OUTPUT_ACT = 'linear'
LOSS_MODE = 'mean_squared_error'
BIAS_REG = 'random_uniform'
DROPOUT = 0.1

def get_weights_file(checkpoint_path, file_name=None):
    #todo: get latest checkpoint file in this folder
    file_list = glob.glob(os.path.join(checkpoint_path,"weights*"))
    latest_file = max(file_list, key=os.path.getctime)

    return latest_file


class TrainRNN():

    #todo: save model
    #todo: plot training and validate error

    def __init__(self, batch_size, in_dim, out_dim, num_units, num_layers=1,
                 directories="./pred", model_name="test", load=False):
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
        self.directories = directories
        self.model_name = model_name
        self.load = load

        self._build_model()


    def _build_model(self):
        '''
        build the rnn model
        :return:
        '''
        t = time.time()
        print('Beginning LSTM compilation')
        self.model = Sequential()
        # self.model.add(Input(shape=(None, self.in_dim)))
        self.model.add(Masking(mask_value=0.0, input_shape=(None, self.in_dim)))

        for i in range(0, self.num_layers):
            self.model.add(Bidirectional(GRU(self.num_units, return_sequences=True, return_state=False,
                                       activation = LSTM_ACT, recurrent_activation = REC_ACT,
                                      kernel_regularizer=regularizers.l2(0.01), bias_initializer = BIAS_REG,
                                      dropout=DROPOUT)))
        self.model.add(TimeDistributed(Dense(self.out_dim, activation=OUTPUT_ACT, bias_initializer=BIAS_REG)))

        self.model.compile(optimizer='RMSprop',
                           loss=LOSS_MODE)

        # t = time.time()
        # print('Beginning LSTM compilation')
        # input_x = Input(shape=(None, self.in_dim))
        # masked_x = Masking(mask_value=0.0, input_shape=(None, self.in_dim))(input_x)
        #
        # rnn_layers = []
        #
        # for i in range(0, self.num_layers):
        #     if i == 0:
        #         rnn_layers.append(GRU(self.num_units, return_sequences=True, return_state=True,
        #                                activation = LSTM_ACT, recurrent_activation = REC_ACT,
        #                               kernel_regularizer=regularizers.l2(0.01), bias_initializer = BIAS_REG,
        #                               dropout=DROPOUT,
        #                               name='0lstm')(
        #             inputs = masked_x))
        #     else:
        #         rnn_layers.append(GRU(self.num_units, return_sequences=True,return_state=True,
        #                                activation = LSTM_ACT, recurrent_activation = REC_ACT,
        #                               kernel_regularizer=regularizers.l2(0.01), bias_initializer = BIAS_REG,
        #                               dropout = DROPOUT,
        #                               name=str(i) + 'lstm')(
        #             inputs = rnn_layers[i - 1][0]))
        #
        # train_lstm = rnn_layers[-1][0]
        # output_layer = TimeDistributed(Dense(self.out_dim, activation=OUTPUT_ACT),name='output_layer')(inputs = train_lstm)
        # self.rnn_layers = rnn_layers
        # self.model = Model(inputs=input_x, outputs=output_layer)
        #
        # self.model.compile(optimizer='RMSprop',
        #                    loss=LOSS_MODE)

        print('Completed training model compilation in %.3f seconds' % (time.time() - t))


    def training(self, X, Y, epochs):
        '''
        Train the rnn model
        :param x: input
        :param y: output
        '''
        modelDir = os.path.join('./pred', self.model_name)
        weights_name = "weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        tfDir = os.path.join('./pred',self.model_name)
        print("tensorboard directory")
        print(tfDir)
        print("modelDir")
        print(modelDir)

        if self.load:
            try:
                filename=get_weights_file(modelDir, weights_name)
                self.model.load_weights(filename)
                print("load model {} successfully".format(filename))
            except:
                print("failed to load model, please check the checkpoint directory... use default initialization setting")


        tbCb = TensorBoard(log_dir=tfDir, histogram_freq = 1,
                                 write_graph = True, write_images = True)
        saveCb = ModelCheckpoint( os.path.join(modelDir, weights_name), monitor='val_loss', verbose=0, save_best_only=False,
                                        save_weights_only=False, mode='auto', period=40)

        # Perform batch training with epochs
        t=time.time()

        self.model.fit(X, Y, batch_size=self.batch_size, epochs=epochs, validation_split=0.1,
                       verbose=1, callbacks=[tbCb, saveCb])

        averageTime = (time.time() - t) / epochs
        print('Total time:', time.time() - t, ', Average time per epoch:', averageTime)




class PredictRNN():
    def __init__(self, batch_size, in_dim, out_dim, num_units, num_layers=1, out_steps=100,
                 directories="./pred", model_name="test"):
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
        self.directories = directories
        self.model_name = model_name
        self.max_output_steps=out_steps

        self._build_model()


    def _build_model(self):
        '''
        build the rnn model
        :return:
        '''
        # t = time.time()
        # input_x = Input(batch_shape=(self.batch_size, None, self.in_dim))
        # masked_x = Masking(mask_value=0.0, batch_input_shape=(self.batch_size, None, self.in_dim))(input_x)
        #
        # rnn_layers = []
        #
        # for i in range(0, self.num_layers):
        #     if i == 0:
        #         rnn_layers.append(GRU(self.num_units, return_sequences=True, return_state=True,
        #                                activation = LSTM_ACT, recurrent_activation = REC_ACT, stateful = True,
        #                               kernel_regularizer=regularizers.l2(0.01), bias_initializer = BIAS_REG,
        #                               dropout = DROPOUT, name='0lstm')(inputs = masked_x))
        #     else:
        #         rnn_layers.append(GRU(self.num_units, return_sequences=True,return_state=True,
        #                                activation = LSTM_ACT, recurrent_activation = REC_ACT, stateful = True,
        #                               kernel_regularizer=regularizers.l2(0.01), bias_initializer = BIAS_REG,
        #                               dropout = DROPOUT, name=str(i) + 'lstm')(inputs = rnn_layers[i - 1][0]))
        #
        # train_lstm = rnn_layers[-1][0]
        # output_layer = TimeDistributed(Dense(self.out_dim, activation=OUTPUT_ACT),name='output_layer')(inputs = train_lstm)
        # self.rnn_layers = rnn_layers
        # self.model = Model(inputs=input_x, outputs=output_layer)
        #
        # self.model.compile(optimizer='RMSprop',
        #                    loss=LOSS_MODE)

        t = time.time()
        print('Beginning LSTM compilation')
        self.model = Sequential()
        # self.model.add(Input(shape=(None, self.in_dim)))
        self.model.add(Masking(mask_value=0.0, batch_input_shape=(self.batch_size, None, self.in_dim)))

        for i in range(0, self.num_layers):
            self.model.add(Bidirectional(GRU(self.num_units, return_sequences=True, return_state=False,
                                       activation = LSTM_ACT, recurrent_activation = REC_ACT,
                                             stateful = True,
                                      kernel_regularizer=regularizers.l2(0.01), bias_initializer = BIAS_REG,
                                      dropout=DROPOUT)))
        self.model.add(TimeDistributed(Dense(self.out_dim, activation=OUTPUT_ACT, bias_initializer=BIAS_REG)))
        self.model.compile(optimizer='RMSprop',
                           loss=LOSS_MODE)
        print('Completed prediction model compilation in %.3f seconds' % (time.time() - t))

    def load_model(self):
        # load model
        modelDir = os.path.join('./pred', self.model_name)
        weights_name = "weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        tfDir = os.path.join('./pred', self.model_name)
        # print("tensorboard directory")
        # print(tfDir)
        # print("modelDir")
        # print(modelDir)

        try:
            filename = get_weights_file(modelDir, weights_name)
            self.model.load_weights(filename)
            print("load model {} successfully".format(filename))
        except:
            print("failed to load model, please check the checkpoint directory... use default initialization setting")

    def predict(self, X, Y = None):
        '''
        The inference step
        Firstly iterate the RNN on all ground truth x;
        then predict y by the output from last step;
        :param x:
        :param y:
        :return:
        '''
        # print("the shape of inputs is:")
        # print(X.shape)
        self.model.reset_states()
        predict_result = self._inference_function(inputs = X, Y = Y)
        return predict_result


    def _inference_function(self, inputs, Y = None):
        # predict_result = np.copy(inputs)
        # initial_output = self.model.predict(inputs, batch_size=self.batch_size)
        # predict_result = predict_result[~(predict_result == 0).all(2)]
        # predict_result = np.expand_dims(predict_result, axis=0)
        # predict_result = np.concatenate( (predict_result, np.expand_dims(initial_output[:,-1,:], axis=1)), axis=1)


        initial_output = self.model.predict(inputs, batch_size=self.batch_size)
        # predict_result = np.copy(initial_output)
        predict_result = np.concatenate( (np.expand_dims(inputs[:,0,:], axis=1), initial_output), axis=1)

        print("inputs")
        print(inputs)
        if Y is not None:
            print("Y:")
            print(Y)
        print("predict result")
        print(predict_result)

        for _ in range(self.max_output_steps):
            new_input = predict_result[:,-1,:]
            new_input = np.expand_dims(new_input, axis=1)

            out = self.model.predict(new_input)

            print("new_input")
            print(new_input)
            print("out")
            print(out)


            predict_result = np.concatenate((predict_result, out), axis=1)
        return predict_result


def CreateSeqs(batch_size):
    '''
    Prepare random sequences for test usage
    :return: sequences dataset
    '''
    data1 = np.random.random(size=(batch_size*10, 100, 3))  # batch_size = 1, timespan = 100
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
    my_rnn = TrainRNN(batch_size, in_dim, out_dim, num_units, num_layers, ".")
    x, y = CreateSeqs(batch_size)
    my_rnn.training(x,y,10)
    # my_rnn.run_inference(x, y)
    return 0

if __name__ == '__main__':
    main()
# padding sequence and prepare to cut it