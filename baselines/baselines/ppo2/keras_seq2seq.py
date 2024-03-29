import os, sys, time, glob
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Masking, TimeDistributed, GRU, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K





def get_weights_file(checkpoint_path, file_name=None):
    #todo: get latest checkpoint file in this folder
    file_list = glob.glob(os.path.join(checkpoint_path,"weights*"))
    latest_file = max(file_list, key=os.path.getctime)

    return latest_file

def mean_absolute_error_custome(y_true, y_pred):
    # return K.mean(K.abs(y_pred - y_true), axis=-1)
    return tf.reduce_sum(tf.square(tf.norm(y_pred - y_true, ord='euclidean', axis=1)))


# epochs = 100  # Number of epochs to train for.
LSTM_ACT = 'tanh'
REC_ACT = 'hard_sigmoid'
OUTPUT_ACT = 'linear'
LOSS_MODE = 'mean_squared_error'
# LOSS_MODE = mean_absolute_error_custome
BIAS_REG = 'random_uniform'

DROPOUT = 0.1

class TrainRNN():

    def __init__(self, batch_size, in_dim, out_dim, out_timesteps, num_units, initial_epoch=0, num_layers=1,
                 directories="./pred", model_name="seq2seq", load=False):
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
        self.initial_epoch = initial_epoch

        self.max_outsteps = out_timesteps

        self._build_model()



    def _build_model(self):
        '''
        build the rnn model
        :return:

        reference: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
        '''
        t = time.time()
        print('Beginning LSTM compilation')
        dec_ini_states = [ [] for _ in range(self.num_units)]

        # enc
        enc_layers = []
        encoder_inputs = Input(shape=(None, self.in_dim), name="enc_inputs")
        masking = Masking(mask_value=0.0)(encoder_inputs)
        for i in range(0, self.num_layers):
            if i == 0:
                enc_layers.append(GRU(self.num_units, return_sequences=True, return_state=True,
                                       name="enc_"+str(i)+"lstm")(inputs=masking))
                dec_ini_states = [enc_layers[i][1]]
            else:
                enc_layers.append(GRU(self.num_units, return_sequences=True, return_state=True,
                                       name="enc_"+str(i)+"lstm")(
                                inputs = enc_layers[i - 1][0]))
                dec_ini_states += [enc_layers[i][1]]



        dec_layers = []
        # Set up the decoder, which will only process one timestep at a time.
        decoder_inputs = Input(shape=(1, self.out_dim), name = 'dec_input')
        for i in range(self.num_layers):
            dec_layers.append(GRU(self.num_units, return_sequences=True, return_state=True, name="dec_"+str(i)+"lstm"))
        decoder_dense = Dense(self.out_dim, activation=OUTPUT_ACT, name='outputs')

        all_outputs = []
        inputs = decoder_inputs
        for _ in range(self.max_outsteps):
            dec_layers_outputs = []
            # Run the decoder on one timestep
            for i, dec_layer in enumerate(dec_layers):
                if i == 0:
                    dec_layers_outputs.append(dec_layer(inputs = inputs,
                                                        initial_state = [dec_ini_states[i]]))
                else:
                    dec_layers_outputs.append(dec_layer(inputs = dec_layers_outputs[i-1][0],
                                                        initial_state = [dec_ini_states[i]]))

            for i in range(self.num_layers):
                dec_ini_states[i] = dec_layers_outputs[i][1]

            inputs = decoder_dense(dec_layers_outputs[-1][0])
            all_outputs.append(inputs)

        # Concatenate all predictions
        decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

        # Define and compile model as previously
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(optimizer='rmsprop', loss=LOSS_MODE)

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
            self.load_model()
            # try:
            #     filename=get_weights_file(modelDir, weights_name)
            #     self.model.load_weights(filename)
            #     print("load model {} successfully".format(filename))
            # except:
            #     print("failed to load model {}, please check the checkpoint directory... use default initialization setting".format(filename))


        tbCb = TensorBoard(log_dir=tfDir, histogram_freq = 1,
                                 write_graph = False, write_images = False)
        saveCb = ModelCheckpoint( os.path.join(modelDir, weights_name), monitor='val_loss', verbose=0, save_best_only=False,
                                        save_weights_only=False, mode='auto', period=2)

        # Perform batch training with epochs
        t=time.time()

        # training process
        # Prepare decoder input data that just contains the start character
        # Note that we could have made it a constant hard-coded in the model
        x_length = X.shape[0]

        decoder_input_data = np.zeros((x_length, 1, self.out_dim))
        decoder_input_data[:, 0, :] = 1.

        # Train model as previously

        self.model.fit([X, decoder_input_data], Y,
                       batch_size=self.batch_size,
                       epochs=epochs+self.initial_epoch,
                       initial_epoch = self.initial_epoch,
                       validation_split=0.2,verbose=1, callbacks=[tbCb, saveCb])


        averageTime = (time.time() - t) / epochs
        print('Total time:', time.time() - t, ', Average time per epoch:', averageTime)

    def load_model(self):
        # load model
        modelDir = os.path.join('./pred', self.model_name)
        weights_name = "weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        tfDir = os.path.join('./pred', self.model_name)

        try:
            filename = get_weights_file(modelDir, weights_name)
            self.model.load_weights(filename, by_name=True)
            print("load model {} successfully".format(filename))
        except:
            print("failed to load model in Dir {}, please check the checkpoint directory... use default initialization setting".format(modelDir))


class PredictRNN():
    def __init__(self, batch_size, in_dim, out_dim, in_timesteps, out_timesteps, num_units, num_layers=1, out_steps=100,
                 directories="./pred", model_name="seq2seq"):
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
        self.max_outsteps= out_timesteps


        self._build_model()


    def _build_model(self):
        '''
        build the rnn model
        :return:
        '''
        t = time.time()
        print('Beginning LSTM compilation')

        # The first part is unchanged
        enc_layers = []
        # encoder_inputs = Input(shape=(None, self.in_dim), name="enc_inputs")
        # encoder_inputs = Masking(mask_value=0.0)(Input(shape=(None, self.in_dim), name="enc_inputs"))
        encoder_inputs = Input(shape=(None, self.in_dim), name="enc_inputs")
        masking = Masking(mask_value=0.0)(encoder_inputs)
        for i in range(0,self.num_layers):
            if i == 0:
                enc_layers.append(GRU(self.num_units, return_sequences=True, return_state=True,
                                       name="enc_"+str(i)+"lstm")(inputs=masking))
                enc_states = [enc_layers[i][1]]
            else:
                enc_layers.append(GRU(self.num_units, return_sequences=True, return_state=True,
                                       name="enc_"+str(i)+"lstm")(
                                inputs = enc_layers[i - 1][0]))
                enc_states += [enc_layers[i][1]]


        self.encoder_model = Model(encoder_inputs, enc_states)

        # Set up the decoder, which will only process one timestep at a time.
        dec_layers = []
        decoder_inputs = Input(shape=(1, self.out_dim), name = 'dec_input')
        decoder_states_inputs = [Input(shape=(self.num_units,)) for _ in range(self.num_layers)]



        for i in range(0,self.num_layers):
            if i == 0:
                dec_layers.append(GRU(self.num_units, return_sequences=True, return_state=True,
                                       name="dec_"+str(i)+"lstm")(
                                       inputs=decoder_inputs, initial_state=[decoder_states_inputs[i]]))
                dec_states = [dec_layers[i][1]]
            else:
                dec_layers.append(GRU(self.num_units, return_sequences=True, return_state=True,
                                       name="dec_"+str(i)+"lstm")(
                                       inputs = dec_layers[i - 1][0], initial_state=[decoder_states_inputs[i]]))
                dec_states += [dec_layers[i][1]]

        decoder_dense = Dense(self.out_dim, activation=OUTPUT_ACT, name='outputs')

        # decorder
        decoder_outputs, state_c = dec_layers[-1]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + dec_states)

        print('Completed training model compilation in %.3f seconds' % (time.time() - t))



    def load_model(self):
        # load model
        modelDir = os.path.join('./pred', self.model_name)
        weights_name = "weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        tfDir = os.path.join('./pred', self.model_name)


        try:
            filename = get_weights_file(modelDir, weights_name)
            print("file name is: ", filename)
            self.encoder_model.load_weights(filename, by_name=True)
            self.decoder_model.load_weights(filename, by_name=True)
            print("load model {} successfully".format(filename))
        except:
            print("failed to load model in Dir {}, please check the checkpoint directory... use default initialization setting".format(modelDir))

    def predict(self, X, Y = None):
        '''
        The inference step
        Firstly iterate the RNN on all ground truth x;
        then predict y by the output from last step;
        :param x:
        :param y:
        :return:
        '''
        stop_condition = False
        decoded_sequence = np.zeros((self.batch_size, self.max_outsteps, self.out_dim))
        decoded_len = 0

        # encoder step
        # encoder step
        target_seq, states_value = self.get_encoder_latent_state(X)
        encoder_state_value = np.copy(np.asarray(states_value))

        #decoder inference step
        while not stop_condition:
            target_seq, states_value = self.inference_one_step(target_seq, states_value)

            # Sample a token
            decoded_sequence[:, decoded_len, :] = np.swapaxes(target_seq, 0, 1)
            decoded_len += 1
            if (decoded_len >= self.max_outsteps):
                stop_condition = True

        # print("shape of decoded sequence:", decoded_sequence.shape)
        full_sequence = np.concatenate((X, decoded_sequence), axis=1)
        return decoded_sequence, encoder_state_value

    def get_encoder_latent_state(self, inputs):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(inputs)
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((self.batch_size, 1, self.out_dim))
        # Populate the first character of target sequence with the start character.
        target_seq[:, 0, :] = 1.
        # print("encoder state value: ", encoder_state_value)
        return target_seq, states_value


    def inference_one_step(self, target_seq, states_value):
        '''

        :param target_seq: numpy array (1,1,3)
        :param states_value: list[num of layers * numpy array(1, num_units)]
        :return:
        '''

        output_result = self.decoder_model.predict(
            [target_seq] + states_value)

        output_seq = output_result[0]
        # Update the target sequence
        target_seq = output_seq
        # # Update states
        states_value = output_result[1:]

        return target_seq, states_value



def CreateSeqs(batch_size):
    '''
    Prepare random sequences for test usage
    :return: sequences dataset
    '''
    data1 = np.random.random(size=(batch_size*10, 100, 3))  # batch_size = 1, timespan = 100
    x=data1
    y=data1
    return x,y


def main():

    batch_size=1
    in_dim = 3
    out_dim = 3
    num_units = 64
    num_layers = 1
    # my_rnn = TrainRNN(batch_size, in_dim, out_dim, num_units, num_layers, 5)
    my_rnn = PredictRNN(batch_size, in_dim, out_dim, num_units, num_layers, 5)
    x, y = CreateSeqs(batch_size)
    # my_rnn.training(x,y,10)
    my_rnn.predict(x, y)
    return 0

if __name__ == '__main__':
    main()