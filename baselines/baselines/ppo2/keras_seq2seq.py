import os, sys, time, glob
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Masking, TimeDistributed, GRU, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from attention import AttentionLayer

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

def slice_layer(x):
    return x[:,1:,:]


# epochs = 100  # Number of epochs to train for.
LSTM_ACT = 'tanh'
REC_ACT = 'hard_sigmoid'
OUTPUT_ACT = 'linear'
# LOSS_MODE = 'mean_squared_error'
LOSS_MODE = mean_absolute_error_custome
BIAS_REG = 'random_uniform'

DROPOUT = 0.1




class TrainRNN():

    def __init__(self, batch_size, in_dim, out_dim, in_timesteps , out_timesteps, num_units, num_layers=1,
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

        self.in_timesteps = in_timesteps
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
        dec_ini_states = [ [] for _ in range(2*self.num_units)]

        # enc
        enc_layers = []
        encoder_inputs = Input(shape=(self.in_timesteps, self.in_dim), name="enc_inputs")
        for i in range(0, self.num_layers):
            if i == 0:
                enc_layers.append(LSTM(self.num_units, return_sequences=True, return_state=True,
                                       name="enc_"+str(i)+"lstm")(inputs=encoder_inputs))
                dec_ini_states = [enc_layers[i][1], enc_layers[i][2]]
            else:
                enc_layers.append(LSTM(self.num_units, return_sequences=True, return_state=True,
                                       name="enc_"+str(i)+"lstm")(
                                inputs = enc_layers[i - 1][0]))
                dec_ini_states += [enc_layers[i][1], enc_layers[i][2]]
        enc_output = enc_layers[-1][0]

        #dec
        dec_layers = []

        # Set up the decoder, which will only process one timestep at a time.
        decoder_inputs = Input(shape=(1, self.out_dim), name = 'dec_input')
        for i in range(self.num_layers):
            dec_layers.append(LSTM(self.num_units, return_sequences=True, return_state=True, name="dec_"+str(i)+"lstm"))
        decoder_dense = Dense(self.out_dim, activation=OUTPUT_ACT, name='outputs')
        #att
        attn_layer = AttentionLayer(name="atten")

        all_outputs = []
        inputs = decoder_inputs
        for steps in range(self.max_outsteps):
            dec_layers_outputs = []
            # Run the decoder on one timestep
            for i, dec_layer in enumerate(dec_layers):
                if i == 0:
                    dec_layers_outputs.append(dec_layer(inputs = inputs,
                                                        initial_state = [dec_ini_states[2*i], dec_ini_states[2*i+1]]))
                else:
                    dec_layers_outputs.append(dec_layer(inputs = dec_layers_outputs[i-1][0],
                                                        initial_state = [dec_ini_states[2 * i], dec_ini_states[2 * i + 1]]))

            for i in range(self.num_layers):
                dec_ini_states[2*i] = dec_layers_outputs[i][1]
                dec_ini_states[2*i+1] = dec_layers_outputs[i][2]

            dec_h = dec_layers_outputs[-1][0]
            attn_out, attn_states = attn_layer([enc_output, dec_h])
            #output
            inputs = decoder_dense(attn_out)
            all_outputs.append(inputs)


            # # for attention update, send output to new attention enc states
            # enc_output = Lambda(slice_layer)(enc_output)
            # enc_output = tf.keras.layers.concatenate([enc_output, dec_h], axis=1)


        # Concatenate all predictions
        decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

        print("decoder_outputs")
        print(decoder_outputs)



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
            try:
                filename=get_weights_file(modelDir, weights_name)
                self.model.load_weights(filename)
                print("load model {} successfully".format(filename))
            except:
                print("failed to load model, please check the checkpoint directory... use default initialization setting")


        tbCb = TensorBoard(log_dir=tfDir, histogram_freq = 1,
                                 write_graph = True, write_images = True)
        saveCb = ModelCheckpoint( os.path.join(modelDir, weights_name), monitor='val_loss', verbose=0, save_best_only=False,
                                        save_weights_only=False, mode='auto', period=1)

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
                       epochs=epochs,
                       validation_split=0.2,verbose=1, callbacks=[tbCb, saveCb])

        # self.model.fit(X, Y, batch_size=self.batch_size, epochs=epochs, validation_split=0.1,
        #                verbose=1, callbacks=[tbCb, saveCb])

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
            print("failed to load model {}, "
                  "please check the checkpoint directory... use default initialization setting".format(filename))




class PredictRNN():
    def __init__(self, batch_size, in_dim, out_dim, in_timesteps, out_timesteps, num_units, num_layers=1, out_steps=100,
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

        self.in_timesteps = in_timesteps
        self.max_outsteps= 4*in_timesteps


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
        encoder_inputs = Input(shape=(self.in_timesteps, self.in_dim), name="enc_inputs")
        for i in range(0,self.num_layers):
            if i == 0:
                enc_layers.append(LSTM(self.num_units, return_sequences=True, return_state=True,
                                       name="enc_"+str(i)+"lstm")(inputs=encoder_inputs))
                enc_states = [enc_layers[i][1], enc_layers[i][2]]
            else:
                enc_layers.append(LSTM(self.num_units, return_sequences=True, return_state=True,
                                       name="enc_"+str(i)+"lstm")(
                                inputs = enc_layers[i - 1][0]))
                enc_states += [enc_layers[i][1], enc_layers[i][2]]
        enc_output = [enc_layers[-1][0]]

        self.encoder_model = Model(encoder_inputs, enc_output + enc_states)

        # Set up the decoder, which will only process one timestep at a time.
        dec_layers = []
        decoder_inputs = Input(shape=(1, self.out_dim), name = 'dec_input')
        decoder_states_inputs = [Input(shape=(self.num_units,)) for _ in range(2*self.num_layers)]

        enc_output = Input(shape=(self.in_timesteps, self.num_units))
        #att
        attn_layer = AttentionLayer(name="atten")

        for i in range(0,self.num_layers):
            if i == 0:
                dec_layers.append(LSTM(self.num_units, return_sequences=True, return_state=True,
                                       name="dec_"+str(i)+"lstm")(
                                       inputs=decoder_inputs, initial_state=[decoder_states_inputs[i],decoder_states_inputs[i+1]]))
                dec_states = [dec_layers[i][1], dec_layers[i][2]]
            else:
                dec_layers.append(LSTM(self.num_units, return_sequences=True, return_state=True,
                                       name="dec_"+str(i)+"lstm")(
                                       inputs = dec_layers[i - 1][0], initial_state=[decoder_states_inputs[2*i],decoder_states_inputs[2*i+1]]))
                dec_states += [dec_layers[i][1], dec_layers[i][2]]

        decoder_dense = Dense(self.out_dim, activation=OUTPUT_ACT, name='outputs')
        dec_output = [dec_layers[-1][0]] #output from lstm cells


        # decorder && attention
        decoder_outputs, state_h, state_c = dec_layers[-1]
        attn_out, attn_states = attn_layer([enc_output, decoder_outputs])
        decoder_outputs = decoder_dense(attn_out) #output after dense
        self.decoder_model = Model(
            [decoder_inputs] + [enc_output] + decoder_states_inputs,
            [decoder_outputs] + dec_output + dec_states)

        print('Completed training model compilation in %.3f seconds' % (time.time() - t))



    def load_model(self):
        # load model
        modelDir = os.path.join('./pred', self.model_name)
        weights_name = "weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        tfDir = os.path.join('./pred', self.model_name)

        try:
            filename = get_weights_file(modelDir, weights_name)
            self.encoder_model.load_weights(filename, by_name=True)
            self.decoder_model.load_weights(filename, by_name=True)
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

        # self.model.reset_states()
        predict_result = self._inference_function(inputs = X, Y = Y)
        return predict_result


    def _inference_function(self, inputs, Y = None):
        stop_condition = False
        decoded_sequence = []
        enc_in_bc = np.empty((self.batch_size, self.in_timesteps, self.out_dim), dtype = np.float32)

        step = 0

        while not stop_condition:
            if step % self.in_timesteps == 0:

                target_seq = np.zeros((1, 1, self.out_dim))
                target_seq[0, 0, :] = 1.

                if step == 0:
                    enc_input = inputs

                else:
                    enc_input = np.asarray(enc_in_bc)
                    enc_in_bc = np.empty((self.batch_size, self.in_timesteps, self.out_dim), dtype = np.float32)


                # Encode the input as state vectors.
                enc_output_result = self.encoder_model.predict(enc_input)
                enc_output = enc_output_result[0]
                states_value = enc_output_result[1:]


            output_result = self.decoder_model.predict(
                [target_seq] + [enc_output] + states_value)


            output_seq = output_result[0]
            # Sample a token
            decoded_sequence.append(output_seq[0,0,:])

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.out_dim))
            target_seq[0, 0, :] = output_seq

            # # Update states
            states_value = output_result[2:]

            enc_in_bc[:,step%self.in_timesteps,:] = output_seq



            # enc_output = np.delete(enc_output,0,1)
            # enc_output = np.concatenate((enc_output, output_result[1]), axis=1)




            step += 1
            if step >= self.max_outsteps:
                stop_condition = True

        decoded_sequence = np.asarray(decoded_sequence)
        decoded_sequence = decoded_sequence.reshape((self.batch_size, self.max_outsteps, self.out_dim))

        full_sequence = np.concatenate((inputs, decoded_sequence), axis=1)

        return full_sequence


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
    my_rnn = TrainRNN(batch_size, in_dim, out_dim, num_units, num_layers, 5)
    x, y = CreateSeqs(batch_size)
    my_rnn.training(x,y,10)
    # my_rnn.run_inference(x, y)
    return 0

if __name__ == '__main__':
    main()