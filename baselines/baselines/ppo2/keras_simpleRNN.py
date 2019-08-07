import os, sys, time, glob
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Masking, TimeDistributed, GRU, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


def get_weights_file(checkpoint_path, file_name=None):
    #todo: get latest checkpoint file in this folder
    file_list = glob.glob(os.path.join(checkpoint_path,"weights*"))
    latest_file = max(file_list, key=os.path.getctime)

    return latest_file


class SimpleRNN():
    def __init__(self, batch_size,  in_dim=3, out_dim=3, directories="./simpleRNN", model_name="test", load=False):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_units = 16
        self.batch_size = batch_size
        self.directories = directories
        self.model_name = model_name
        self.load = load

        self._build_model()

    def _build_model(self):
        '''
        build the simple rnn model
        :return:
        '''
        t = time.time()
        print('Begin to build the simple rnn model...')
        self.model = Sequential()
        self.model.add(Masking(mask_value=0.0, input_shape=(None, self.in_dim)))

        self.model.add(GRU(self.num_units, return_sequences=True))
        self.model.add(GRU(self.num_units, return_sequences=False))

        self.model.add(Dense(self.out_dim))

        self.model.compile(optimizer='RMSprop', loss='mean_squared_error')

        print('Completed simple rnn model compilation in %.3f seconds' % (time.time() - t))

    def training(self, X, Y, epochs):
        '''
        :param X: input
        :param Y: output
        :param epochs: joint training epochs
        :return:
        '''
        modelDir = os.path.join(self.directories, self.model_name)
        weights_name = "weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        tfDir = os.path.join(self.directories, self.model_name)
        print("tensorboard directory")
        print(tfDir)
        print("modelDir")
        print(modelDir)

        if self.load:
            try:
                filename = get_weights_file(modelDir, weights_name)
                self.model.load_weights(filename)
                print("load model {} successfully".format(filename))
            except:
                print(
                    "failed to load model, please check the checkpoint directory... use default initialization setting")

        tbCb = TensorBoard(log_dir=tfDir, histogram_freq=1,
                           write_graph=True, write_images=True)
        saveCb = ModelCheckpoint(os.path.join(modelDir, weights_name), monitor='val_loss', verbose=0,
                                 save_best_only=False,
                                 save_weights_only=False, mode='auto', period=2)

        # Perform batch training with epochs
        t = time.time()

        self.model.fit(X, Y, batch_size=self.batch_size, epochs=epochs, validation_split=0.1,
                       verbose=1, callbacks=[tbCb, saveCb])

        averageTime = (time.time() - t) / epochs
        print('Total time:', time.time() - t, ', Average time per epoch:', averageTime)

    def predict(self, X, Y = None):
        predict_result = self.model.predict(X, batch_size=self.batch_size)
        if Y is not None:
            print("Y:")
            print(Y)
        print("predict result")
        print(predict_result)

        return predict_result


# for testing
def CreateSeqs(batch_size):
    '''
    Prepare random sequences for test usage
    :return: sequences dataset
    '''
    x = np.random.random(size=(batch_size*10,3,3))
    y = np.random.random(size=(batch_size*10,3))
    # print([data1][0].shape) # (1, 20)
    return x,y


def main():
    batch_size=1
    in_dim = 3
    out_dim = 3
    my_rnn = SimpleRNN(batch_size, in_dim, out_dim)
    x, y = CreateSeqs(batch_size)
    my_rnn.training(x,y,10)
    my_rnn.predict(x,y)
    return 0

if __name__ == '__main__':
    main()

