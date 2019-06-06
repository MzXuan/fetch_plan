import os, sys, glob, gc, joblib
import numpy as np
from tensorflow.keras.models import model_from_json
if sys.version_info >= (3, 0):
    import pickle
else:
    import cPickle as pickle

dataDir = os.path.dirname(os.path.realpath(__file__))
batchDir = os.path.join(dataDir, 'minibatches')
rawDataDir = os.path.join(dataDir, 'rawdata')
modelsDir = os.path.join(dataDir, 'models')
comparisonDir = os.path.join(dataDir, 'comparison')
verbose = True

def verifyFilename(filename):
    '''
    Creates parent directories and verifies if the file already exists. Returns False if the file already exists.
    '''
    filedir = os.path.dirname(filename)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    elif os.path.isfile(filename):
        return False
    return True

def saveLSTMModel(model, modelDir, epoch):
    '''
    Saves a Keras LSTM network model architecture and weights into a specified directory.
    Returns False if the model (epoch) already exists in the specified modelDir.
    '''
    archName = None
    if epoch == 0:
        # Save model only on the first epoch
        archName = 'architectureLSTM'
    weightsName = 'weights_%d' % epoch
    # Save LSTM network
    return saveGenericModel(model, modelDir, archName=archName, weightsName=weightsName)

def saveGenericModel(model, modelDir, archName=None, weightsName=None):
    '''
    Saves a Keras neural network model architecture and weights into a specified directory.
    Returns False if the model or weights already exist in the specified modelDir.
    '''
    if archName is not None:
        # Save model architecture as JSON format
        archFilename = os.path.join(modelsDir, modelDir, '%s.json' % archName)
        if not verifyFilename(archFilename):
            print('Model architecture already exists:', weightsFilename)
            return False
        with open(archFilename, 'w') as f:
            f.write(model.to_json())

    if weightsName is not None:
        # Save model weights in HDF5 format (requires HDF5 and h5py to be install)
        weightsFilename = os.path.join(modelsDir, modelDir, '%s.h5' % weightsName)
        if not verifyFilename(weightsFilename):
            print('Weights model already exists:', weightsFilename)
            return False
        model.save_weights(weightsFilename)
    return True