############# Module Loading ##############
from collections import OrderedDict
import csv
import argparse
import os
import numpy as np
#import matplotlib
import pandas as pd
from mpi4py import MPI
#matplotlib.use("Agg")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence, text
from clr_callback import *
from smiles_regress_transformer_funcs_large import *
from tensorflow.python.client import device_lib
import json
from smiles_pair_encoders_functions import *
import time
from tqdm import tqdm

#######HyperParamSetting#############

json_file = 'config.json'
hyper_params = ParamsJson(json_file)

######## Load model #############

model = ModelArchitecture(hyper_params).call()
model.load_weights(f'model.weights.h5')

model.save('saved_model/my_model')
