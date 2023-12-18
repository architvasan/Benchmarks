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
import sys

# Limit GPU memory growth
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#    try:
#        for gpu in gpus:
#            tf.config.experimental.set_memory_growth(gpu, True)
#    except RuntimeError as e:
#        print(e)

#######HyperParamSetting#############

json_file = 'config.json'
hyper_params = ParamsJson(json_file)

######## Set up MPI #############


comm, size, rank = initialize_mpi()

######## Load model #############

print(f"rank is {rank}")
#model = tf.keras.models.load_model('saved_model/my_model')
model = ModelArchitecture(hyper_params).call()
model.load_weights(f'model.weights.h5')
model.summary()
#sys.exit()

####### Oranize data files #########

split_files, split_dirs = large_scale_split(hyper_params, size, rank)
print(f"{rank}:len(split_files)")
##### Set up tokenizer ########
if hyper_params['tokenization']['tokenizer']['category'] == 'smilespair':
    vocab_file = hyper_params['tokenization']['tokenizer']['vocab_file']
    spe_file = hyper_params['tokenization']['tokenizer']['spe_file']
    tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)
print(f"{rank}: tokenizer set")

####### Iterate over files ##############
BATCH = hyper_params['general']['batch_size']
cutoff = 0
start_total = time.time()

for fil, dirs in zip(split_files, split_dirs):

    if True:
        try:
            Data_smiles_inf, x_inference = large_inference_data_gen(hyper_params, tokenizer, dirs, fil, rank)
            print(f"{rank}: inference set")
            
            #Data_smiles_inf_split = np.array_split(x_inference, 4)
            Output = model.predict(x_inference, batch_size = BATCH)#, verbose=0)

            '''
            Combine SMILES and predicted docking score.
            Sort the data based on the docking score,
            remove data below cutoff score.
            write data to file in output directory
            '''
            SMILES_DS = np.vstack((Data_smiles_inf, np.array(Output).flatten())).T
            SMILES_DS = sorted(SMILES_DS, key=lambda x: x[1], reverse=True)

            filtered_data = list(OrderedDict((item[0], item) for item in SMILES_DS if item[1] >= cutoff).values())
            filename = f'output/{dirs}/{os.path.splitext(fil)[0]}.dat'
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['smiles', 'score'])
                writer.writerows(filtered_data)

            del (Data_smiles_inf)
            del(Output)
            del(x_inference)
            del(SMILES_DS)
            del(filtered_data)
        except:
            continue




    #except:
    #    break
        #continue

end_total = time.time()

print(f"total time to go through pipeline is {end_total - start_total}")
file1 = open(f"time_info_ranks{size}.csv", "a")  # append mode
file1.write(f"{rank},{end_total - start_total} \n")
file1.close()

