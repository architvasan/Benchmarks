# Setup

import argparse
import os
import numpy as np
import matplotlib
import pandas as pd

matplotlib.use("Agg")

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
import horovod.keras as hvd ### importing horovod to use data parallelization in another step
#HVD-1 initialize horovid 
hvd.init() 
print("I am rank %d of %d" %(hvd.rank(), hvd.size()))
#parallel_threads = parallel_threads//hvd.size()
#os.environ['OMP_NUM_THREADS'] = str(parallel_threads)
#num_parallel_readers = parallel_threads

#HVD-2: GPU pinning
gpus = tf.config.experimental.list_physical_devices('GPU')
# Ping GPU to each9 rank
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu,True)
if gpus:
	tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


# File loading
file_path = os.path.dirname(os.path.realpath(__file__))

# psr and args take input outside of the script and assign:
# (1) file paths for data_path_train and data_path_vali
# (2) number of training epochs

psr = argparse.ArgumentParser(description="input csv file")
psr.add_argument("--in_train", default="in_train")
psr.add_argument("--in_vali", default="in_vali")
psr.add_argument("--ep", type=int, default=400)
args = vars(psr.parse_args()) # returns dictionary mapping of an object
print(args)

EPOCH = args["ep"]
BATCH = 32 # batch size used for training

data_path_train = args["in_train"]
data_path_vali = args["in_vali"]

DR = 0.1  # Dropout rate

# define r2 for reporting

def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


# Implement a Transformer block as a layer
# embed_dim: number of tokens. This is used for the key_dim for the multi-head attention calculation
# ff_dim: number of nodes in Dense layer
# epsilon: needed for numerical stability... not sure what this means to be honest

class TransformerBlock(layers.Layer):
    # __init__: defining all class variables
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    # call: building simple transformer architecture
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# Implement embedding layer
# Two seperate embedding layers, one for tokens, one for token index (positions).


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        print(x.shape)
        return x + positions
        

# Input and prepare dataset

vocab_size = 40000  #number of possible 'words' in SMILES data
maxlen = 250  #length of each SMILE sequence in input


data_train = pd.read_csv(data_path_train)
data_vali = pd.read_csv(data_path_vali)

data_train.head()

# Dataset has type and smiles as the two fields
# reshaping is done so that y is formatted as [[y_1],[y_2],...] with floats
y_train = data_train["type"].values.reshape(-1, 1) * 1.0 
y_val = data_vali["type"].values.reshape(-1, 1) * 1.0

tokenizer = text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(data_train["smiles"])


def prep_text(texts, tokenizer, max_sequence_length):
    # Turns text into into padded sequences.
    text_sequences = tokenizer.texts_to_sequences(texts) # turns text into tokens
    return sequence.pad_sequences(text_sequences, maxlen=maxlen) # pad all sequences so they all have same length


x_train = prep_text(data_train["smiles"], tokenizer, maxlen)
x_train = np.array_split(x_train,hvd.size())
y_train = np.array_split(y_train,hvd.size())
print(x_train, y_train)
#print(x_train.shape)
#print(y_train.shape)
#x_train = tf.data.Dataset.from_tensor_slices(x_train)
#y_train = tf.data.Dataset.from_tensor_slices(y_train)
#x_train = np.array_split(x_train, hvd.rank())
#print(f'number of shards is {hvd.size}')
#x_train = x_train.shard(num_shards=hvd.size(), index=hvd.rank())
#y_train = y_train.shard(num_shards=hvd.size(), index=hvd.rank())
#print(x_train)
#train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#train_dataset = train_dataset.shard(num_shards=hvd.size(), index=hvd.rank())


x_val = prep_text(data_vali["smiles"], tokenizer, maxlen)

# Create regression/classifier model using N transformer layers

embed_dim = 128  # Embedding size for each token
num_heads = 16  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = transformer_block(x)
x = transformer_block(x)
x = transformer_block(x)

# x = layers.GlobalAveragePooling1D()(x)  --- the original model used this but the accuracy was much lower

x = layers.Reshape((1, 32000), input_shape=(250, 128,))(
    x
)  # reshaping increases parameters but improves accuracy a lot
x = layers.Dropout(0.1)(x)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(16, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="relu")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

# Train and Evaluate

#HVD scale learning rate by number of workers
lrate = 0.00001 * hvd.size()
opt = Adam(learning_rate=lrate)

#HVD Wrap optimizer in hvd Distributed Optimizer delegates gradient comp to original optimizer, averages gradients, and applies averaged gradients
opt = hvd.DistributedOptimizer(opt)
print(f"opt is {opt}")
model.compile(
    loss="mean_squared_error", optimizer=opt, metrics=["mae", r2]
)

# set up a bunch of callbacks to do work during model training..

#HVD broadcast initial variables from rank0 to all other processes 

hvd_broadcast = hvd.callbacks.BroadcastGlobalVariablesCallback(0)

# learning rate tuning at each epoch
# is it possible to do batch size tuning at each epoch as well? 
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.75,
    patience=20,
    verbose=1,
    mode="auto",
    epsilon=0.0001,
    cooldown=3,
    min_lr=0.000000001,
)

checkpointer = ModelCheckpoint(
    filepath="smile_regress.autosave.model.h5",
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

callbacks = [hvd_broadcast, reduce_lr]

csv_logger = CSVLogger("smile_regress.training.log")

early_stop = EarlyStopping(monitor="val_loss", patience=100, verbose=1, mode="auto")

if hvd.rank() == 0:
    callbacks.append(csv_logger)
    callbacks.append(early_stop)

x_train_feed = x_train[hvd.rank()]
y_train_feed = y_train[hvd.rank()]

history = model.fit(
    x_train_feed,
    y_train_feed,
    batch_size=BATCH,
    epochs=EPOCH,
    verbose=1,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
)

#model.load_weights("smile_regress.autosave.model.h5")
