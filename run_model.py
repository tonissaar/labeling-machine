import os
import numpy as np
import pandas as pd
import tensorflow as tf
import io
import json
import glob
import argparse
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from  tensorflow.keras.preprocessing.text import tokenizer_from_json


parser = argparse.ArgumentParser(description='Run inference on a test set.')
parser.add_argument('--path', help='path to test set folder', default='ml_interview_ads_data')
args = parser.parse_args()


with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

model = tf.keras.models.load_model("model.h5")
labels = pd.read_json('labels.json')
labels[0] = labels['0']

data_list = []
filelist = glob.glob(args.path + '/*.csv')
for file in filelist:
    tmp = pd.read_csv(file, index_col=None, header=None)
    tmp['file'] = os.path.basename(file)
    data_list.append(tmp)

data = pd.concat(data_list, axis=0, ignore_index=True)

data = pd.merge(data[[0, 1]], labels[[0, 'label_enc']], on = 0)

sentences_train = data[1].to_numpy()
labels_train = data['label_enc'].to_numpy()

X_train = tokenizer.texts_to_sequences(sentences_train)
maxlen = 100
X_train_padded = pad_sequences(X_train, padding='post', maxlen=maxlen)
y_train = to_categorical(labels_train)

loss, accuracy = model.evaluate(X_train_padded, y_train, verbose=True)
print("Test set Accuracy: {:.4f}".format(accuracy))