import os
import numpy as np
import pandas as pd
import tensorflow as tf
import io
import json
import plot_util
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
import glob

print('TF version')
print(tf.__version__)

GLOVE_PATH = 'glove/glove.6B.100d.txt'
DATA_PATH = 'ml_interview_ads_data/*.csv'

parser = argparse.ArgumentParser(description='Train model on a training set.')
parser.add_argument('--path', help='path to training data folder', default='ml_interview_ads_data')
args = parser.parse_args()


data_list = []
filelist = glob.glob(args.path + '/*.csv')
for file in filelist:
    tmp = pd.read_csv(file, index_col=None, header=None)
    tmp['file'] = os.path.basename(file)
    data_list.append(tmp)

data = pd.concat(data_list, axis=0, ignore_index=True)
data['label_enc'] = pd.factorize(data[0])[0]

labels = data[[0, 'label_enc']].groupby(['label_enc']).first().reset_index()
print('Labels: ')
print(labels)

unique_podcast_count = data['file'].unique().shape[0]
podcasts_train = data['file'].sample(n= int(unique_podcast_count * 0.7), random_state=1)

data_train = data[data['file'].isin(podcasts_train)]
data_test = data[~data['file'].isin(podcasts_train)]

minimum_class_count_train = data_train[[0, 1]].groupby([0]).count().min()[1]
data_train = data_train.groupby([0]).apply(pd.DataFrame.sample, n=minimum_class_count_train, replace=False).sample(frac=1)

minimum_class_count_test = data_test[[0, 1]].groupby([0]).count().min()[1]
#data_test = data_test.groupby([0]).apply(pd.DataFrame.sample, n=minimum_class_count_test, replace=False)

sentences_train = data_train[1].to_numpy()
labels_train = data_train['label_enc'].to_numpy()

sentences_test = data_test[1].to_numpy()
labels_test = data_test['label_enc'].to_numpy()

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

y_train = to_categorical(labels_train)
y_test = to_categorical(labels_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

maxlen = 100
X_train_padded = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test_padded = pad_sequences(X_test, padding='post', maxlen=maxlen)


embedding_dim = 100
embeddings_index = dict()
f = open(GLOVE_PATH)
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector



model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                            weights=[embedding_matrix],
                           output_dim=embedding_dim, 
                           input_length=maxlen,
                           trainable=False))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(4, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

model.summary()

history = model.fit(X_train_padded, y_train,
                    epochs=30,
                    verbose=True,
                    validation_data=(X_test_padded, y_test),
                    batch_size=32)


loss, accuracy = model.evaluate(X_train_padded, y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=True)
print("Testing Accuracy:  {:.4f}".format(accuracy))

model.save("model.h5")

tokenizer_json = tokenizer.to_json()
with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

print("Saved model to disk")

plot_util.plot_history(history)

