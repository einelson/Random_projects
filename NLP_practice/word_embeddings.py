from matplotlib.cbook import flatten
import numpy as np
from sklearn import metrics
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import keras

# load in dataset
data, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)


# split into train and testing dataset
train_data, test_data = data['train'], data['test']

# containers for sequences and labels
train_sentences = []
test_sentences = []

train_labels = []
test_labels = []

# preprocess the data and put it into correct containers
for sent, label in train_data:
    train_sentences.append(str(sent.numpy().decode('utf8')))
    train_labels.append(label.numpy())

for sent, label in test_data:
    test_sentences.append(str(sent.numpy().decode('utf8')))
    test_labels.append(label.numpy())

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)


# set up parameters
vocab_size = 10000
embedding_dim = 16
max_len = 150
trunc_type = 'post'
oov_tok = '<oov>'


# set up tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index


# train data
train_seq = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_seq, maxlen=max_len, truncating=trunc_type)

# test data
test_seq = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_seq, maxlen=max_len)

# set up model
model = keras.models.Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    Flatten(),
    Dense(6, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(train_padded, train_labels, epochs=10, validation_data=(test_padded, test_labels))

# derive weights from embedding layer
# Isolate first layer
li = model.layers[0]

weights = li.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)
print(weights)