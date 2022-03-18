from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# sentences to tokenize
train_sentence = ['it is raining','it will not be sunny tomorrow', 'It is a sunny day']


# set up tokenizer
tokenizer = Tokenizer(oov_token='<OOV>')

# train tokenizer on sentences
tokenizer.fit_on_texts(train_sentence)

# store word index- get list of unique words
word_index = tokenizer.word_index
print(word_index )

# create sequence with tokenizer- puts sentences into a sequences corresponding with index
sequences = tokenizer.texts_to_sequences(train_sentence)
print(sequences)

# pad sequence- pads so all sequences are same length
padded_seq = pad_sequences(sequences, padding='post') #padding at the end
print(padded_seq)