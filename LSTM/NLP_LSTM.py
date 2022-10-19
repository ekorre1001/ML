# Implementation of siamese LSTM
# inspiration and some implementation taken from https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb
# Code copied from a juputer notebook
import numpy as np
import pandas as pd

from keras.layers.merge import concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping
import tensorflow as tf
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# define some variables
max_length = 30
EMBEDDING_DIM = 300



# load w2v
print('Indexing word vectors')
# must change to binary False if using GLove
word2vec = KeyedVectors.load_word2vec_format('/content/drive/MyDrive/Bert-model/GoogleNews-vectors-negative300.bin.gz', binary=True)

# function to Clean the text
def text_to_wordlist(text, remove_stopwords=False):

    text = text.lower().split()

    # try to remove the stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text, credit to the https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb for text cleaning
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
  
    return(text)

# read the csv file
df = pd.read_csv('/content/questions.csv')
df.dropna(inplace=True)
# clean the questions
df["question1"]=df.apply(lambda x: text_to_wordlist(x["question1"]), axis=1)
df["question2"]=df.apply(lambda x: text_to_wordlist(x["question2"]), axis=1)

## check tutorial https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
sent_1 = df.question1.values 
sent_2 = df.question2.values
labels = df.is_duplicate.values

# define some test and train
k = len(sent_1)
test_1 = sent_1[int(0.7*k):]
test_2 = sent_2[int(0.7*k):]
test_labels = labels[int(0.7*k):]
train_1 = sent_1[0:int(0.7*k)]
train_2 = sent_1[0:int(0.7*k)]
labels= labels[0:int(0.7*k)]

# begging tokenizing and padding
t = Tokenizer()
t.fit_on_texts(sent_1 + sent_2)

# texts_1: what is the step by step guide to invest in share market in india
# sequences_1: [2, 3, 1, 1215, 57, 1215, 2572, 7, 585, 8, 771, 386, 8, 36]

seq_1 = t.texts_to_sequences(train_1)
seq_2 = t.texts_to_sequences(train_2)
tseq_1 = t.texts_to_sequences(test_1)
tseq_2 = t.texts_to_sequences(test_2)

# map word to integer
word_index = t.word_index
# padded with 0
paddedTrain_1 = pad_sequences(seq_1, maxlen=max_length)
paddedTrain_2 = pad_sequences(seq_2, maxlen=max_length)
labels = np.array(labels)
paddedTest_1 = pad_sequences(tseq_1, maxlen=max_length)
paddedTest_2 = pad_sequences(tseq_2, maxlen=max_length)
test_labels = np.array(test_labels)

# embedding matrix
vocab_size = len(word_index)+1
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
       # Words not found in embedding index will be all-zeros.
        embedding_matrix[i] = word2vec.word_vec(word)

# split into train  and val
from sklearn.model_selection import train_test_split
X1_train, X1_val = train_test_split(paddedTrain_1, test_size=0.3, random_state=14)
X2_train, X2_val = train_test_split(paddedTrain_2, test_size=0.3, random_state=14)
y_train, y_val = train_test_split(labels, test_size=0.3, random_state=14)

# define the layers

embedding_layer = Embedding(vocab_size,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=max_length,
        trainable=False)
# trainable = false because we dont want to update the word embeddings
lstm_layer = LSTM(units=128, dropout=0.3, recurrent_dropout=0.2)
sequence_1_input = Input(shape=(max_length,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
# output = lstm(inputs)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(max_length,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
# output = lstm(inputs)
y1 = lstm_layer(embedded_sequences_2)

concat_layer = concatenate([x1, y1])
# keras layer chain (layer1)(layer2)
add_drop = Dropout(0.2)(concat_layer)


dense_layer = Dense(units=64, activation="relu")(add_drop)
dense_drop = Dropout(0.2)(dense_layer)


sigmoid_layer = Dense(1, activation='sigmoid')(dense_drop)

# define model

model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=sigmoid_layer)
model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['acc'])
m = model.fit([X1_train, X2_train], y_train, 
        validation_data=([X1_val, X2_val], y_val), 
        epochs=1, batch_size=64, shuffle=True)

result = model.evaluate([paddedTest_1, paddedTest_2], test_labels, batch_size=64)
print(result)