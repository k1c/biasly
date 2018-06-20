

# # Import Packages

# For Data Preparation
import tensorflow as tf
import numpy as np
import pandas as pd
import re # regular expressions

# To clean up texts
import nltk.data
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
nltk.download('punkt')
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

# For Word Embedding
from collections import Counter
import gensim
import gensim.models as g
from gensim.models import Word2Vec
from gensim.models import Phrases

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt

# For the Model
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import LSTM, Bidirectional,Dropout, Input, SpatialDropout1D, CuDNNLSTM, Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from mlxtend.plotting import plot_learning_curves
import matplotlib.pyplot as plt
from mlxtend.data import iris_data
from mlxtend.preprocessing import shuffle_arrays_unison

import logging

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from keras.models import model_from_json






# # Data Preparation

# Data Processing

# Load the data from a CSV (returns type(dataframe))
bias_data = pd.read_csv('./scrapeBIG2.csv', sep=',',  encoding='latin-1') # encoding='latin-1', names=cols, header=None,
print("Data loaded.")

# Randomize data
bias_data = bias_data.reindex(np.random.permutation(bias_data.index))

# Replace NAN with Zero(0)
bias_data.fillna(0, inplace=True)

print("Size of the data: ", len(bias_data))


sentence = bias_data['Sentence']
label = bias_data['Bias']


# Convert a sentence into a list of words

def sentence_to_wordlist(sentence, remove_stopwords=False):
    # 1. Remove non-letters
    sentence_text = re.sub(r'[^\w\s]','', sentence)
    # 2. Remove all numbers
    sentence_text = re.sub(r'[0-9]+', '', sentence_text)
    # 3. Convert words to lower case and split them
    words = sentence_text.lower().split()
    # 4. Stemming
    words = [stemmer.stem(w) for w in words] 
    # 5. Lemmatizing
    words = [lemmatizer.lemmatize(word) for word in words]
    # 6. Return a list of words
    return(words)


# whole data into a list of sentences where each sentence is a list of word items
def list_of_sentences(data):
    sentences = []
    for i in data:
        sentences.append(sentence_to_wordlist(i))
    return sentences



sentences = list_of_sentences(bias_data['Sentence'])
labels = bias_data['Bias'].tolist()


# # Word Embedding

# Create Word Vectors

wv_model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=16, sg=0, negative=5)
word_vectors = wv_model.wv
words = list(wv_model.wv.vocab)

# Calling init_sims will make the model will be better for memory
# if we don't want to train the model over and over again
wv_model.init_sims(replace=True)

# save model
wv_model.save('model.bin')


# Build dictionary & inv_vocab

def create_vocab(data_collect, max_vocab):
    # Get raw data
    x_list = data_collect
    sample_count = sum([len(x) for x in x_list])
    words = []
    for data in x_list:
        words.extend([data])
    count = Counter(words) # word count
    inv_vocab = [x[0] for x in count.most_common(max_vocab)]
    vocab = {x: i for i, x in enumerate(inv_vocab, 1)}
    return vocab, inv_vocab

vocab, inv_vocab = create_vocab(words, len(words))


# Find the max length sentence
def find_max_length_sentence(sentence):
    max_length = 0
    for i in sentence:
        length = len(sentence_to_wordlist(i))
        if max_length < length:
            max_length = length
    return max_length


seq_length = find_max_length_sentence(sentence)


# Map each word to corresponding vector
def map_to_vec(word):
    vec = wv_model[word]
    return vec


# Embedding Matrix
def make_emb_matrix(inv_vocab):
    emb_matrix = []
    for word in inv_vocab:
        emb_matrix.append(map_to_vec(word))
    return emb_matrix


embedding = np.asarray(make_emb_matrix(inv_vocab))


# Creating the training and validation sets
X_train, X_test, Y_train, Y_test = train_test_split(sentences, labels,
                                                    test_size = 0.2,
                                                    random_state = 2,
                                                    shuffle= True)



# # Initialize Word Embeddings in Keras

wv_dim = 100
num_words = len(word_vectors.vocab)
vocab = Counter(words)


word_index = {t[0]: i+1 for i,t in enumerate(vocab.most_common(num_words-1))}

train_sequences = [[word_index.get(t, 0) for t in sentence]
             for sentence in X_train[:len(X_train)]]

test_sequences = [[word_index.get(t, 0)
                   for t in sentence] for sentence in X_test[:len(X_test)]]

# Pad zeros to match the size of matrix
train_data = pad_sequences(train_sequences, maxlen=seq_length, padding="post", truncating="post")
test_data = pad_sequences(test_sequences, maxlen=seq_length, padding="post", truncating="post")


# Initialize the matrix with random numbers
wv_matrix = (np.random.rand(num_words, wv_dim) - 0.5) / 5.0
for word, i in word_index.items():
    if i >= num_words:
        continue
    try:
        embedding_vector = word_vectors[word]
        # words not found in embedding index will be all-zeros.
        wv_matrix[i] = embedding_vector
    except:
        pass






# # LSTM Model

# Embedding
wv_layer = Embedding(num_words,
                     wv_dim,
                     mask_zero=False,
                     weights=[wv_matrix],
                     input_length=seq_length,
                     trainable=False)

# Inputs
comment_input = Input(shape=(seq_length,), dtype='int64')
embedded_sequences = wv_layer(comment_input)

# LSTM
embedded_sequences = SpatialDropout1D(0.2)(embedded_sequences)
x = Bidirectional(LSTM(64, return_sequences=False))(embedded_sequences)

# Output
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
preds = Dense(1, activation='sigmoid')(x)

# build the model
model = Model(inputs=[comment_input], outputs=preds)
model.compile(loss='binary_crossentropy',   #binary_crossentropy
              optimizer=Adam(lr=0.001, clipnorm=.25, beta_1=0.7, beta_2=0.99),
              metrics=['accuracy'])


hist = model.fit(train_data, Y_train, validation_data=(test_data, Y_test), epochs=15, batch_size=32)

# Final evaluation of the model
scores = model.evaluate(test_data, Y_test, verbose=2, batch_size =32)
#print("LSTM Score: %.2f%%" % (scores*100)) # Evaluation of the loss function for a given input



# # Save and Load the Model (_.h5 .json)

# Save the weights
model.save_weights('biasly_model_weights.h5')

# Save the model architecture
with open('biasly_model_architecture.json', 'w') as f:
    f.write(model.to_json())


# Model reconstruction from JSON file
with open('biasly_model_architecture.json', 'r') as f:
    loaded_model = model_from_json(f.read())

# Load weights into the new model
loaded_model.load_weights('biasly_model_weights.h5')




# # Predict Sentences using LSTM
def predict_sentences():
    #print("Please input a sentence: ")
    pred_text = sentence_to_wordlist(input())
    pred_sequences = [word_index.get(t, 0) for t in pred_text]
    pred_data = pad_sequences([pred_sequences], maxlen=seq_length, padding="post", truncating="post")
    prediction = loaded_model.predict(pred_data)
    #return "LSTM Model thinks the sentence is %.2f%% biased."%(prediction[0][0] * 100)
    return "%.2f%%"%(prediction[0][0] *100)

print(predict_sentences())

