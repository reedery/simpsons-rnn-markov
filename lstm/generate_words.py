import numpy as np
import sys
import re
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

"""
TODO 
add parentheses to regex so stage directions remain in the data
fix spaces before punctuation marks in output
"""
print "reading data"
textfile = "../gpu_sized.txt"
textfile = open(textfile).read().lower()
#badtext = '/;-*^%_@~:,.()!?' #characters to srip from the text
#textfile = [word.translate(None, badtext) for word in textfile]
textfile = re.findall(r"[\w']+|[.,:!?;\n]", textfile)

print "creating ID-word translation dictionaries"
def translation_dicts(textfile):
    words = sorted(list(set(textfile)))
    word_to_id = {}
    id_to_word = {}
    num = 0
    for w in words:
        word_to_id[w] = num
        id_to_word[num] = w
        num += 1
    return id_to_word, word_to_id, words

id_to_word, word_to_id, words = translation_dicts(textfile)
print str(len(words)) + " unique words in data set"
print str(len(textfile)) + " total words in data set"


sequence_length = 30 #num of words per observation. experiment with different values of this.
beg_index = 0
end_index = sequence_length
X_mat = []
y_vector = []

print "populating X and y data"
#x rows are sequences of [sequence_length] words in the order they appear in the script
for i in range(len(textfile) - sequence_length):
    x_row = textfile[i:i+sequence_length] #sequence of words in text form
    x_row = [word_to_id[j] for j in x_row] #convert from words to IDs
    y_val = textfile[i+sequence_length] #the word you're predicting based on the previous [sequence_length] words
    y_val = word_to_id[y_val] #encode as ID
    X_mat.append(x_row)
    y_vector.append(y_val)

y_vector = np.asarray(y_vector)

print "one-hot encoding X and y" #there's probably a faster way to do this step
#convert y to one-hot encoding
y_mat = np_utils.to_categorical(y_vector)

#reshape X, one-hot encode the IDs
#final shape (num sequences, num words per sequence, num possible words)
num_patterns = len(X_mat)
num_vocab = len(words)
X = np.reshape(X_mat, (num_patterns, sequence_length, 1)) #might be redundant. left over from prev version of the code
n_words = np.max(X) + 1 #plus 1 for the zero index
X = np.eye(n_words)[X]
X = np.reshape(X, (num_patterns, sequence_length, n_words))

print "building model"
# define the LSTM model
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add(Dense(y_mat.shape[1], activation='softmax'))

# load the network weights
filename = "weights-improvement-32-0.2376.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# pick a random seed
start = np.random.randint(0, len(X_mat) - 1)
pattern = X_mat[start]
# a random seed taken from the text
print "Seed:"
print "\"", ''.join([id_to_word[value] for value in pattern]), "\""

# trying out a custom seed
#pattern = "                                                             to be or not to be that is the question"
#print pattern
#pattern = [int_to_char[c] for c in pattern]

#generate characters
for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = np.eye(n_words)[x]
    x = np.reshape(x, (1, sequence_length, n_words))
    #x = x / float(num_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = id_to_word[index]
    seq_in = [id_to_word[value] for value in pattern]
    sys.stdout.write(result)
    #if result not in [",", ".", ":", "?", "!", "\n"]:
    sys.stdout.write(" ")
    pattern.append(index)
    pattern = pattern[1:len(pattern)]


