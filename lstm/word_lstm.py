import numpy as np
import re
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from datetime import datetime
import json 

"""
TODO 
add parentheses to regex so stage directions remain in the data
fix spaces before punctuation marks in output
"""
print "reading data"
textfile = "../tiny.txt"
#textfile = "../gpu_sized.txt"
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
#uncomment/update if you want to use old weights as a starting point for training
#filename = "/Users/jamesledoux/Desktop/weights-improvement-26-1.8959.hdf5"
#model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath="test-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


print "training model"
#fit the model
#note: using small values here to test locally. fix nb_epoch when running on gpu. 
#model.fit(X, y_mat, batch_size=256, nb_epoch=200, callbacks=callbacks_list)  #increase num epoch once we get this working well
start_time = datetime.now()
history = model.fit(X, y_mat, validation_split=0.20, nb_epoch=2, batch_size=64, verbose=1, callbacks=callbacks_list)
total_time = datetime.now() - start_time
print "training time: " + str(total_time)

print "saving history"
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)

print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




#