import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

textfile = "/Users/jamesledoux/Desktop/tolkien.txt"
textfile = open(textfile).read().lower()
badtext = '/:;][{}-=+)(*&^%-_$#@!~1234567890'
textfile = textfile.translate(None, badtext)

def translation_dicts(textfile):
    chars = sorted(list(set(textfile)))
    char_to_num_dict = {}
    num_to_char_dict = {}
    num = 0
    for c in chars:
        char_to_num_dict[c] = num
        num_to_char_dict[num] = c
        num += 1
    return num_to_char_dict, char_to_num_dict, chars

num_to_char_dict, char_to_num_dict, chars = translation_dicts(textfile)

sequence_length = 40 #num of chars per observation
beg_index = 0
end_index = sequence_length
X_mat = []
y_vector = []

for i in range(len(textfile) - sequence_length):
    x_row = textfile[i:i+sequence_length]
    x_row = [char_to_num_dict[j] for j in x_row]
    y_val = textfile[i+sequence_length]  #the character you're predicting based on the previous 100
    y_val = char_to_num_dict[y_val]
    X_mat.append(x_row)
    #X_mat.append([char_to_num_dict[i] for i in x_row])
    y_vector.append(y_val)

#X_mat = np.asmatrix(X_mat)
y_vector = np.asarray(y_vector)

#convert y to one-hot encoding
y_mat = np_utils.to_categorical(y_vector)

num_patterns = len(X_mat)
num_vocab = len(chars)
X = np.reshape(X_mat, (num_patterns, sequence_length, 1))
X = X/float(num_vocab) #normalize X

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))  #WE NEED TO GO DEEPER!!!!
model.add(Dropout(0.2))
model.add(Dense(y_mat.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#fit the model
model.fit(X, y_mat, batch_size=64, nb_epoch=2, callbacks=callbacks_list)  #increase num epoch once we get this working well
