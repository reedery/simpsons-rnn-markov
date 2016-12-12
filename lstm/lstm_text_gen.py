"""
@Author: Drew Hoo, James Ledoux, Emily Lu, Ryan Reede, Charlie Fox, Lameron

- Preprocess
- Train 1 epoch
- Save weights from epoch
- generate output from the weights of that epoch
- save file
- take the saved weights from prev epoch and load them back in
- train another epoch
- repeat for some number of epochs

Based off of fchollet's lstm text gen script found here:
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

This code is dedicated to all the [haters] out there who told me I'd never amount
to nothin', to all the people who told me tf wouldnt be possible, becasue
it wasn't, and so, you get what we have here today (12/11/16), which is the way I want it.
Well, I got it. I don't like keras any more than you.
 - Cool Hand Biggie 
"""
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
import numpy as np
import random
import sys
import traceback
import pdb
from os import path
from datetime import datetime
import json
import re


def lstm(file_path, epochs):
    # Get text file from path
    print file_path
    text = open(file_path).read().lower()
    len_text = len(text)
    print 'Text info: len {}, type {}'.format(len_text, type(text))

    # Get Unique chars from text
    chars = sorted(list(set(text)))
    len_chars = len(chars)
    print 'Total Unique Chars: ', len_chars

    # Set up translation dicts
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # fchollet = 40 => Ours = ~100
    # rows are of size maxlen
    maxlen = 80
    # step through text file
    step = 3
    sentences = []
    next_chars = []

    for i in range(0, len(text) - maxlen, step):
        end_index = i + maxlen
        sentences.append(text[i: end_index])
        next_chars.append(text[end_index])
    print 'Total number sequences: ', len(sentences)

    # Start making your sparse matrices
    print 'Vectorizing...'
    X = np.zeros((len(sentences), maxlen, len_chars), dtype=np.bool)
    y = np.zeros((len(sentences), len_chars), dtype=np.bool)

    # Check space complexity
    space_X = sys.getsizeof(X)
    space_y = sys.getsizeof(y)
    print 'Space X: {} Bytes'.format(space_X)
    print 'Space_y: {} Bytes'.format(space_y)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    # Buid and run Model
    print 'Building and Running Model ...'
    run(X, y, len_chars, epochs, len_text, maxlen, char_indices, indices_char, text)

def run(X, y, len_chars, num_epochs, len_text, maxlen,char_indices, indices_char, text):
    X_shape_1, X_shape_2 = X.shape[1], X.shape[2]
    y_shape = y.shape[1]
    full_start_time = datetime.now()
    for epoch_counter in range(num_epochs):
        print '\n Starting Epoch {} ... '.format(epoch_counter)
        model = build_model(X_shape_1, X_shape_2, y_shape, epoch_counter - 1)
        # Fit for for 1 epoch only
        start_time = datetime.now()
        # Commented out callbacks for now
        history = model.fit(X, y, validation_split=0.20, nb_epoch=1, batch_size=512, verbose=1)
        model_total_time = datetime.now() - start_time
        print "training time: " + str(model_total_time)
        save_history(history)
    
        # Save the weights from the training
        print '\n Saving weights ...'
        model_weights = 'char_training/model_weights_' + str(epoch_counter) + '.h5'
        model.save_weights(model_weights)
        print 'Weights Saved'
        gen_output(model, len_text, maxlen, len_chars, char_indices, indices_char, text, epoch_counter)
        print '\n Finished output of Epoch: {}'.format(epoch_counter)

    total_time = datetime.now() - full_start_time
    print "Semi-total Run Time: " + str(model_total_time)


def build_model(X_shape_1, X_shape_2, y_shape, prev_epoch_counter):
    # define the LSTM model via our old code - No callbacks for now
    model = Sequential()
    model.add(LSTM(512, input_shape=(X_shape_1, X_shape_2), return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(512))
    model.add(Dropout(0.4))
    model.add(Dense(y_shape, activation='softmax'))
    
    # Check if weights file exists, should not exist only for first run
    file_path = 'char_training/model_weights_' + str(prev_epoch_counter) + '.h5'
    print 'File path of model weights: ', file_path
    if path.isfile(file_path):
        print 'found file, loading weights... '
        model.load_weights(file_path)
    else:
	print '.h5 file not found'
    # Compile and return model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def gen_output(model, len_text, maxlen, len_chars, char_indices, indices_char, text, epoch):
    stdout = sys.stdout
    output_path = 'char_lstm_output_files/lstm_output_text_{:02d}.txt'.format(epoch)
    sys.stdout = open(output_path, 'w')
    start_index = random.randint(0, len_text - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print '\n Output for Epoch {:02d} with diversity {}'.format(epoch, diversity)
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print 'Seed: "' + sentence + '" \n'
        sys.stdout.write(generated)

        for i in range(1000):
            x = np.zeros((1, maxlen, len_chars))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
    sys.stdout = stdout

def save_history(history):
    print "Saving History"
    with open('char_training/character-training_history.json', 'w') as f:
        json.dump(history.history, f)
    print 'History Saved'

""" 
The sample() method below is by fchollet
This is a helper function to sample an index from a probability array
Adjusting the normalizing factor of the predictions is something we never thought
of doing, but it has a huge effect on the generated output. 
Log(preds) -> normalize -> exp(preds) -> normalize again -> Multinomial dist where pval = preds
Never would have thought of this.
"""
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

if __name__ == '__main__':
    try:
        files = ["../2000_lines.txt","../400kb_script.txt", "../800kb_script.txt", "../eight_scripts.txt", "../quarter_scripts.txt", "../all_scripts"]
        epochs = 80
        lstm(files[2], epochs)
    except Exception as e:
        #raise e
	   traceback.print_exc()
