import random
import time
import csv
import json


class TextGenerator:

    def __init__(self):
        self.model = {}
        self.words = []
        self.order = 0

    def read_csv(self, file_name):
        with open(file_name) as f:
            reader = csv.DictReader(f)
            for row in reader:
                words = row['raw_text'].split()
                self.words.extend(words)

    def read_txt(self, file_name):
        with open(file_name, 'r') as f:
            full_text = ' '.join(line.rstrip() for line in f)
            self.words = full_text.split(' ')

    def load_model(self, file_name):
        with open(file_name, 'r') as f:
            self.model = json.load(f)

    def save_model(self, file_name):
        with open(file_name, 'w') as out:
            json.dump(self.model, out, sort_keys=True, indent=4)

    # create markov model of specified order
    def train_model(self):
        words = self.words
        # print words
        for i in range(len(words)-1):

            if (i+1) % 10000 == 0:
                print str(i+1) + ' words trained'

            # if the word is not in the main dictionary, add it and add
            # the following word in its value dictionary with a frequency of 1
            if words[i] not in self.model.keys():
                self.model[words[i]] = {words[i+1]: 1}

            # else if the word is in the main dictionary, check to see if the next
            # word is in it's value dictionary and create or update the frequency accordingly
            else:
                if words[i+1] not in self.model[words[i]].keys():
                    self.model[words[i]][words[i+1]] = 1
                else:
                    self.model[words[i]][words[i+1]] += 1
        if '' in self.model:
            del self.model['']


    # generate text of specified length based on the model
    def generate_text(self, seed, length):
        last_word = seed
        output = last_word
        while len(output) < length:
            next_words = self.model.get(last_word)
            choices = next_words.keys()
            frequencies = [i/float(sum(next_words.values())) for i in next_words.values()]
            # print last_k, choices, frequencies
            choice = random_pick(choices, frequencies)
            output += ' ' + choice
            last_word = choice
        return output


# make sure this works properly
def random_pick(char_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(char_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability: break
    return item


if __name__ == '__main__':
    t = TextGenerator()
    t.load_model('simpsons.json')
    print t.generate_text('Bart', 1000)
