import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import random
import tensorflow
import json

with open("intent.json") as file:
    data = json.load(file)

# print(data['intents'][0]['patterns'])

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)   #tokenize- getting the words written by splitting every word from the sentence or phrase. directly done by nltk, adds to a list
        words.extend(wrds) #add up all the words in the 'words' list
        docs_x.append(wrds)
        docs_y.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# stemming-will take a word and bring it down to the root word. Eg. whats up will be brought down to the root word what
words = [stemmer.stem(w.lower()) for w in words if w not in '?']
words = sorted(list(set(words))) #purpose of set()- to remove any duplicates

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]
    for w in words:
        if w in words:
            bag.append(1)

        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)