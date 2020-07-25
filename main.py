import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import random
import tensorflow as tf
import json
import pickle

with open("basic.json") as file:
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
        if w in wrds:
            bag.append(1)

        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)


tf.reset_default_graph() #reset any previous data stored

#this tf and tflearn code is a little complicated, refer tutorials for future refference
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

# working with input from the user
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
             if w == se:
                 bag[i] = 1

    return numpy.array(bag)

def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input('You: ')
        if inp.lower() == 'quit':
            break

        results = model.predict([bag_of_words(inp, words)])[0] #getting the user input and predicts the tag which has the highest probability
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses)) #choice() -returns a random item from the list
        else:
            print("I didn't get that, try again")

chat()
