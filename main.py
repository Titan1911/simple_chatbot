import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer

import numpy
import tflearn
import random
import tensorflow
import json

with open("intend.json") as file:
    data = json.load(file)

print(data)