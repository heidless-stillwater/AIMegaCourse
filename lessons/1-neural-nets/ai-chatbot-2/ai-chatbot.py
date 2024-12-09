import nltk 
# nltk.download('punkt_tab')

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# import tensorflow as tf    

import numpy as np  

import tflearn   

from icecream import ic
import random
import logging
import json

print('test')

logger = logging.getLogger()

file_root = "aimovieclassification"
logfilename = f"./logs/{file_root}.log"

print(f"file_root: {file_root}")
print(f"logfilename: {logfilename}")

fhandler = logging.FileHandler(filename=logfilename, mode='a')

formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)

logger.setLevel(logging.DEBUG)

logging.debug("logging:configured")

# st.write("aimovieclassification:started")

################
# load 'intents'
#
with open("data/intents.json") as file:
  data = json.load(file)

# print(data['intents'])

#####################
# prep data for model
#
words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
  for pattern in intent["patterns"]:
    wrds = nltk.word_tokenize(pattern)
    words.extend(wrds)
    docs_x.append(wrds)
    docs_y.append(intent["tag"])

  if intent["tag"] not in labels:
    labels.append(intent["tag"])

print(f"\nwords: {words}\n")

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

print(f"\nwords: {words}\n")

# print("labels: ", labels)
# # print("docs_x: ", docs_x)
# # print("docs_y: ", docs_y)

##################################
# one-hot encoding

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

  output_row = out_empty[:]   # copy full array
  output_row[labels.index(docs_y[x])] = 1

  training.append(bag)

training = np.array(training)
output = np.array(output)

############
# buid model
#
tf.reset_default_graph()

net = tflearn.input_data(shape=[none, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)   

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)

model.save("saved_models/model.tflearn")
