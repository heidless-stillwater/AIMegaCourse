import tensorflow as td
from tensorflow import keras
import streamlit as st
import numpy as np

from icecream import ic
import logging

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
# generale model
#

data = keras.datasets.imdb

vocab_size = 88000
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=vocab_size)

# st.write(train_data[0])

word_index = data.get_word_index()

word_index = { k:(v+3) for k, v in word_index.items() }

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# pad to make all entries the same size
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

# #######
# # model
# #

# def decode_review(text):
#   return " ".join([reverse_word_index.get(i, "?") for i in text])

# # st.write(decode_review(test_data[0]))

# model = keras.Sequential()
# model.add(keras.layers.Embedding(vocab_size, 16))
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(16, activation="relu"))
# model.add(keras.layers.Dense(1, activation="sigmoid"))

# model.summary()

# # streamlit version of model.summary()
# # model.summary(print_fn=lambda x: st.write(x))
# model.summary()

# ##############
# # traing model
# #
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# # take first 10000 of 25000 data as validation data set
# x_val = train_data[:10000]    
# x_train = train_data[10000:]

# y_val = train_labels[:10000]    
# y_train = train_labels[10000:]

# fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# results = model.evaluate(test_data, test_labels)
# # st.write(results)

# acc_score = results[1]
# # st.write("acc_score: ", acc_score)

# loss = results[0]
# # st.write("loss: ", loss)

# model.save("saved_models/model.h5")


############
# load model
#
st.write("loading model")
model = keras.models.load_model("saved_models/model.h5")


# ############
# # prediction
# #
# st.subheader("Review")

# test_review = test_data[0]

# # predict = model.predict([test_review])
# predict = model.predict(np.expand_dims(test_review, 0))
# # st.write(predict)
# # st.write(decode_review(test_review))
# st.write("Prediction: " + str(predict[0]))
# st.write("Actual: " + str(test_labels[0]))
# st.write(results)

def review_encode(s):
	encoded = [1]

	for word in s:
		if word.lower() in word_index:
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2)

	return encoded


with open("test.txt", encoding="utf-8") as f:
  st.write("file opened")
  for line in f.readlines():
    # st.write("line: ", line)
    nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
    encode = review_encode(nline)
    encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
    predict = model.predict(encode)
    # predict = model.predict(np.expand_dims(encode, 0))
    st.write(line)
    st.write(encode)
    st.write(predict[0])
    st.write(predict)

