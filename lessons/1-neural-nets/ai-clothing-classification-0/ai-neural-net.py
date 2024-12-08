# import os
# import pandas as pd
# import numpy as np
# import streamlit as st 

# import sklearn
# from sklearn import datasets
# from sklearn import metrics
# from sklearn import svm
# from sklearn.utils import shuffle
# from sklearn.neighbors import KNeighborsClassifier 
# from sklearn import linear_model, preprocessing

# from sklearn.preprocessing import scale
# from sklearn.datasets  import load_digits
# from sklearn.cluster import KMeans

# import matplotlib.pyplot as pyplot
# import pickle
# from matplotlib import style

import os
import tensorflow as tf   
import streamlit as st

from tensorflow import keras
import numpy as np   
import matplotlib.pyplot as plt

import plotly.express as px   

from icecream import ic

# configure logging
import logging
    
path_name = os.path.basename(__file__)
print(f"path_name: {path_name}")

file_root = os.path.splitext(path_name)[0]
print(f"file_root: {file_root}")

# configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f"./logs/{file_root}.log",
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
    
def log_this(arr, msg):
    # logger.info(f"in {log_this.__name__}")
    # arr = np.arange(0,20)
    # msg = "TEST MSG"
    logger.info(f"{msg}: {arr}")
    # logger.info(f"arr.shape: {arr.shape}")  

def ai_neural_net():
    st.write("in ai_neural_net")

    data = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = data.load_data()

    # st.image(train_images)
    # st.write(train_labels[0])
    # st.write("train_image")
    # st.write(train_images[0])

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    st.write(class_names)

    st.dataframe(class_names)

    img0 = st.image(train_images[0])
    img1 = st.image(train_images[1])

    train_images = train_images/255.0
    test_images = test_images/255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(train_images, train_labels, epochs=1)

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    st.write("model accuracy: ", test_acc)

    prediction = model.predict(test_images)


    arr = np.random.normal(1, 1, size=100)
    fig, ax = plt.subplots()
    # ax.hist(arr, bins=20)
    ax.imshow(test_images[1], cmap=plt.cm.binary)


    for i in range(10):
      st.image(test_images[i])
      st.write(class_names[np.argmax(prediction[i])])

    #   plt.grid(False)
    #   plt.imshow(test_images[i], cmap=plt.cm.binary)
    #   # plt.xlabel("Actual: " + test_labels[i])
    #   plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    # #   plt.show()
    #   st.pyplot(fig)



      # st.write(test_images[i])

      # img = st.image(test_images[i])
      # st.write(class_names[np.argmax(prediction[i])])

    # prediction


def main():
    logger.info('neural net') 
    logger.info('----------')
    st.title('ai neural net') 
    ai_neural_net()

    

if __name__ == '__main__':
    main()

