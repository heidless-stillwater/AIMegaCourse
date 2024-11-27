import os
import pandas as pd
import numpy as np

import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

from icecream import ic

# configure logging
import logging
    
path_name = os.path.basename(__file__)
# print(f"path_name: {path_name}")

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
    
    
def regression_test():
    logger.info('Starting Regression Test') 
    
    # logger.info('loading data')
    data = pd.read_csv("data/student_mat_2173a47420.csv", sep=";")
    # print(data.head())
 
    data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

    predict = "G3"

    # attributes
    X = np.array(data.drop([predict], axis=1))
    # log_this(X, "X")    
    
    # labels
    y = np.array(data[predict])
    # ic(y)

    f_name = "studentmodel.pickle"
    m_dir = "saved_models"
    save_file = f"{m_dir}/{f_name}"
    
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    # best = 0
    # for _ in range(10000):
    #   x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    #   linear = linear_model.LinearRegression()
      
    #   linear.fit(x_train, y_train)
    #   acc = linear.score(x_test, y_test)
    #   logger.info(f"acc: {acc} | best: {best}")
      
    #   if acc > best:
    #     best = acc 
    #     with open(save_file, "wb") as f:
    #         pickle.dump(linear, f)        
      
    # load
    pickle_in = open(save_file, "rb")
    linear = pickle.load(pickle_in)
    
    acc = linear.score(x_test, y_test)
    logger.info(f"acc: {acc}")

    # log_this(linear.coef_, "Co: ")
    # log_this(linear.intercept_, "Intercept: ")

    predictions = linear.predict(x_test)
    log_this(predictions, "\npredictions")
    
    # for x in range(len(predictions)):
    #   logger.info(f"predictions: {predictions[x]} | x_test[x]: {x_test[x]} | y_test[x]: {y_test[x]}") 

    p = "G1"
    style.use("ggplot")
    pyplot.scatter(data[p], data["G3"])
    pyplot.xlabel(p)
    pyplot.ylabel("final grade")
    pyplot.show()
  


def main():
    logger.info('--------') 
    # logger.info('Calling Regression Test')
    regression_test()
    
if __name__ == '__main__':
    main()

