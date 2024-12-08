import os
import pandas as pd
import numpy as np
import streamlit as st 

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import linear_model, preprocessing

import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

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
  
def get_var_name(var):
    for name, value in locals().items():
        logger.info(f"var: {var}")
        st.write(f"var: {var}")
        if value is var:
            return name
          
def k_nearest_neighbour():
  logger.info('Starting k-nearest-neightbour') 
  st.title("k_nearest_neighbour")
  st.subheader("classification & regression algorithm ")

  st.write(""" 
    
  (K-NN) algorithm is a versatile and widely used machine learning algorithm that is primarily used for its simplicity and ease of implementation. 
  
  It does not require any assumptions about the underlying data distribution. It can also handle both numerical and categorical data         
            
  It can also handle both numerical and categorical data, making it a flexible choice for various types of datasets in classification and regression tasks. 
            
  It is a non-parametric method that makes predictions based on the similarity of data points in a given dataset. 
            
  K-NN is less sensitive to outliers compared to other algorithms.

  The K-NN algorithm works by finding the K nearest neighbors to a given data point based on a distance metric, such as Euclidean distance. 
            
  The class or value of the data point is then determined by the majority vote or average of the K neighbors. 
            
  This approach allows the algorithm to adapt to different patterns and make predictions based on the local structure of the data.
              
  $$
  Euclidean\ Distance
  $$
              
  $$
  m = d = \sqrt{(X_2-X_1)^2 + (y_2 - y_1)^2}
  $$
            
  ***
          
  """)
  
  #########
  # SIDEBAR
  #
  # # Sidebar - Prediction selection
  attribute_lst = ["class", "lug_boot", "safety", "persons", "door", "maint", "buying"]
  # attribute_select = st.sidebar.selectbox('SelectBox', attribute_lst, None)
  # attribute_multiselect = st.sidebar.multiselect('Multiselect', attribute_lst, attribute_lst)
  # st.subheader("attribute: selected")
  # st.write(attribute_select)

  # if attribute_select:
  #     st.write(f"SET")
  # else:
  #     st.write("NOT SET")

  ######
  # MAIN
  data = pd.read_csv(("data/CarDataSet/car.data"))
  st.subheader("raw data")
  st.write(data.head(3))
  
  # cls_list = data["class"]
  # log_this(cls_list, "cls_list")

  # Get column names
  col_names = data.columns
  log_this(col_names, "col_names")
  # st.write(f"col_names: {col_names}")

  # encode text labels into integer values
  le = preprocessing.LabelEncoder()

  buying = le.fit_transform(list(data["buying"]))
  maint = le.fit_transform(list(data["maint"]))
  door = le.fit_transform(list(data["door"]))
  persons = le.fit_transform(list(data["persons"]))
  safety = le.fit_transform(list(data["safety"]))
  lug_boot = le.fit_transform(list(data["lug_boot"]))    
  cls = le.fit_transform(list(data["class"]))
  
  # st.write(f"buying: {buying}")    
  predict = "class"
  # predict = attribute_select
  # st.subheader("features")
  # st.write(f"predict: {predict}")

  # features
  X = list(zip(buying, maint, door, persons, lug_boot, safety))
  st.write(X)

  # labels
  y = list(cls)

  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

  n_neighbors = 5
  model = KNeighborsClassifier(n_neighbors)
  st.write(f"model: ", model, "k = ", n_neighbors)

  model.fit(x_train, y_train)
  acc = model.score(x_test, y_test)
  st.header('metrics')
  st.write(f"accuracy: ", acc)
  
  # get unique values
  names = list(set(data["class"]))
  # st.write(f"names: ", names)

  predicted = model.predict(x_test) 

  ###############
  # DISCREPANCIES
  #
  st.header("discrepances")
  x_test_discrepancies = []
  for x in range(len(predicted)):
    print(("Predicted", names[predicted[x]], "Data:", x_test[x], "Actual:", names[y_test[x]]))

    n = model.kneighbors([x_test[x]], 9, True) 
    
    print("N: ", n)       

    # if predicted[x] != y_test[x]:
    #   x_test_discrepancies.append(x)


  d_len = len(x_test_discrepancies)
  st.write("count: ", d_len)

  d_df = pd.DataFrame(x_test_discrepancies)

  st.write(d_df.head(d_len))
  # st.write(("Predicted", predicted[xtd], "Data:", x_test[xtd], "Actual:", y_test[xtd]))        

  # # KNEIGBOURS

def main():
    logger.info('--------') 
    # logger.info('Calling Regression Test')
    # regression_test()
    k_nearest_neighbour()
    
if __name__ == '__main__':
    main()

