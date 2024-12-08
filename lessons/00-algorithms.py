import os
import pandas as pd
import numpy as np

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import linear_model, preprocessing

from sklearn import datasets
from sklearn import svm

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
  
def get_var_name(var):
    for name, value in locals().items():
        logger.info(f"var: {var}")
        if value is var:
            return name
          
def k_nearest_neighbour():
    logger.info('Starting k-nearest-neightbour') 
    
    data = pd.read_csv(("data/CarDataSet/car.data"))
    # logger.info(f"\n{data.head()}")
    
    cls_list = data["class"]
    log_this(cls_list, "cls_list")
    logger.info(f"uniq: {set(cls_list)}")
    
    # Get column names
    # col_names = data.columns
    # log_this(col_names, "col_names")

    # encode text labels into integer values
    le = preprocessing.LabelEncoder()
   
    # for attribute in col_names:
    #   # log_this(attribute, f"TSTER: {attribute}")

    #   attr_name = f"{attribute}"
    #   log_this(attr_name, f"attr_name")

    #   attr_lst = le.fit_transform(list(data[attribute]))
      
    #   logger.info(f"\nlst: {attribute} | uniq: {set(attr_lst)} | lst: {attr_lst}")
      
    buying = le.fit_transform(list(data["buying"]))
    # log_this(buying, "buying")
    
    maint = le.fit_transform(list(data["maint"]))
    # log_this(maint, "maint")

    door = le.fit_transform(list(data["door"]))
    # log_this(door, "door")

    persons = le.fit_transform(list(data["persons"]))
    # log_this(persons, "persons")
    
    safety = le.fit_transform(list(data["safety"]))
    # log_this(safety, "safety")
    
    lug_boot = le.fit_transform(list(data["lug_boot"]))
    # log_this(lug_boot, "lug_boot")
    
    cls = le.fit_transform(list(data["class"]))
    # log_this(cls, "class")
    
    predict = "class"
    
    # features
    X = list(zip(buying, maint, door, persons, lug_boot, safety, cls))
    # log_this(X, "X")
    
    # labels
    y = list(cls)
    # log_this(y, "y")
    
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    model = KNeighborsClassifier(n_neighbors=5)
    
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    # log_this(acc, "acc: ")
    
    # get unique values
    # cls_list = data["class"]
    names = list(set(data["class"]))
    # logger.info(f"names: {names}")

    predicted = model.predict(x_test)    

    for x in range(len(predicted)):
      logger.info(("Predicted", names[predicted[x]], "Data:", x_test[x], "Actual:", names[y_test[x]]))
      
      n = model.kneighbors([x_test[x]], 10, True)
      logger.info(f"N: {n}")
      
def support_vector_machine():
  logger.info('Starting support_vector_machine') 
  cancer = datasets.load_breast_cancer()
  
  # cdf = pd.DataFrame(cancer)
  # cdf.head
  
  f_names = cancer.feature_names
  # print(f_names)
  
  t_names = cancer.target_names 
  # print(t_names)

  X = cancer.data
  y = cancer.target  
  
  # logger.info("x: ", X, "y: ", y)  
  
  X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)
  
  # logger.info(X_train, y_test)
  
  classes = t_names
  
  log_this(cancer.target_names, "t_names: ")
  

def main():
    logger.info('--------') 
    # logger.info('Calling Regression Test')
    # regression_test()
    # k_nearest_neighbour()
    support_vector_machine()
    
if __name__ == '__main__':
    main()

