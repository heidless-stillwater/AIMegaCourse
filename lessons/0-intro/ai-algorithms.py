import os
import pandas as pd
import numpy as np
import streamlit as st 

import sklearn
from sklearn import datasets
from sklearn import metrics
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import linear_model, preprocessing

from sklearn.preprocessing import scale
from sklearn.datasets  import load_digits
from sklearn.cluster import KMeans

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

  # st.write(""" 
    
  # (K-NN) algorithm is a versatile and widely used machine learning algorithm that is primarily used for its simplicity and ease of implementation. 
  
  # It does not require any assumptions about the underlying data distribution. It can also handle both numerical and categorical data         
            
  # It can also handle both numerical and categorical data, making it a flexible choice for various types of datasets in classification and regression tasks. 
            
  # It is a non-parametric method that makes predictions based on the similarity of data points in a given dataset. 
            
  # K-NN is less sensitive to outliers compared to other algorithms.

  # The K-NN algorithm works by finding the K nearest neighbors to a given data point based on a distance metric, such as Euclidean distance. 
            
  # The class or value of the data point is then determined by the majority vote or average of the K neighbors. 
            
  # This approach allows the algorithm to adapt to different patterns and make predictions based on the local structure of the data.
              
  # $$
  # Euclidean\ Distance
  # $$
              
  # $$
  # m = d = \sqrt{(X_2-X_1)^2 + (y_2 - y_1)^2}
  # $$
            
  # ***
          
  # """)
  
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

def support_vector_machine():
  logger.info('Support Vector Machine') 
  st.title("Support Vector Machine")
  st.subheader("classification & regression algorithm ")

  # st.write("""      

  # (SVN)
                  
  # $$
  # Euclidean\ Distance
  # $$

  # $$
  # f(X_1, X2) => X_3
  # $$
   
  # ***
          
  # """)
  
  #########
  # SIDEBAR
  #
  # # Sidebar - Prediction selection
  # attribute_lst = ["class", "lug_boot", "safety", "persons", "door", "maint", "buying"]
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
  # data = pd.read_csv(("data/CarDataSet/car.data"))

  cancer = datasets.load_breast_cancer()

  X = cancer.data   
  y = cancer.target

  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)

  # st.write(x_train, y_train)

  # get unique values
  # classes = list(set(data["class"]))
  classes = ["malignant", "benign"]

  clf = svm.SVC(kernel="linear", C=2)
  clf.fit(x_train, y_train)

  y_pred = clf.predict(x_test)

  acc = metrics.accuracy_score(y_test, y_pred)

  st.write("acc: ", acc )

def k_means_clustering():
  logger.info('k_means_clustering') 
  st.title("k_means_clustering")
  st.subheader("classification & regression algorithm ")

  st.write(""" 
      
  $$
  f(X_1, X2) => X_3
  $$
  
  ***

  #### How K-Means Clustering Works
  The **K-Means clustering** algorithm is a classification algorithm that follows the steps outlined below to cluster data points together. 

  It attempts to separate each area of our high dimensional space into sections that represent each class. 

  When we are using it to predict it will simply find what section our point is in and assign it to that class.

  #### steps:
  - Step 1: Randomly pick K points to place K centroids 
  - Step 2: Assign all of the data points to the centroids by distance. The closest centroid to a point is the one it is assigned to. 
  - Step 3: Average all of the points belonging to each centroid to find the middle of those clusters (center of mass). Place the corresponding centroids into that position. 
  - Step 4: Reassign every point once again to the closest centroid. 
  - Step 5: Repeat steps 3-4 until no point changes which centroid it belongs to.

  ***

  """)
  
  #########
  # SIDEBAR
  #
  # # Sidebar - Prediction selection
  # attribute_lst = ["class", "lug_boot", "safety", "persons", "door", "maint", "buying"]
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
  # data = pd.read_csv(("data/CarDataSet/car.data"))

  digits = load_digits()

  data = scale(digits.data)

  st.subheader("parameters")

  # digit_df = st.dataframe(data) 
  # digit_df.head(3)

  y = digits.target  

  # number of classifications
  k = len(np.unique(y))
  st.write('num centroids = k: ', k)

  # samples. features = data.shape
  def bench_k_means(estimator, name, data):
    estimator.fit(data)
    st.write('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

  clf = KMeans(n_clusters=k, init="random", n_init=10, max_iter=300)
  st.write('classifier function (clf): ', clf)

  st.subheader("k-means fit")
  bench_k_means(clf, "1", data)





def main():
    logger.info('--------') 
    # logger.info('Calling Regression Test')
    # regression_test()
    # k_nearest_neighbour()
    # support_vector_machine()
    k_means_clustering()
    
if __name__ == '__main__':
    main()

