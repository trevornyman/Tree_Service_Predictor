#split data into testing and training set
import os
import settings
import pandas as pd

def read():
  tree_cleaning_data = pd.read_csv("311_Service_Requests_-_Tree_Debris.csv")
  return tree_cleaning_data




import numpy as np
from sklearn.cross_validation import train_test_split

X, y = tree_cleaning_data.drop('Wait_time', axis=1), tree_cleaning_data['Wait_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
