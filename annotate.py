#split data into testing and training set
import os
import settings
import pandas as pd

def read():
  tree_cleaning_data = pd.read_csv("311_Service_Requests_-_Tree_Debris.csv")
  return tree_cleaning_data


import os
import settings
import pandas as pd


HEADERS = {"data" : ["Creation_date",
                    "Status",
                    "Completion_date",
                    "Service_request_number",
                    "Type_service_request",
                    "If_yes_where_is_debris_located?",
                    "Current_activity",
                    "Most_recent_action",
                    "Street_address",
                    "Zip_code",
                    "X_coordinate",
                    "Y_coordinate",
                    "Ward",
                    "Police_district",
                    "Community_area",
                    "Lat",
                    "Lon",
                    "Location"]
          }

import numpy as np
from sklearn.cross_validation import train_test_split

X, y = tree_cleaning_data.drop('Wait_time', axis=1), tree_cleaning_data['Wait_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



#change date fields from str to num so we can perform subtraction to get a measure of the time the request took to fulfill.


#we might want to predict 'Wait_time' from day, month, or year, so we'll want to split those off the date-string as separate variables.

for start in ["Creation Date", "Completion Date"]:
        #column = "".format(start)
        tree_cleaning_data["{}_day".format(start)] = pd.to_numeric(tree_cleaning_data[start].str.split('/').str.get(1))
        tree_cleaning_data["{}_year".format(start)] = pd.to_numeric(tree_cleaning_data[start].str.split('/').str.get(2))
        tree_cleaning_data["{}_month".format(start)] = pd.to_numeric(tree_cleaning_data[start].str.split('/').str.get(0))
        
        
        
#Now we can convert the date-strings to datetime so we can perform numerical calculations on them, specifically,
# we can calculate a variable called 'Wait_time' by subtracting the date a request is created from the date it's completed.
#'Wait_time' will then serve as our dependent variable in our analysis

tree_cleaning_data['Creation Date'] = pd.to_datetime(tree_cleaning_data['Creation Date'])
tree_cleaning_data['Completion Date'] = pd.to_datetime(tree_cleaning_data['Completion Date'])

#create new variable, 'Wait_time' that will serve as our measure for how long the requests are taking to fulfill.
tree_cleaning_data['Wait_time']= tree_cleaning_data['Completion Date'] - tree_cleaning_data['Creation Date']

#The data set includes a redundant header row on the 0th line, so we'll remove that
tree_cleaning_data = tree_cleaning_data.drop(tree_cleaning_data.index[[0]])


#we can also remove some columns that won't be useful for making our predictions.
#Since our data goes back 6 years, 'Status' is overwhelmingly "complete", and won't be a useful predictor
#'Type of Service Request' is always set to "tree removal", since this is tree removal data.
#'Current Activity' and 'Most Recent Action' aren't part of the initial information a user would enter; they 
# are added in later as the request makes its way through the pipeline.  Since we want to make 'Wait_time' predictions
# on the data initially available, we won't use these as predictors.
#'Street Address' is a bit too fine-grained to be useful, and 'Location' is redundant combination of 'Latitude' and 'Longitude'.

worthless_labels = ['Status', 'Type of Service Request' , 'Current Activity' , 'Most Recent Action', 'Street Address', 'Location']
tree_cleaning_data = tree_cleaning_data.drop(tree_cleaning_data[worthless_labels], axis = 1)

#initially, 'If Yes, where is the debris located?' was a string, but it's really a categorial variable with 3 distinct values:
#alley, parkway, or vacant lot.  So, we can convert this to a categorical int and use it in our machine learning algorithm.

tree_cleaning_data['If Yes, where is the debris located?'] = tree_cleaning_data['If Yes, where is the debris located?'].astype('category').cat.codes


#Next we'll define a function elim to eliminate all rows that have any missing values.  Because we got a feel for our data
#before diving into our analysis, we know that we won't lose that much data by removing these rows.  (We lost .41% of our data).
def elim(df):
    for column in df.columns.values:
        df = df[pd.notnull(df[column])]
    return df
    
tree_cleaning_data = elim(tree_cleaning_data)
