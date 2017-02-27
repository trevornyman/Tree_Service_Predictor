import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score




#processing data for analysis
def read():
  tree_cleaning_data = pd.read_csv("311_Service_Requests_-_Tree_Debris.csv")
  return tree_cleaning_data


def edit(data):
  for start in ["Creation Date", "Completion Date"]:
        #column = "".format(start)
        data["{}_day".format(start)] = pd.to_numeric(data[start].str.split('/').str.get(1))
        data["{}_year".format(start)] = pd.to_numeric(data[start].str.split('/').str.get(2))
        data["{}_month".format(start)] = pd.to_numeric(data[start].str.split('/').str.get(0))

  data['Creation Date'] = pd.to_datetime(data['Creation Date'])
  data['Completion Date'] = pd.to_datetime(data['Completion Date'])
  data['Wait_time']= data['Completion Date'] - data['Creation Date']
  data = data.drop(data.index[[0]])

  worthless_labels = ['Status', 
                    'Type of Service Request', 
                    'Current Activity', 
                    'Most Recent Action', 
                    'Street Address', 
                    'Location']

  data = data.drop(data[worthless_labels], axis = 1)
  data['If Yes, where is the debris located?'] = data['If Yes, where is the debris located?'].astype('category').cat.codes
  
  wait_time_as_number = []
  for time in data['Wait_time']:
    time = time.total_seconds()
    wait_time_as_number.append(time/86400)
  
  data['Wait_time'] = wait_time_as_number
  
  return data

def elim(df):
    for column in df.columns.values:
        df = df[pd.notnull(df[column])]
    return df

  
  
#splitting data
X, y = data.drop('Wait_time', axis=1), data['Wait_time']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train = X_train.drop(['Creation Date', 'Completion Date', 'Service Request Number'], axis=1)
X_test = X_test.drop(['Creation Date', 'Completion Date', 'Service Request Number'], axis=1)



#training model
model = LinearRegression()
model.fit(X_train, y_train)


#prediction
pred = model.predict(X_test)


#evaluating model
print ("R2 score: %s" % model.score(X_test, y_test))




if __name__ == "__main__":
  data = read()
  data = edit(data)
  data = elim(data)
  
