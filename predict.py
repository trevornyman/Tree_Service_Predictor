import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

tree_cleaning_data = pd.read_csv("311_Service_Requests_-_Tree_Debris.csv")


for start in ["Creation Date", "Completion Date"]:
        #column = "".format(start)
        tree_cleaning_data["{}_day".format(start)] = pd.to_numeric(tree_cleaning_data[start].str.split('/').str.get(1))
        tree_cleaning_data["{}_year".format(start)] = pd.to_numeric(tree_cleaning_data[start].str.split('/').str.get(2))
        tree_cleaning_data["{}_month".format(start)] = pd.to_numeric(tree_cleaning_data[start].str.split('/').str.get(0))
        
tree_cleaning_data['Creation Date'] = pd.to_datetime(tree_cleaning_data['Creation Date'])
tree_cleaning_data['Completion Date'] = pd.to_datetime(tree_cleaning_data['Completion Date'])

tree_cleaning_data['Wait_time']= tree_cleaning_data['Completion Date'] - tree_cleaning_data['Creation Date']

wait_time_as_number = []
for time in tree_cleaning_data['Wait_time']:
    time = time.total_seconds()
    wait_time_as_number.append(time/86400)

tree_cleaning_data['Wait_time'] = wait_time_as_number


tree_cleaning_data = tree_cleaning_data.drop(tree_cleaning_data.index[[0]])
worthless_labels = ['Status', 'Type of Service Request' , 'Current Activity' , 'Most Recent Action', 'Street Address', 'Location']
tree_cleaning_data = tree_cleaning_data.drop(tree_cleaning_data[worthless_labels], axis = 1)

tree_cleaning_data['If Yes, where is the debris located?'] = tree_cleaning_data['If Yes, where is the debris located?'].astype('category').cat.codes

#print (tree_cleaning_data.dtypes)

#print (len(tree_cleaning_data))

def elim(df):
    for column in df.columns.values:
        df = df[pd.notnull(df[column])]
    return df
    
tree_cleaning_data = elim(tree_cleaning_data)



X, y = tree_cleaning_data.drop('Wait_time', axis=1), tree_cleaning_data['Wait_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train = X_train.drop(['Creation Date', 'Completion Date', 'Service Request Number'], axis=1)
X_test = X_test.drop(['Creation Date', 'Completion Date', 'Service Request Number'], axis=1)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

#results = pd.DataFrame(data=[pred, y_test])
#print (results)

print ("R2 score: %s" % model.score(X_test, y_test))
