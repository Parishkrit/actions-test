import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score
from joblib import dump


dataset= pd.read_csv('winequality-red.csv')
# dataset=dataset.to_pandas_dataframe()

x=dataset.drop(["quality"],axis=1)
y=dataset["quality"]

#scale down the data
scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

#spliting the dataset
x_train,x_test,y_train,y_test=train_test_split(x_scaler,y,test_size=0.33,random_state=32)

#creating the model
clf1=DecisionTreeClassifier(max_depth=7,min_impurity_decrease=0.01,random_state=32)
# clf2=RandomForestClassifier(n_estimators=100)

clf1.fit(x_train,y_train)
# clf2.fit(x_train,y_train)

#predicting
y_pred1=clf1.predict(x_test)
acc1=accuracy_score(y_test,y_pred1)
# pre1=precision_score(y_test,y_pred1)
# recall1=recall_score(y_test,y_pred1)
# run.log("acc_dt,pre_dt,recall_dt",acc1,pre1,recall1)
# run.log("acc_dt,",acc1)


# y_pred2=clf2.predict(x_test)
# acc2=accuracy_score(y_test,y_pred2)
# pre2=precision_score(y_test,y_pred2)
# recall2=recall_score(y_test,y_pred2)
# run.log("acc+rf,pre_rf,recall_rf",acc2,pre2,recall2)
# run.log("acc+rf",acc2)

# run.log_list("accuracy_list",[acc1,acc2])
# run.log("max_acc",run.log(max([acc1,acc2])))

# print("dumping the models")
# import os 


# output_dir = 'Users/parishkrit.goel/model'
# os.makedirs(output_dir, exist_ok= True)

#creating dir for model
# if 'models' not in os.listdir():
    # os.mkdir('models')

# dump(clf1,'Users/parishkrit.goel/outputs/tree.pkl')
dump(clf1,'tree.pkl')
# dump(clf2,'Users/parishkrit.goel/outputs/rf.pkl')

# print("model dumping is completed")

# run.complete()
# print("experiment is completed")














# print('running the train.py file ..............................................')
# print('-'*1000)


# from azureml.core import Workspace, Run
# from azureml.core import Workspace, Dataset

# import numpy as np
# import os
# import argparse
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# import pandas as pd
# import joblib


# ws = Workspace.get(name="",
#                subscription_id='',
#                resource_group='MLOPS-practice')


# from azureml.core import Datastore
# az_store = Datastore.get(ws,"pg_datastore")

# from azureml.core import Dataset
# # #create the path of csv file
# dataset_path  = [(az_store,"winequality-white.csv")]
# dataset = Dataset.Tabular.from_delimited_files(path = dataset_path)

# az_dataset = dataset.get_by_name(ws,"wine_data")

# df = az_dataset.to_pandas_dataframe()


# X = df.drop(["quality"],axis = 1)
# y = df["quality"]  

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20)

# run = Run.get_context()

# #logging the shape of x_train,x_test,y_train,y_test
# run.log("size of x_train", X_train.shape)
# run.log("size of y_train", y_train.shape)

# run.log("size of x_test", X_test.shape)
# run.log("size of y_test", y_test.shape)


# # creating models
# m1 = LinearRegression()

# m1.fit(X_train,y_train)

# # prediciting Values
# y_pred = m1.predict(X_test)

# #accuracy
# acc1 = m1.score(X_test,y_test)
# run.log("Score of logistic regression ",acc1)
# acc2 = mean_squared_error(y_test, y_pred)
# run.log("MSE of logistic regression ",acc2)



# os.makedirs('outputs', exist_ok=True)
# joblib.dump(value=m1, filename='outputs/m1.pkl')


# # # Register the model
# # run.register_model(model_path='outputs/m1.pkl', model_name='lr_wine',
# #                 tags={'Training context':'Script'},
# #                 properties= 'MSE': str(acc2)})

# # Upload the model
# # run.upload_file(name = m1, path_or_stream = 'outputs/m1.pkl')

# run.complete()
