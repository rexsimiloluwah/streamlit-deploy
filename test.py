import numpy  as np
import pickle
import sklearn
import pandas as pd


train_data = pd.read_csv('./data/train_data.csv')

# Dropping the target feature/ predictor column from the dataset

target = train_data['Surface_Roughness(um)']
train_data.drop('Surface_Roughness(um)', axis=1, inplace=True)

#Splitting the Data into training and validation sets 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_data, target, test_size=0.2, random_state= 42)

# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)

#Importing the required Libraries
#Importing the required Library from sklearn
from sklearn.svm import SVR 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


svr_poly = SVR(kernel='poly', C=1e3, degree=2)  #Applying a polynomial Kernel

def baseline_performance():
    models = []

    models.append(('Linear Regression', LinearRegression()))
    models.append(('Stochastic Gradient Boosting', SGDRegressor()))
    models.append(('DecisionTreeRegressor', DecisionTreeRegressor()))
    models.append(('KNeighborsRegressor', KNeighborsRegressor()))
    models.append(('RandomForestRegressor', RandomForestRegressor()))
    models.append(('Multi-Layer Perceptron', MLPRegressor()))


    #Performance Metric used - Root Mean squared Error

    from sklearn.metrics import mean_squared_error

    for name,model in models:
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        print('{} : {}'.format(name, mean_squared_error(pred, y_test)))

# baseline_performance()

# tree = DecisionTreeRegressor()
# tree.fit(x_train, y_train)

# pickle.dump(tree, open("./models/tree.pickle", "wb"))

# svr_poly.fit(x_train, y_train)

# pickle.dump(svr_poly, open("./models/support_vector_regression.pickle", "wb"))

# forest = RandomForestRegressor()
# forest.fit(x_train, y_train)
# pickle.dump(forest, open("./models/random_forest.pickle", "wb"))

lin = LinearRegression()
lin.fit(x_train, y_train)
pickle.dump( lin , open("./models/linear_regression.pickle", "wb"))