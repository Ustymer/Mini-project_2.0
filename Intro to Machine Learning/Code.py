''' === Exercise: Your First Machine Learning Model === '''

import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *

print("Setup Complete")

y = home_data.SalePrice
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_names]
print(X.describe())
print(X.head())

from sklearn.tree import DecisionTreeRegressor
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(X, y)
predictions = iowa_model.predict(X)
print(predictions)

print(predictions[:5])
print(y.head())

''' === Exercise: Model Validation === '''

from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
iowa_model = DecisionTreeRegressor(random_state = 1)
iowa_model.fit(train_X, train_y)

# Predict with all validation observations
val_predictions = iowa_model.predict(val_X)
# print the top few validation predictions
print(val_predictions[:5])
# print the top few actual prices from validation data
print(val_y.head())

from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y, val_predictions)

print(val_mae)

''' === Exercise: Underfitting and Overfitting === '''

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
res = {}
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    res[my_mae] = max_leaf_nodes
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = res[min(res.keys())]
# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
# fit the final model using ALL data (training + validation)
final_model.fit(X, y)

''' === Exercise: Random Forests === '''

from sklearn.ensemble import RandomForestRegressor
# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
# fit your model
rf_model.fit(train_X, train_y)
# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_mae = mean_absolute_error(val_y, rf_model.predict(val_X))
print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))

''' === Exercise: Machine Learning Competitions === '''

# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state=1)
# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(X, y)
test_data_path = '../input/test.csv'
test_data = pd.read_csv(test_data_path)
test_X = test_data[features]
test_preds = rf_model_on_full_data.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)

