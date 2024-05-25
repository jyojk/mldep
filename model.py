# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import mlflow

with mlflow.start_run():

    dataset = pd.read_csv('hiring.csv')

    dataset['experience'].fillna(0, inplace=True)

    dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

    X = dataset.iloc[:, :3]

    #Converting words to integer values
    def convert_to_int(word):
        word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                    'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
        return word_dict[word]

    X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

    y = dataset.iloc[:, -1]

    #Splitting Training and Test Set
    #Since we have a very small dataset, we will train our model with all availabe data.

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    regressor = LinearRegression()

    #Fitting model with trainig data
    regressor.fit(X, y)

    df_test = pd.read_csv("test_hiring.csv")

    X_test = df_test.iloc[:,:-1]
    y_test = df_test['salary']
    print(X_test, y_test)

    # Log model parameters as MLflow metrics 
    mlflow.log_metric("training_data_size", len(X))
    mlflow.log_param("model_type", "LinearRegression")

    print(regressor.score(X_test, y_test))

    print(regressor.predict(X_test))
    print(r2_score(y_test, regressor.predict(X_test)))
    print(mean_absolute_error(y_test, regressor.predict(y_test)))

    # Saving model to disk
    pickle.dump(regressor, open('model.pkl','wb'))
    mlflow.log_artifact("model.pkl")


    # Loading model to compare the results
    model = pickle.load(open('model.pkl','rb'))
    print(model.predict([[2, 9, 6]]))
