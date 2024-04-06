import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def TrainModel():
    data = pd.read_excel("Iris Flower.xlsx")
    data.head()
    data = data.drop('Id', axis = 1)

    x = data.drop(columns = ['Species'])
    y = data['Species']

    x_train, x_test, y_train, y_test = train_test_split(x.values, y, train_size=0.70)
    global model 
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    print(f"Accuracy: {model.score(x_test, y_test)}")

def PredictionModel(inputArray):
    input_data = np.array([inputArray])
    prediction = model.predict(input_data) # type: ignore
    return prediction