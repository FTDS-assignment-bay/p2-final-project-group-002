from sklearn.preprocessing import MinMaxScaler
import numpy as np

def train_test_split(train):
    scaler = MinMaxScaler(feature_range= (0, 1))
    scaled_test = scaler.fit_transform(np.array(train).reshape(-1, 1)) 
    
    # Creating a testing set with 60 time-steps and 1 output
    X_test = []
    y_test = []

    for i in range(5, len(scaled_test)):
        X_test.append(scaled_test[i-5:i, 0])
        y_test.append(scaled_test[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_test, y_test, scaler