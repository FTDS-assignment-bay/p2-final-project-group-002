import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import numpy as np
from tensorflow.keras.regularizers import L2
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, LSTM, Dropout

# Load the trained model
model_bca = tf.keras.models.load_model("model_bca.keras")
model_bni = tf.keras.models.load_model("model_bni.keras")
model_bri = tf.keras.models.load_model("model_bri.keras")
model_bsi = tf.keras.models.load_model("model_bsi.keras")
model_mri = tf.keras.models.load_model("model_mri.keras")

def model_predict(user_input, X_test, scaler):
  if user_input == 'BBCA.JK':
    predictions = model_bca.predict(X_test)
  elif user_input == 'BBNI.JK':
    predictions = model_bni.predict(X_test)
  elif user_input == 'BBRI.JK':
    predictions = model_bri.predict(X_test)
  elif user_input == 'BRIS.JK':
    predictions = model_bsi.predict(X_test)
  elif user_input == 'BMRI.JK':
    predictions = model_mri.predict(X_test)
  return scaler.inverse_transform(predictions)