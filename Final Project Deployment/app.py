import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
from src import stock_data, train_test_split, model_predict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
import tensorflow as tf
import numpy as np
from tensorflow.keras.regularizers import L2
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, LSTM, Dropout


st.image('banner.png', caption='Image credit : Kumar Laxmi')
# Function to display result, history, and data information
def display_results(user_input,result, data_inf, train):
    st.subheader(f'Data for :blue[{user_input.upper()}] the past 5 days.')
    st.write(data_inf)
    st.subheader(f'Annual data line Chart :blue[{user_input.upper()}]')
    plot = sns.lineplot(train)
    st.pyplot(plot.get_figure())
    future_date = datetime.now() + timedelta(days=1)
    formatted_date = future_date.strftime('%Y-%m-%d')
    st.markdown(f'''
        ## Stock Prediction Analysis for <span style="font-size:24px;">{user_input.upper()}</span>
        
        <p style="font-size:24px;">
        {formatted_date} {user_input.upper()} predicted price is : <b>{float(f"{result:.2f}")}</b>
        </p>
    ''', unsafe_allow_html=True)
    
    
    
    
# Main function to run the app
def main():
    st.title("Smart Stock Prediction")
    selected_stock_symbol =  st.selectbox("Select your stock ticker:", ["BANK BCA", "BANK BNI", "BANK BRI", "BANK MANDIRI", "BANK BSI"])

    # Dictionary
    stock_symbols = {
        "BANK BCA": "BBCA.JK",
        "BANK BNI": "BBNI.JK",
        "BANK BRI": "BBRI.JK",
        "BANK MANDIRI": "BMRI.JK",
        "BANK BSI": "BRIS.JK"
    }

    # Memilih simbol saham berdasarkan input pengguna
    user_input = stock_symbols[selected_stock_symbol]

    stock_name = selected_stock_symbol + ' (' + user_input + ')' 

    if st.button("Predict"):
        with st.spinner('Loading...'):
            train = stock_data(user_input)
            if len(train)>0: 
                x_train, y_train, scaler = train_test_split(train)
                result = model_predict(user_input, x_train, scaler) 
                data_inf = train[-5:]
                display_results(stock_name,result[0][0], data_inf, train)
            else: # [0][0], data_inf
                st.write('Invalid stock ticker. Please verify the ticker symbol on the following website: [Yahoo Finance.](https://finance.yahoo.com/)')


if __name__ == "__main__":
    main()
