from airflow.models import DAG
from airflow.operators.python import PythonOperator
#from airflow.providers.postgres.operators.postgres import PostgresOperator
# from airflow.utils.task_group import TaskGroup
import pytz
from datetime import datetime, timedelta
from sqlalchemy import create_engine #koneksi ke postgres
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from airflow.utils.task_group import TaskGroup
import yfinance as yf
# from elasticsearch.helpers import bulk


def fetch_stock_data_daily():
    # Define the ticker symbols
    ticker_symbols = ['BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'BBNI.JK', 'BRIS.JK']

    # Define the start and end dates for the year 2024
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 3, 31)

    # Loop through each ticker symbol
    for ticker_symbol in ticker_symbols:
        # Get data on this ticker within the specified date range
        ticker_data = yf.download(ticker_symbol, start=start_date, end=end_date)

        # Filter the DataFrame to keep only the 'Close' column
        dataset = ticker_data["Close"]
        dataset = pd.DataFrame(dataset)

        # Save the data to a CSV file
        file_path = f'/opt/airflow/dags/stock_data_ds_{ticker_symbol}.csv'
        dataset.to_csv(file_path, index=False)


def load_data_to_db():

    """
    Loading raw CSV file to postgres
    
    """

    database = "stock_final_project"
    username = "admin"
    password = "admin"
    host = "postgres"
    port = '5434'
    schema = 'data_scientist'

    # Membuat URL koneksi PostgreSQL
    postgres_url = f"postgresql+psycopg2://{username}:{password}@{host}/{database}"


    # Create a SQLAlchemy engine to connect to the PostgreSQL database
    engine = create_engine(postgres_url)
    conn = engine.connect()

    # Define the directory where CSV files are located
    csv_directory = '/opt/airflow/dags/'

    # Define the list of CSV files to load into the database
    csv_files = ['stock_data_ds_BBCA.JK.csv', 'stock_data_ds_BBRI.JK.csv', 'stock_data_ds_BMRI.JK.csv', 'stock_data_ds_BBNI.JK.csv', 'stock_data_ds_BRIS.JK.csv']

    # Loop through each CSV file and load its data into the database
    for csv_file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_directory + csv_file)

        # Define the table name based on the CSV file name
        table_name = csv_file.split('.')[0].replace("'", '')

        # Load the data from the DataFrame into the database table
        df.to_sql(table_name, conn, schema=schema, if_exists='replace', index=False)

def fetch_data_from_postgres():
    """
    Loading table from postgres to pandas
    
    """

    stock_tables = ["stock_data_ds_BBCA", "stock_data_ds_BBRI", "stock_data_ds_BMRI", "stock_data_ds_BBNI", "stock_data_ds_BRIS"]

    # fetch data
    database = "stock_final_project"
    username = "admin"
    password = "admin"
    host = "postgres" 
    schema = 'data_scientist'

    # Membuat URL koneksi PostgreSQL
    postgres_url = f"postgresql+psycopg2://{username}:{password}@{host}/{database}"

    # Gunakan URL ini saat membuat koneksi SQLAlchemy
    engine = create_engine(postgres_url)
    conn = engine.connect()

    # Loop through each CSV file and load its data into the database
    for table in stock_tables:
       df = pd.read_sql_query(f'select * from {schema}."{table}"', conn)
       # Save the data to a CSV file
       file_path = f'/opt/airflow/dags/stock_data_ds_{stock_tables}.csv'  # Adjusted file path
       df.to_csv(file_path, index=False)
    

def feature_engineering(): 
    ''' 
    Splitting the train and test for model analysis
    '''
    # Reading data
    df_BCA = pd.read_csv("/opt/airflow/dags/stock_data_ds_BBCA.JK.csv")
    df_BRI = pd.read_csv("/opt/airflow/dags/stock_data_ds_BBRI.JK.csv")
    df_MANDIRI = pd.read_csv("/opt/airflow/dags/stock_data_ds_BMRI.JK.csv")
    df_BNI = pd.read_csv("/opt/airflow/dags/stock_data_ds_BBNI.JK.csv")
    df_BSI = pd.read_csv("/opt/airflow/dags/stock_data_ds_BRIS.JK.csv")


    for data_df, ticker_symbol in zip([df_BCA, df_BRI, df_MANDIRI, df_BNI, df_BSI], 
                                      ['BBCA', 'BBRI', 'BMRI', 'BBNI', 'BRIS']):
        
        # Calculate the index to split the data (75% train, 25% test)
        train_size = int(len(data_df) * 0.85)
        test_size = len(data_df) - train_size

        # Splitting the data into train and test sets
        train_data = data_df.iloc[:-test_size]
        test_data = data_df.iloc[-test_size:]

        # Create dataset with 60 time steps and (59 input and only 1 output in each ) as this is a regression problem
        X_train = []
        y_train = []

        for i in range(60, len(train_data)):
            X_train.append(train_data.iloc[i-5:i, 0].values)  # Corrected slicing here
            y_train.append(train_data.iloc[i, 0])

        # convert Xs, y to arrays
        X_train, y_train = np.array(X_train), np.array(y_train)

        # reshape data -->> Xs = (rows, timestep, [n_cols = 2]), y = (rows,   )
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Creating a testing set with 60 time-steps and 1 output
        X_test = []
        y_test = []

        for i in range(5, len(test_data)):
            X_test.append(test_data.iloc[i-5:i, 0].values)  # Corrected slicing here
            y_test.append(test_data.iloc[i, 0])

        X_test, y_test = np.array(X_test), np.array(y_test)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        ''' 
        Scaling the data for model analysis
        '''

        # Scaling data (You can apply any scaling technique here)
        # For example, Min-Max Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Scaling X_train and X_test
        scaled_X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
        scaled_X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

        # Scaling y_train and y_test
        scaled_y_train = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        scaled_y_test = scaler.transform(y_test.reshape(-1, 1)).flatten()

        # Convert scaled data back to DataFrame
        train_scaled_df = pd.DataFrame(scaled_y_train, columns=['Close'])  # Assuming 'Close' is the column name
        test_scaled_df = pd.DataFrame(scaled_y_test, columns=['Close'])  # Assuming 'Close' is the column name

        # Save scaled datasets to CSV files
        train_scaled_df.to_csv(f"/opt/airflow/dags/{ticker_symbol}_y_train_scaled.csv", index=False)
        test_scaled_df.to_csv(f"/opt/airflow/dags/{ticker_symbol}_y_test_scaled.csv", index=False)
        
        # Save scaled X_train and X_test to CSV files
        pd.DataFrame(scaled_X_train.reshape(scaled_X_train.shape[0], -1)).to_csv(f"/opt/airflow/dags/{ticker_symbol}_X_train_scaled.csv", index=False)
        pd.DataFrame(scaled_X_test.reshape(scaled_X_test.shape[0], -1)).to_csv(f"/opt/airflow/dags/{ticker_symbol}_X_test_scaled.csv", index=False)

def load_scaled_data():

    """
    Loading raw CSV file to postgres
    
    """

    database = "stock_final_project"
    username = "admin"
    password = "admin"
    host = "postgres"
    port = '5434'
    schema = 'data_scientist'

    # Membuat URL koneksi PostgreSQL
    postgres_url = f"postgresql+psycopg2://{username}:{password}@{host}/{database}"


    # Create a SQLAlchemy engine to connect to the PostgreSQL database
    engine = create_engine(postgres_url)
    conn = engine.connect()

    # Define the directory where CSV files are located
    csv_directory = '/opt/airflow/dags/'

        # Define the list of ticker symbols
    ticker_symbols = ['BBCA', 'BBNI', 'BBRI', 'BMRI', 'BRIS']

    # Define the list of file types
    file_types = ['X_train_scaled.csv', 'y_train_scaled.csv', 'X_test_scaled.csv', 'y_test_scaled.csv']

    # Initialize an empty list to store the CSV files
    scaled_csv_files = []

    # Generate the list of CSV files based on the ticker symbols and file types
    for ticker in ticker_symbols:
        for file_type in file_types:
            scaled_csv_files.append(f'{ticker}_{file_type}')

    # Loop through each CSV file and load its data into the database
    for csv_files in scaled_csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_directory + csv_files)

        # Define the table name based on the CSV file name
        table_name = csv_files.split('.')[0]

        # Load the data from the DataFrame into the database table
        df.to_sql(table_name, conn, schema=schema, if_exists='replace', index=False)


timezone = pytz.timezone('Asia/Jakarta')

# Define default configuration settings for a task or workflow in Apache Airflow
default_args = {
    'owner': 'group_2',
    'start_date': datetime(2020, 12 , 25, 12, 00, tzinfo=timezone) # Set start date behind project due date to:
    # 1. Avoid accidental execution before due date
    # 2. Allow time for testing before the actual deadline
    # 3. Ensure consistency in workflow scheduling
    # 4. Facilitate backfilling of historical data without triggering premature executions
}
with DAG(
    "Group_2_Stock_Data_Automation_DS", 
    description='Final Project',
    schedule_interval='00 12 * * *', # Set the schedule interval for executing the Airflow DAG
    default_args=default_args, # Set the default arguments for the DAG
    catchup=False # Disable catch-up scheduling to prevent backfilling for past intervals
) as dag:
        
    extract_data_from_yfinance = PythonOperator(
        task_id='fetch_daily_stock_data_Extract',
        python_callable=fetch_stock_data_daily)
    
    load_data = PythonOperator(
        task_id='loading_data_to_postgresDB',
        python_callable=load_data_to_db)
    
    fetch_data = PythonOperator(
        task_id='fetching_data_from_postgresDB',
        python_callable=fetch_data_from_postgres)

    data_transform = PythonOperator(
        task_id='train_test_split_Transform',
        python_callable=feature_engineering)
    
    load_to_data_scientist = PythonOperator(
        task_id='load_scaled_data_to_DS',
        python_callable=load_scaled_data)
    

    extract_data_from_yfinance >> load_data >> fetch_data >> data_transform >> load_to_data_scientist
    
"""
In Cron notation (which Apache Airflow uses for scheduling), the format is as follows: minute, hour, day_of_month, month, day_of_week.

So, if you change schedule_interval='10 5 * * *', it means:

10: Execute at the 10th minute of the hour.
5: Execute at 5 AM.
*: Execute every day of the month.
*: Execute every month.
*: Execute every day of the week.
"""



