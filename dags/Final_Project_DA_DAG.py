from airflow.models import DAG
from airflow.operators.python import PythonOperator
#from airflow.providers.postgres.operators.postgres import PostgresOperator
# from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
from sqlalchemy import create_engine #koneksi ke postgres
import pandas as pd
import pytz
from airflow.utils.task_group import TaskGroup
import yfinance as yf
# from elasticsearch.helpers import bulk

# Define the function to fetch data from Yahoo Finance

def fetch_stock_data_daily():
    ''' 
    Fetch daily stock data from Yahoo Finance within the specified date range
    '''
    # Define the ticker symbols
    ticker_symbols = ['BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'BBNI.JK', 'BRIS.JK']
    
    # Define the timezone (GMT+7)
    timezone = pytz.timezone('Asia/Jakarta')
    
    # Set the desired end date (March 31, 2024)
    end_date = datetime(2024, 3, 31, 23, 59, 59, tzinfo=timezone)
    
    # Convert end_date to UTC
    end_date_utc = end_date.astimezone(pytz.utc)
    
    # Calculate the start date 5 years ago from the end date
    start_date = end_date_utc - timedelta(days=365*5)

    # Loop through each ticker symbol
    for ticker_symbol in ticker_symbols:
        # Get data on this ticker within the specified date range
        ticker_data = yf.download(ticker_symbol, start=start_date, end=end_date_utc)
        
        # Add a 'Bank_ID' column to identify the bank
        ticker_data['Bank_ID'] = ticker_symbol.split('.')[0]
        
        # Reset index to include 'Date' column in DataFrame
        ticker_data.reset_index(inplace=True)
        
        # Save the data to a CSV file
        file_path = f'/opt/airflow/dags/stock_data_{ticker_symbol}.csv'  # Adjusted file path
        ticker_data.to_csv(file_path, index=False)

def merge_dataset(): 
    ''' 
    Merging datasets based on ticker symbols
    '''
    
    # Read the CSV files into DataFrames
    df_BCA = pd.read_csv(f"/opt/airflow/dags/stock_data_BBCA.JK.csv")
    df_BRI = pd.read_csv(f"/opt/airflow/dags/stock_data_BBRI.JK.csv")
    df_MANDIRI = pd.read_csv(f"/opt/airflow/dags/stock_data_BMRI.JK.csv")
    df_BNI = pd.read_csv(f"/opt/airflow/dags/stock_data_BBNI.JK.csv")
    df_BSI = pd.read_csv(f"/opt/airflow/dags/stock_data_BRIS.JK.csv")
    
    # Add a 'Bank_ID' column to each DataFrame
    df_BCA['Bank_ID'] = 'BBCA'
    df_BRI['Bank_ID'] = 'BBRI'
    df_MANDIRI['Bank_ID'] = 'BMRI'
    df_BNI['Bank_ID'] = 'BBNI'
    df_BSI['Bank_ID'] = 'BRIS'

    # Concatenate the DataFrames vertically
    merged_df = pd.concat([df_BCA, df_BRI, df_MANDIRI, df_BNI, df_BSI], ignore_index=True)
    
    # Move 'Bank_ID' column to the first position
    cols = merged_df.columns.tolist()
    cols = ['Bank_ID'] + [col for col in cols if col != 'Bank_ID']
    merged_df = merged_df[cols]
    
    # Save the merged data to a CSV file
    merged_df.to_csv('/opt/airflow/dags/dataset_for_analysis.csv', index=False)


def load_csv_to_postgres():

    """
    Loading raw CSV file to postgres
    
    """

    database = "stock_final_project"
    username = "admin"
    password = "admin"
    host = "postgres"

    # Membuat URL koneksi PostgreSQL
    postgres_url = f"postgresql+psycopg2://{username}:{password}@{host}/{database}"

    # Gunakan URL ini saat membuat koneksi SQLAlchemy
    engine = create_engine(postgres_url)
    # engine= create_engine("postgresql+psycopg2://airflow:airflow@postgres/airflow")
    conn = engine.connect()

    df = pd.read_csv('/opt/airflow/dags/dataset_for_analysis.csv')
    #df.to_sql(nama_table_db, conn, index=False, if_exists='replace')
    df.to_sql('table_DA', conn, index=False, if_exists='replace')
    

     
        
# Define default configuration settings for a task or workflow in Apache Airflow
# Define the timezone (GMT+7)
timezone = pytz.timezone('Asia/Jakarta')

default_args = {
    'owner': 'group_2',
    'start_date': datetime(2020, 12 , 25, 12, 00, tzinfo=timezone) # Set start date behind project due date to:
    # 1. Avoid accidental execution before due date
    # 2. Allow time for testing before the actual deadline
    # 3. Ensure consistency in workflow scheduling
    # 4. Facilitate backfilling of historical data without triggering premature executions
}
with DAG(
    "Group_2_Stock_Data_Automation_DA",
    description='Final Project',
    schedule_interval='00 00 * * *', # Set the schedule interval for executing the Airflow DAG
    default_args=default_args, # Set the default arguments for the DAG
    catchup=False # Disable catch-up scheduling to prevent backfilling for past intervals
) as dag:
        
    fetch_data_yfinance = PythonOperator(
        task_id='fetch_daily_stock_data_Extract',
        python_callable=fetch_stock_data_daily)
    
    merging_dataset = PythonOperator(
        task_id='merging_dataset_Transform',
        python_callable=merge_dataset)
    
    load_data_to_database = PythonOperator(
        task_id='loading_data_to_postgresDB_Load',
        python_callable=load_csv_to_postgres)

    fetch_data_yfinance >> merging_dataset >> load_data_to_database
    
"""
In Cron notation (which Apache Airflow uses for scheduling), the format is as follows: minute, hour, day_of_month, month, day_of_week.

So, if you change schedule_interval='10 5 * * *', it means:

10: Execute at the 10th minute of the hour.
5: Execute at 5 AM.
*: Execute every day of the month.
*: Execute every month.
*: Execute every day of the week.
"""



