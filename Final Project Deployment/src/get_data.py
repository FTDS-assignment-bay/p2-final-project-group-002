import yfinance as yf
from datetime import datetime, timedelta
import numpy as np


def stock_data (tick):
  data = yf.download(tick, start=datetime.now() - timedelta(days=365),end=datetime.now())
  return data.Close