from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import os
import random
import pickle
from datetime import date, timedelta, datetime
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from PIL import Image
import ta
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.accounts import AccountDetails
from oandapyV20.endpoints.orders import OrderCreate
import requests
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, DBSCAN
from supabase import create_client
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error, r2_score

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'

app = FastAPI(title="Forex Signal Generator API", description="FastAPI backend for forex trading signals using LSTM and CNN-LSTM models")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Constants
ACCESS_TOKEN = 'd917178f8075576c341cbe85848de18e-9575706fb366ffd63dbbd057ecc8d847'
ACCOUNT_ID = '101-004-31663011-001'

# Supabase setup
def get_supabase_client():
    supabase_url = 'https://bpxzfdxxidlfzvgdmwgk.supabase.co'
    supabase_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJweHpmZHh4aWRsZnp2Z2Rtd2drIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDI3NjM0MTQsImV4cCI6MjA1ODMzOTQxNH0.vQq2-VYCJyTQDq3QN2mJprmmBR2w7HMorqBuzz43HRU'
    return create_client(supabase_url, supabase_key)

supabase = get_supabase_client()

# Load logos
try:
    priscomac_logo = Image.open("IMG-20250531-WA0006 (2).jpg")
    bitcoin_logo = Image.open('Photos 5_31_2025 10_33_31 AM (2).png')
    priscomac_resize = priscomac_logo.resize((200, 100))
except:
    pass

# Global variables for models and scaler
scaler = MinMaxScaler(feature_range=(0, 1))
lstm_model = None
cnn_lstm_model = None
data_df = None

from app.routers.data_router import router as data_router
from app.routers.model_router import router as model_router
from app.routers.signal_router import router as signal_router
from app.routers.indicators_router import router as indicators_router
from app.routers.trading_router import router as trading_router

app.include_router(data_router)
app.include_router(model_router)
app.include_router(signal_router)
app.include_router(indicators_router)
app.include_router(trading_router)

@app.get("/")
async def read_root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(f.read())
    return html_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
