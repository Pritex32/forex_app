from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.api import SimpleExpSmoothing
import matplotlib.pyplot as plt

router = APIRouter(prefix="/api/models", tags=["models"])

class TrainRequest(BaseModel):
    data: list  # list of dicts from data fetch
    model_type: str  # "lstm" or "cnn_lstm"

class PredictRequest(BaseModel):
    model_type: str
    n_periods: int = 5

global scaler, lstm_model, cnn_lstm_model, data_df

scaler = MinMaxScaler(feature_range=(0, 1))
lstm_model = None
cnn_lstm_model = None
data_df = None

@router.post("/train")
async def train_model(request: TrainRequest):
    global scaler, lstm_model, cnn_lstm_model, data_df

    df = pd.DataFrame(request.data)
    if df.empty:
        raise HTTPException(status_code=400, detail="No data provided")

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    data_df = df.copy()
    open_price = df[['open']].values

    scaler.fit(open_price)
    scaler_data = scaler.transform(open_price)

    x = []
    y = []
    for i in range(60, len(scaler_data)):
        x.append(scaler_data[i-60:i])
        y.append(scaler_data[i])
    x = np.array(x)
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    if request.model_type == "lstm":
        lstm = Sequential()
        lstm.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        lstm.add(Dropout(0.5))
        lstm.add(LSTM(units=50, return_sequences=True))
        lstm.add(Dropout(0.5))
        lstm.add(LSTM(units=50, return_sequences=True))
        lstm.add(Dropout(0.5))
        lstm.add(LSTM(units=50, return_sequences=False))
        lstm.add(Dropout(0.5))
        lstm.add(Dense(units=1))

        lstm.compile(optimizer='adam', loss='mean_absolute_error')
        lstm.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=1, verbose=1)
        lstm_model = lstm

        # Save model
        lstm.save('models/lstm_model.h5')

        # Predictions for metrics
        pred = lstm.predict(x_test)
        inv_pred = scaler.inverse_transform(pred)
        inv_y_test = scaler.inverse_transform(y_test)
        rmse = np.sqrt(mean_squared_error(inv_y_test, inv_pred))
        r2 = r2_score(inv_y_test, inv_pred)

        return {"message": "LSTM model trained", "rmse": rmse, "r2": r2}

    elif request.model_type == "cnn_lstm":
        cnn_lstm = Sequential()
        cnn_lstm.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
        cnn_lstm.add(MaxPooling1D(2))
        cnn_lstm.add(LSTM(units=50))
        cnn_lstm.add(Dropout(0.3))
        cnn_lstm.add(Dense(units=1))

        cnn_lstm.compile(optimizer='adam', loss='mse')
        cnn_lstm.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=1, verbose=1)
        cnn_lstm_model = cnn_lstm

        # Save model
        cnn_lstm.save('models/cnn_lstm_model.h5')

        return {"message": "CNN-LSTM model trained"}

    else:
        raise HTTPException(status_code=400, detail="Invalid model type")

@router.post("/predict")
async def predict_future(request: PredictRequest):
    global scaler, lstm_model, cnn_lstm_model, data_df

    if data_df is None:
        raise HTTPException(status_code=400, detail="No data loaded. Train model first.")

    if request.model_type == "lstm" and lstm_model is None:
        if os.path.exists('models/lstm_model.h5'):
            lstm_model = keras.models.load_model('models/lstm_model.h5')
        else:
            raise HTTPException(status_code=400, detail="LSTM model not trained")

    if request.model_type == "cnn_lstm" and cnn_lstm_model is None:
        if os.path.exists('models/cnn_lstm_model.h5'):
            cnn_lstm_model = keras.models.load_model('models/cnn_lstm_model.h5')
        else:
            raise HTTPException(status_code=400, detail="CNN-LSTM model not trained")

    last_days = scaler.transform(data_df[['open']].values[-60:]).reshape(1, 60, 1)
    future_prediction = []

    for _ in range(request.n_periods):
        if request.model_type == "lstm":
            nxt_pred = lstm_model.predict(last_days)
        elif request.model_type == "cnn_lstm":
            nxt_pred = cnn_lstm_model.predict(last_days)
        future_prediction.append(nxt_pred[0, 0])
        last_days = np.append(last_days[:, 1:, :], [[[nxt_pred[0, 0]]]], axis=1)

    forecast_array = np.array(future_prediction)
    future_prediction_inv = scaler.inverse_transform(forecast_array.reshape(-1, 1))

    future_days = pd.date_range(start=data_df.index[-1] + pd.Timedelta(days=1), periods=request.n_periods, freq='D')
    f_df = pd.DataFrame({'dates': future_days, 'open': future_prediction_inv.flatten()})

    return f_df.to_dict('records')
