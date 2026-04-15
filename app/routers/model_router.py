from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import pickle
import tempfile
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import SimpleExpSmoothing
import matplotlib.pyplot as plt
from supabase import create_client
import xgboost as xgb
import ta

router = APIRouter(prefix="/api/models", tags=["models"])

# Supabase setup
def get_supabase_client():
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    return create_client(supabase_url, supabase_key)

supabase = get_supabase_client()

class TrainRequest(BaseModel):
    data: list  # list of dicts from data fetch
    model_type: str  # "lstm" or "cnn_lstm"

class PredictRequest(BaseModel):
    model_type: str
    n_periods: int = 5

global scaler, lstm_model, cnn_lstm_model, data_df, xgboost_model

scaler = MinMaxScaler(feature_range=(0, 1))
lstm_model = None
cnn_lstm_model = None
data_df = None
xgboost_model = None

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

    # Save data_df to Supabase
    data_bytes = pickle.dumps(data_df)
    supabase.storage.from_('models').upload('data_df.pkl', data_bytes, upsert=True)

    open_price = df[['open']].values

    scaler.fit(open_price)

    # Save scaler to Supabase
    scaler_bytes = pickle.dumps(scaler)
    supabase.storage.from_('models').upload('scaler.pkl', scaler_bytes, upsert=True)
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
        os.makedirs('models', exist_ok=True)
        lstm.save('models/lstm_model.h5')

        # Upload to Supabase
        with open('models/lstm_model.h5', 'rb') as f:
            model_bytes = f.read()
        supabase.storage.from_('models').upload('lstm_model.h5', model_bytes, upsert=True)

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
        os.makedirs('models', exist_ok=True)
        cnn_lstm.save('models/cnn_lstm_model.h5')

        # Upload to Supabase
        with open('models/cnn_lstm_model.h5', 'rb') as f:
            model_bytes = f.read()
        supabase.storage.from_('models').upload('cnn_lstm_model.h5', model_bytes, upsert=True)

        return {"message": "CNN-LSTM model trained"}

    elif request.model_type == "xgboost":
        # Calculate target: 1 if next close > current, -1 if <, 0 if ==
        data_df['target'] = np.where(data_df['close'].shift(-1) > data_df['close'], 1,
                                     np.where(data_df['close'].shift(-1) < data_df['close'], -1, 0))
        data_df.dropna(inplace=True)

        # Calculate features
        data_df['rsi'] = ta.momentum.RSIIndicator(data_df['close'], window=14).rsi()
        macd = ta.trend.MACD(data_df['close'])
        data_df['macd'] = macd.macd()
        data_df['macd_signal'] = macd.macd_signal()
        data_df['atr'] = ta.volatility.AverageTrueRange(data_df['high'], data_df['low'], data_df['close'], window=14).average_true_range()

        # Drop NaN from features
        feature_cols = ['rsi', 'macd', 'macd_signal', 'atr']
        data_df.dropna(subset=feature_cols, inplace=True)

        X = data_df[feature_cols].values
        y = (data_df['target'] + 1).values  # 0, 1, 2

        xgboost_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            eval_metric='mlogloss',
            n_estimators=100
        )
        xgboost_model.fit(X, y)

        # Save to Supabase
        model_bytes = pickle.dumps(xgboost_model)
        supabase.storage.from_('models').upload('xgboost_model.pkl', model_bytes, upsert=True)

        return {"message": "XGBoost model trained"}

    else:
        raise HTTPException(status_code=400, detail="Invalid model type")

@router.post("/predict")
async def predict_future(request: PredictRequest):
    global scaler, lstm_model, cnn_lstm_model, data_df

    if data_df is None:
        # Load data_df from Supabase
        try:
            data_response = supabase.storage.from_('models').download('data_df.pkl')
            data_df = pickle.loads(data_response)
        except Exception as e:
            raise HTTPException(status_code=400, detail="No data loaded. Train model first.")

        # Load scaler from Supabase
        try:
            scaler_response = supabase.storage.from_('models').download('scaler.pkl')
            scaler = pickle.loads(scaler_response)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Scaler not found. Train model first.")

    if request.model_type == "lstm" and lstm_model is None:
        # Download model from Supabase
        try:
            model_response = supabase.storage.from_('models').download('lstm_model.h5')
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
                temp_file.write(model_response)
                temp_path = temp_file.name
            lstm_model = load_model(temp_path)
            os.unlink(temp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail="LSTM model not found. Train model first.")

    if request.model_type == "cnn_lstm" and cnn_lstm_model is None:
        # Download model from Supabase
        try:
            model_response = supabase.storage.from_('models').download('cnn_lstm_model.h5')
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
                temp_file.write(model_response)
                temp_path = temp_file.name
            cnn_lstm_model = load_model(temp_path)
            os.unlink(temp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail="CNN-LSTM model not found. Train model first.")

    if request.model_type == "xgboost":
        if xgboost_model is None:
            # Download model from Supabase
            try:
                model_response = supabase.storage.from_('models').download('xgboost_model.pkl')
                xgboost_model = pickle.loads(model_response)
            except Exception as e:
                raise HTTPException(status_code=400, detail="XGBoost model not found. Train model first.")
        # Calculate features for the last data point
        data_df_copy = data_df.copy()
        data_df_copy['rsi'] = ta.momentum.RSIIndicator(data_df_copy['close'], window=14).rsi()
        macd = ta.trend.MACD(data_df_copy['close'])
        data_df_copy['macd'] = macd.macd()
        data_df_copy['macd_signal'] = macd.macd_signal()
        data_df_copy['atr'] = ta.volatility.AverageTrueRange(data_df_copy['high'], data_df_copy['low'], data_df_copy['close'], window=14).average_true_range()
        last_row = data_df_copy.tail(1)
        X_pred = last_row[['rsi', 'macd', 'macd_signal', 'atr']].values
        prediction = xgboost_model.predict(X_pred)[0] - 1  # -1, 0, 1
        signal = 'Buy' if prediction == 1 else 'Sell' if prediction == -1 else 'Hold'
        return {"signal": signal}

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
