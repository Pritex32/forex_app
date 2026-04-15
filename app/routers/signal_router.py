from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import ta
from app.routers.model_router import scaler, data_df, lstm_model, cnn_lstm_model, xgboost_model

router = APIRouter(prefix="/api/signals", tags=["signals"])

class SignalRequest(BaseModel):
    model_type: str  # "lstm", "cnn_lstm", or "xgboost"
    risk_pct: float = 1.0
    sl_pips: int = 30
    tp_pips: int = 90

def predict_future_price(df, model, scaler):
    closes = df['open'].values[-60:]
    scaled = scaler.transform(closes.reshape(-1, 1)).reshape(1, 60, 1)
    prediction = model.predict(scaled)
    predicted_price = scaler.inverse_transform(prediction)[0, 0]
    return predicted_price

def add_indicators(df):
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['open'], window=14).average_true_range()
    df['ma50'] = df['open'].rolling(window=50).mean()
    df['rsi'] = ta.momentum.RSIIndicator(close=df['open'], window=14).rsi()
    return df

def generate_signal(predicted_price, current_price, volatility, threshold=0.002):
    diff = predicted_price - current_price
    dynamic_threshold = threshold * volatility
    if diff > dynamic_threshold:
        return "Buy"
    elif diff < -dynamic_threshold:
        return "Sell"
    else:
        return "Hold"

def generate_signal_enhanced(predicted_price, current_price, atr, ma50, rsi, threshold=0.002):
    diff = predicted_price - current_price
    dynamic_threshold = threshold * current_price + atr
    trend_up = bool(current_price > ma50)
    momentum_up = bool(rsi > 50)
    if diff > dynamic_threshold and trend_up and momentum_up:
        return "Buy"
    elif diff < -dynamic_threshold and not trend_up and rsi < 50:
        return "Sell"
    else:
        return "Hold"

def calculate_camarilla_pivot_points(df):
    df['R4'] = df['close'] + (df['high'] - df['low']) * 1.1 / 2
    df['R3'] = df['close'] + (df['high'] - df['low']) * 1.1 / 4
    df['R2'] = df['close'] + (df['high'] - df['low']) * 1.1 / 6
    df['R1'] = df['close'] + (df['high'] - df['low']) * 1.1 / 12
    df['S1'] = df['close'] - (df['high'] - df['low']) * 1.1 / 12
    df['S2'] = df['close'] - (df['high'] - df['low']) * 1.1 / 6
    df['S3'] = df['close'] - (df['high'] - df['low']) * 1.1 / 4
    df['S4'] = df['close'] - (df['high'] - df['low']) * 1.1 / 2
    df['pivot'] = (df['R2'] + df['S2']) / 2
    return df

@router.post("/generate")
async def generate_signals(request: SignalRequest):
    global data_df, scaler, lstm_model, cnn_lstm_model, xgboost_model

    if data_df is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    df = add_indicators(data_df.copy())
    current_price = df['open'].values[-1]

    df_pivot = calculate_camarilla_pivot_points(df.copy())
    pivot = df_pivot['pivot'].iloc[-1]
    r2 = df_pivot['R2'].iloc[-1]
    s2 = df_pivot['S2'].iloc[-1]

    swing_high = df['high'].rolling(window=20).max().iloc[-1]
    swing_low = df['low'].rolling(window=20).min().iloc[-1]

    volatility = df['open'].rolling(window=14).std().iloc[-1]
    atr = df['atr'].iloc[-1]
    ma50 = df['ma50'].iloc[-1]
    rsi = df['rsi'].iloc[-1]

    if request.model_type in ["lstm", "cnn_lstm"]:
        if request.model_type == "lstm" and lstm_model is None:
            raise HTTPException(status_code=400, detail="LSTM model not loaded")
        elif request.model_type == "cnn_lstm" and cnn_lstm_model is None:
            raise HTTPException(status_code=400, detail="CNN-LSTM model not loaded")

        model = lstm_model if request.model_type == "lstm" else cnn_lstm_model
        predicted_price = predict_future_price(df, model, scaler)

        signal = generate_signal(predicted_price, current_price, volatility)
        atr_signal = generate_signal_enhanced(predicted_price, current_price, atr, ma50, rsi)

        if predicted_price > r2:
            pivot_signal = "BUY"
        elif predicted_price < s2:
            pivot_signal = "SELL"
        else:
            pivot_signal = "HOLD"

        bullish_confluence = (
            predicted_price > swing_high and
            predicted_price > r2 and
            rsi < 70 and
            predicted_price > pivot
        )

        bearish_confluence = (
            predicted_price < swing_low and
            predicted_price < s2 and
            rsi > 30 and
            predicted_price < pivot
        )

        if bullish_confluence:
            smart_signal = "Strong Buy"
        elif bearish_confluence:
            smart_signal = "Strong Sell"
        else:
            smart_signal = "No strong signal"

    elif request.model_type == "xgboost":
        if xgboost_model is None:
            raise HTTPException(status_code=400, detail="XGBoost model not loaded")

        # Calculate signal directly
        data_df_copy = data_df.copy()
        data_df_copy['rsi'] = ta.momentum.RSIIndicator(data_df_copy['close'], window=14).rsi()
        macd = ta.trend.MACD(data_df_copy['close'])
        data_df_copy['macd'] = macd.macd()
        data_df_copy['macd_signal'] = macd.macd_signal()
        data_df_copy['atr'] = ta.volatility.AverageTrueRange(data_df_copy['high'], data_df_copy['low'], data_df_copy['close'], window=14).average_true_range()
        last_row = data_df_copy.tail(1)
        X_pred = last_row[['rsi', 'macd', 'macd_signal', 'atr']].values
        prediction = xgboost_model.predict(X_pred)[0] - 1
        signal = 'Buy' if prediction == 1 else 'Sell' if prediction == -1 else 'Hold'
        predicted_price = None

        atr_signal = signal
        pivot_signal = signal
        smart_signal = signal

    else:
        raise HTTPException(status_code=400, detail="Invalid model type")

    return {
        "current_price": current_price,
        "predicted_price": predicted_price,
        "signal": signal,
        "atr_signal": atr_signal,
        "pivot_signal": pivot_signal,
        "smart_money_signal": smart_signal,
        "risk_pct": request.risk_pct,
        "sl_pips": request.sl_pips,
        "tp_pips": request.tp_pips
    }
