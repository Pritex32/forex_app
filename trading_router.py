from fastapi import APIRouter
import pandas as pd
import ta
import plotly.graph_objects as go
import plotly.express as px
from app.routers.model_router import data_df

router = APIRouter(prefix="/api/indicators", tags=["indicators"])

@router.get("/charts/price_ma")
async def get_price_ma_chart():
    if data_df is None:
        return {"error": "No data loaded"}

    rollmean = data_df['open'].rolling(50).mean()
    fig = px.line(data_df, x=data_df.index, y='open', title='Open price yearly chart', labels={'open': 'Open Price'})
    fig.add_scatter(x=data_df.index, y=rollmean, mode='lines', name='50 MA')
    return fig.to_json()

@router.get("/charts/resampled")
async def get_resampled_charts():
    if data_df is None:
        return {"error": "No data loaded"}

    df = data_df.copy()
    df.index = pd.to_datetime(df.index)

    monthly_open = df['open'].resample('M').mean().dropna()
    weekly_open = df['open'].resample('W').mean().dropna()
    daily_open = df['open'].resample('D').mean().dropna()
    four_hour_open = df['open'].resample('4H').mean().dropna()
    hourly_open = df['open'].resample('1H').mean().dropna()

    colors = {'monthly': '#FF4136', 'weekly': '#2ECC40', 'daily': '#0074D9', '4h': '#FF851B', '1h': '#B10DC9'}

    figs = {}
    figs['monthly'] = px.line(x=monthly_open.index, y=monthly_open.values, title='Monthly Average Open Price').update_traces(line_color=colors['monthly']).to_json()
    figs['weekly'] = px.line(x=weekly_open.index, y=weekly_open.values, title='Weekly Average Open Price').update_traces(line_color=colors['weekly']).to_json()
    figs['daily'] = px.line(x=daily_open.index, y=daily_open.values, title='Daily Average Open Price').update_traces(line_color=colors['daily']).to_json()
    figs['4h'] = px.line(x=four_hour_open.index, y=four_hour_open.values, title='4-Hour Average Open Price').update_traces(line_color=colors['4h']).to_json()
    figs['1h'] = px.line(x=hourly_open.index, y=hourly_open.values, title='Hourly Average Open Price').update_traces(line_color=colors['1h']).to_json()

    return figs

@router.get("/indicators/values")
async def get_indicator_values():
    if data_df is None:
        return {"error": "No data loaded"}

    df = data_df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(close=df['open'], window=14).rsi()
    df['macd'] = ta.trend.MACD(close=df['open']).macd()
    df['macd_signal'] = ta.trend.MACD(close=df['open']).macd_signal()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['open'], window=14).average_true_range()

    rsi_val = round(df['rsi'].iloc[-1], 2)
    macd_val = round(df['macd'].iloc[-1], 4)
    macd_sig_val = round(df['macd_signal'].iloc[-1], 4)
    atr_val = round(df['atr'].iloc[-1], 4)
    atr_mean = round(df['atr'].rolling(50).mean().iloc[-1], 4)

    return {
        "rsi": rsi_val,
        "macd": macd_val,
        "macd_signal": macd_sig_val,
        "atr": atr_val,
        "atr_mean": atr_mean
    }

@router.get("/charts/rsi")
async def get_rsi_chart():
    if data_df is None:
        return {"error": "No data loaded"}

    df = data_df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(close=df['open'], window=14).rsi()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='brown')))
    fig.add_hline(y=70, line_color='orange', annotation_text="Overbought (70)")
    fig.add_hline(y=50, line_color='red', annotation_text="50% Line")
    fig.add_hline(y=30, line_color='green', annotation_text="Oversold (30)")
    fig.update_layout(title='Relative Strength Index (RSI) Indicator', yaxis=dict(range=[0, 100]))

    return fig.to_json()

@router.get("/charts/macd")
async def get_macd_chart():
    if data_df is None:
        return {"error": "No data loaded"}

    df = data_df.copy()
    df['macd'] = ta.trend.MACD(close=df['open']).macd()
    df['macd_signal'] = ta.trend.MACD(close=df['open']).macd_signal()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['macd'], mode='lines', name='MACD', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], mode='lines', name='Signal', line=dict(color='orange')))
    fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=0, y1=0, line=dict(color='black', dash='dot'))
    fig.update_layout(title='MACD - Moving Average Convergence Divergence')

    return fig.to_json()

@router.get("/charts/atr")
async def get_atr_chart():
    if data_df is None:
        return {"error": "No data loaded"}

    df = data_df.copy()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['open'], window=14).average_true_range()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['atr'], name='ATR', line=dict(color='green')))
    fig.update_layout(title='Average True Range (ATR) Indicator')

    return fig.to_json()

@router.get("/pivot/values")
async def get_pivot_values():
    if data_df is None:
        return {"error": "No data loaded"}

    df = data_df.copy()
    df['R4'] = df['close'] + (df['high'] - df['low']) * 1.1 / 2
    df['R3'] = df['close'] + (df['high'] - df['low']) * 1.1 / 4
    df['R2'] = df['close'] + (df['high'] - df['low']) * 1.1 / 6
    df['R1'] = df['close'] + (df['high'] - df['low']) * 1.1 / 12
    df['S1'] = df['close'] - (df['high'] - df['low']) * 1.1 / 12
    df['S2'] = df['close'] - (df['high'] - df['low']) * 1.1 / 6
    df['S3'] = df['close'] - (df['high'] - df['low']) * 1.1 / 4
    df['S4'] = df['close'] - (df['high'] - df['low']) * 1.1 / 2
    df['pivot'] = (df['R2'] + df['S2']) / 2

    latest = df.iloc[-1]
    return {
        "pivot": latest['pivot'],
        "r1": latest['R1'],
        "r2": latest['R2'],
        "r3": latest['R3'],
        "r4": latest['R4'],
        "s1": latest['S1'],
        "s2": latest['S2'],
        "s3": latest['S3'],
        "s4": latest['S4']
    }
