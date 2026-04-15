from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

router = APIRouter(prefix="/api/support_resistance", tags=["support_resistance"])

class SupportResistanceRequest(BaseModel):
    data: List[Dict[str, Any]]  # List of OHLCV data
    window: int = 10

@router.post("/detect")
async def detect_support_resistance(request: SupportResistanceRequest):
    df = pd.DataFrame(request.data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    recent_df = df.tail(200)
    current_price = recent_df['close'].iloc[-1]

    # Detect zones
    def detect_zones(df, window=25):
        support, resistance = [], []
        for i in range(window, len(df) - window):
            low = df['low'].iloc[i]
            high = df['high'].iloc[i]
            if all(low < df['low'].iloc[i - window:i]) and all(low < df['low'].iloc[i + 1:i + window + 1]):
                support.append((df.index[i], low))
            if all(high > df['high'].iloc[i - window:i]) and all(high > df['high'].iloc[i + 1:i + window + 1]):
                resistance.append((df.index[i], high))
        return support, resistance

    # Classify levels
    def classify_levels(levels, threshold=0.005):
        major, minor = [], []
        scores = {}
        for _, level in levels:
            strength = sum(abs(level - other[1]) < level * threshold for other in levels)
            if all(abs(level - existing) > level * threshold for existing in major):
                major.append(level)
                scores[level] = strength
            else:
                minor.append(level)
                scores[level] = strength
        return major, minor, scores

    # Detect order blocks
    def detect_order_blocks(df, threshold=0.005, lookback=5):
        blocks = []
        avg_volume = df['volume'].mean()
        for i in range(lookback, len(df) - lookback):
            block_range = df['high'].iloc[i - lookback:i].max() - df['low'].iloc[i - lookback:i].min()
            if block_range < df['close'].iloc[i] * threshold:
                if df['volume'].iloc[i] > avg_volume:
                    start_time = df.index[i - lookback]
                    end_time = df.index[i]
                    low = df['low'].iloc[i - lookback:i + 1].min()
                    high = df['high'].iloc[i - lookback:i + 1].max()
                    block_type = 'bullish' if df['close'].iloc[i + 1] > high else 'bearish'
                    blocks.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'low': low,
                        'high': high,
                        'type': block_type
                    })
        return blocks

    support_raw, resistance_raw = detect_zones(recent_df, window=request.window)
    support_major, support_minor, support_strength = classify_levels(support_raw)
    resistance_major, resistance_minor, resistance_strength = classify_levels(resistance_raw)
    order_blocks = detect_order_blocks(recent_df)

    # Alerts
    alert_msgs = []
    def check_proximity(price, level, margin=0.01):
        return abs(price - level) / price < margin

    for level in support_major:
        if check_proximity(current_price, level):
            alert_msgs.append(f"Price is near strong support ({level:.2f})")

    for level in resistance_major:
        if check_proximity(current_price, level):
            alert_msgs.append(f"Price is near strong resistance ({level:.2f})")

    return {
        "support_major": support_major,
        "resistance_major": resistance_major,
        "support_minor": support_minor,
        "resistance_minor": resistance_minor,
        "order_blocks": order_blocks,
        "alerts": alert_msgs,
        "current_price": current_price
    }
