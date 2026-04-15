from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

router = APIRouter(prefix="/api/twelve_data", tags=["twelve_data"])

class TwelveDataRequest(BaseModel):
    symbol: str = "GBP/USD"
    interval: str = "1h"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    api_key: str = "7bc7a66b670e4d2cbeca1cf9547b17d4"

@router.post("/fetch")
async def fetch_twelve_data(request: TwelveDataRequest):
    symbol = request.symbol.replace('/', ':')  # Twelve Data uses : for forex
    interval = request.interval
    api_key = request.api_key

    # Twelve Data API endpoint for time series
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={api_key}&outputsize=5000"

    if request.start_date:
        url += f"&start_date={request.start_date}"
    if request.end_date:
        url += f"&end_date={request.end_date}"

    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch data from Twelve Data")

    data = response.json()

    if 'values' not in data:
        raise HTTPException(status_code=400, detail="No data available")

    # Convert to DataFrame
    df = pd.DataFrame(data['values'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].rename(columns={'datetime': 'timestamp'})

    # Convert to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    return {"message": f"Fetched {len(df)} 1hr candles from Twelve Data", "data": df.to_dict('records')}
