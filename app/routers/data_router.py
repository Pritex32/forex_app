from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from datetime import datetime, timedelta
import requests
import time
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.exceptions import V20Error
import os
import app.routers.model_router as model_router

ACCESS_TOKEN = 'd917178f8075576c341cbe85848de18e-9575706fb366ffd63dbbd057ecc8d847'
ACCOUNT_ID = '101-004-31663011-001'
TWELVE_DATA_API_KEY = '7bc7a66b670e4d2cbeca1cf9547b17d4'

router = APIRouter(prefix="/api/data", tags=["data"])

class DataFetchRequest(BaseModel):
    instrument: str = "GBP_USD"
    granularity: str = "D"
    start_date: str = "2024-01-01"
    access_token: str = ACCESS_TOKEN
    account_id: str = ACCOUNT_ID
    cache_file: Optional[str] = None
    sleep_time: int = 2
    max_retries: int = 5

@router.post("/fetch")
async def fetch_oanda_data(request: DataFetchRequest):
    try:
        instrument = request.instrument
        granularity = request.granularity
        start_date = request.start_date
        access_token = request.access_token
        account_id = request.account_id
        cache_file = request.cache_file or f'{instrument.lower()}_2021_present.csv'
        sleep_time = request.sleep_time
        max_retries = request.max_retries

        intraday_granularities = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4']

        if granularity in intraday_granularities:
            # ----------- (UNCHANGED) -----------
            ...
        else:
            client = API(access_token=access_token)
            end_dt = pd.Timestamp.utcnow()

            if os.path.exists(cache_file):
                cached_df = pd.read_csv(cache_file, parse_dates=['timestamp'])
                cached_df['timestamp'] = pd.to_datetime(cached_df['timestamp'], utc=True)
                start_dt = cached_df['timestamp'].max() + timedelta(seconds=1)
                all_data = cached_df.to_dict('records')
            else:
                start_dt = pd.to_datetime(start_date, utc=True)
                all_data = []

            prev_last_time = None

            while start_dt < end_dt:
                window_end = min(start_dt + timedelta(days=490), end_dt)

                params = {
                    "granularity": granularity,
                    "from": start_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    "to": window_end.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    "price": "M"
                }

                r = InstrumentsCandles(instrument=instrument, params=params)

                # ✅ FIX: retry loop must be INSIDE main loop
                retries = 0
                while retries <= max_retries:
                    try:
                        response = client.request(r)
                        break
                    except (V20Error, requests.exceptions.RequestException):
                        retries += 1
                        wait_time = sleep_time * retries
                        if retries > max_retries:
                            raise HTTPException(status_code=500, detail="Max retries reached.")
                        time.sleep(wait_time)

                candles = response['candles']

                if not candles:
                    break  # ✅ now valid (inside loop)

                for c in candles:
                    all_data.append({
                        'timestamp': c['time'],
                        'open': float(c['mid']['o']),
                        'high': float(c['mid']['h']),
                        'low': float(c['mid']['l']),
                        'close': float(c['mid']['c']),
                        'volume': int(c['volume'])
                    })

                last_time = max(pd.to_datetime(c['time'], utc=True) for c in candles)

                if prev_last_time is not None and last_time <= prev_last_time:
                    start_dt = prev_last_time + timedelta(seconds=1)
                    continue

                prev_last_time = last_time
                start_dt = last_time + timedelta(seconds=1)

                time.sleep(sleep_time)

        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

        df.to_csv(cache_file, index=False)

        model_router.data_df = df

        return {
            "message": f"Saved {len(df)} candles to cache file: {cache_file}",
            "data": df.to_dict('records')
        }

    except Exception as e:
        return {"error": str(e)}
