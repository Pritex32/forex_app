from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
from oandapyV20 import API
from oandapyV20.endpoints.accounts import AccountDetails
from oandapyV20.endpoints.orders import OrderCreate
from oandapyV20.exceptions import V20Error
from app.routers.data_router import ACCESS_TOKEN, ACCOUNT_ID

router = APIRouter(prefix="/api/trading", tags=["trading"])

class OrderRequest(BaseModel):
    signal: str  # "Buy" or "Sell"
    current_price: float
    risk_pct: float = 1.0
    sl_pips: int = 30
    tp_pips: int = 90
    instrument: str = "GBP_USD"

def calculate_lot_size(balance, risk_percent, stop_loss_pips, pip_value_per_lot=10):
    risk_amount = balance * (risk_percent / 100)
    lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
    return round(lot_size, 2)

def calculate_sl_tp(entry_price, sl_pips, tp_pips, is_buy=True):
    pip = 0.0001
    if is_buy:
        sl = entry_price - sl_pips * pip
        tp = entry_price + tp_pips * pip
    else:
        sl = entry_price + sl_pips * pip
        tp = entry_price - tp_pips * pip
    return round(sl, 5), round(tp, 5)

def get_account_balance():
    client = API(access_token=ACCESS_TOKEN)
    r = AccountDetails(accountID=ACCOUNT_ID)
    try:
        response = client.request(r)
        balance = float(response['account']['balance'])
        return balance
    except V20Error as e:
        raise HTTPException(status_code=500, detail=f"Failed to get account balance: {e}")

@router.post("/place_order")
async def place_order(request: OrderRequest):
    client = API(access_token=ACCESS_TOKEN)

    is_buy = request.signal == "Buy"
    balance = get_account_balance()
    if balance == 0:
        raise HTTPException(status_code=400, detail="Unable to fetch account balance")

    lot_size = calculate_lot_size(balance, request.risk_pct, request.sl_pips)
    units = int(lot_size * 100000) if is_buy else int(-lot_size * 100000)

    sl_price, tp_price = calculate_sl_tp(request.current_price, request.sl_pips, request.tp_pips, is_buy)

    order = {
        "order": {
            "instrument": request.instrument,
            "units": str(units),
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": f"{sl_price}"},
            "takeProfitOnFill": {"price": f"{tp_price}"}
        }
    }

    r = OrderCreate(accountID=ACCOUNT_ID, data=order)
    try:
        response = client.request(r)
        return {"message": f"Trade executed: {request.signal} | Size: {lot_size} lots", "sl": sl_price, "tp": tp_price, "response": response}
    except V20Error as e:
        raise HTTPException(status_code=500, detail=f"Trade failed: {e}")
