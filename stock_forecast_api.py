# stock_forecast_api.py
# Refactored backend that follows the reference notebook workflow (ARIMA + LSTM + Meta NN)
# Keeps Angel One integration as primary live data source, falls back to yfinance.

import os
import sys
import importlib
import warnings
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import pyotp

from dotenv import load_dotenv

# Try to detect optional modules and warn if missing (SmartApi may be optional)
_required = ['flask', 'flask_cors', 'tensorflow', 'statsmodels', 'SmartApi']
_missing = []
for _m in _required:
    try:
        importlib.import_module(_m)
    except Exception:
        _missing.append(_m)
if _missing:
    print(f"Warning: optional modules might be missing: {_missing}")

# Load environment variables
load_dotenv()

# ML and stats imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# SmartAPI (Angel One)
from SmartApi import SmartConnect

# Flask
from flask import Flask, request, jsonify
from flask_cors import CORS

# -------------------------
# Globals & reproducibility
warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

app = Flask(__name__, static_folder='frontend', static_url_path='/')
CORS(app)

# Load Angel One credentials from env (do NOT print these)
ANGEL_API_KEY = os.getenv('ANGEL_API_KEY')
ANGEL_CLIENT_ID = os.getenv('ANGEL_CLIENT_ID')
ANGEL_PASSWORD = os.getenv('ANGEL_PASSWORD')
ANGEL_TOTP_SECRET = os.getenv('ANGEL_TOTP_SECRET')  # optional: if present, will auto-generate TOTPs

if not (ANGEL_API_KEY and ANGEL_CLIENT_ID and ANGEL_PASSWORD):
    print("Warning: Angel One credentials are not fully configured in environment.")
else:
    if ANGEL_TOTP_SECRET:
        print("Angel One TOTP secret detected; will auto-generate codes when available.")

# Small token mapping used for live quotes (expand as needed)
STOCK_TOKENS = {
    "RELIANCE-EQ": "2885",
    "TCS-EQ": "11536",
    "INFY-EQ": "1594",
    "HDFCBANK-EQ": "1333",
    "ICICIBANK-EQ": "4963",
    "SBIN-EQ": "3045",
    "TATAMOTORS-EQ": "3456",
    "WIPRO-EQ": "3787",
    "ITC-EQ": "1660",
    "BHARTIARTL-EQ": "10604"
}

# -------------------------
# Utility functions
def rmse(a, b):
    return math.sqrt(mean_squared_error(a, b))

def mape(actual, pred):
    # avoid division by zero; add small epsilon
    actual = np.array(actual)
    pred = np.array(pred)
    eps = 1e-8
    return np.mean(np.abs((actual - pred) / (actual + eps))) * 100.0

def accuracy_from_mape(actual, pred):
    return max(0.0, 100.0 - mape(actual, pred))

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

# -------------------------
# Angel One helpers
def connect_angelone(totp_code):
    try:
        obj = SmartConnect(api_key=ANGEL_API_KEY)
        data = obj.generateSession(ANGEL_CLIENT_ID, ANGEL_PASSWORD, totp_code)
        if data and data.get('status'):
            return obj, None
        return None, data.get('message') if isinstance(data, dict) else "unknown error"
    except Exception as e:
        return None, str(e)

def get_angelone_data(obj, symbol, from_date=None, to_date=None, exchange="NSE", interval="ONE_DAY"):
    """
    Returns: DataFrame with timestamp index and numeric OHLCV columns or None
    """
    token = STOCK_TOKENS.get(symbol)
    if not token:
        return None
    if to_date is None:
        to_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    if from_date is None:
        from_date = (datetime.now() - timedelta(days=365 * 15)).strftime("%Y-%m-%d 09:15")
    param = {
        "exchange": exchange,
        "symboltoken": token,
        "interval": interval,
        "fromdate": from_date,
        "todate": to_date
    }
    try:
        hist = obj.getCandleData(param)
        if hist and hist.get('status') and 'data' in hist:
            df = pd.DataFrame(hist['data'], columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            # numeric conversion
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()
            return df
    except Exception as e:
        print("get_angelone_data error:", e)
    return None

def get_live_quote(obj, symbol, exchange="NSE"):
    try:
        token = STOCK_TOKENS.get(symbol)
        if not token:
            return None
        q = obj.getMarketData({"mode": "FULL", "exchangeTokens": {exchange: [token]}})
        if q and q.get('status') and 'data' in q:
            d = q['data']['fetched'][0]
            return {
                'symbol': d.get('tradingSymbol'),
                'ltp': safe_float(d.get('ltp')),
                'open': safe_float(d.get('open')),
                'high': safe_float(d.get('high')),
                'low': safe_float(d.get('low')),
                'close': safe_float(d.get('close')),
                'volume': safe_float(d.get('volume')),
                'change': safe_float(d.get('netChange')),
                'change_pct': safe_float(d.get('percentChange'))
            }
    except Exception:
        pass
    return None

# -------------------------
# Model helper functions (aligned with the reference notebook)
def build_sequences(features_scaled, target_scaled, time_step):
    X, y = [], []
    n = len(features_scaled)
    for i in range(time_step, n):
        X.append(features_scaled[i-time_step:i, :])
        y.append(target_scaled[i, 0])
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    return X, y

def train_arima_on_series(series, steps, order=(5,1,0)):
    try:
        model = ARIMA(series, order=order).fit()
        forecast = model.forecast(steps=steps)
        return model, np.asarray(forecast).flatten()
    except Exception as e:
        print("[ARIMA] error:", e)
        return None, None

def train_lstm_model(X_train, y_train, lstm_units=64, lr=1e-3, epochs=100, batch_size=32, val_split=0.1):
    model = Sequential([
        LSTM(lstm_units, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=lr, clipnorm=1.0))
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0)
    model.fit(X_train, y_train, validation_split=val_split, epochs=epochs, batch_size=batch_size,
              shuffle=False, callbacks=[es], verbose=0)
    return model

def recursive_lstm_forecast(lstm_model, last_sequence, steps, target_scaler):
    cur_input = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1]).astype(np.float32)
    preds_scaled = []
    for _ in range(steps):
        pred_scaled = float(lstm_model.predict(cur_input, verbose=0)[0, 0])
        preds_scaled.append(pred_scaled)
        last_vol = cur_input[0, -1, 1]
        new_step = np.array([[pred_scaled, last_vol]])
        cur_input = np.concatenate([cur_input[:, 1:, :], new_step.reshape(1, 1, -1)], axis=1)
    preds = target_scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return preds

def train_meta_nn(meta_X_train, meta_y_train, meta_X_val, meta_y_val, lr=1e-3, epochs=200, batch_size=16):
    meta_model = Sequential([
        Dense(32, activation='relu', input_shape=(meta_X_train.shape[1],)),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    meta_model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    meta_model.fit(meta_X_train, meta_y_train, validation_data=(meta_X_val, meta_y_val),
                   epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)
    return meta_model

# -------------------------
# Main pipeline (runs synchronously inside the request)
def train_and_forecast(df, days=7, n_test=100, time_step=60, lstm_units=64, arima_order=(5,1,0)):
    """
    Input:
      df: DataFrame with 'Close' and 'Volume' columns indexed by datetime
      days: forecast horizon
    Returns:
      dict with 'meta', 'arima', 'lstm' objects containing predictions, metrics, historical snippet
    """
    # Basic checks
    if 'Close' not in df.columns or 'Volume' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' and 'Volume' columns")
    if len(df) <= (time_step + 10):
        raise ValueError("Not enough data for the configured time_step")

    # 1) Train/test split (reserve n_test rows for meta training)
    train_size = len(df) - n_test
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    # 2) Scalers fit on train only (no leakage)
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(train_df[['Close', 'Volume']])
    target_scaler.fit(train_df[['Close']].values.reshape(-1, 1))

    # 3) Transform full dataset
    features_scaled = feature_scaler.transform(df[['Close', 'Volume']])
    target_scaled = target_scaler.transform(df[['Close']].values.reshape(-1, 1))

    # 4) Build sequences
    X_all, y_all = build_sequences(features_scaled, target_scaled, time_step)
    assert X_all.shape[0] == len(df) - time_step

    if n_test >= X_all.shape[0]:
        raise ValueError("n_test is too large for given data/time_step")

    X_train = X_all[:-n_test]
    X_test = X_all[-n_test:]
    y_train = y_all[:-n_test]
    y_test = y_all[-n_test:]

    # ---------------- ARIMA on train -> forecast for test window
    arima_model, arima_test_forecast = train_arima_on_series(train_df['Close'], steps=len(test_df), order=arima_order)
    if arima_test_forecast is None:
        arima_test_forecast = np.array([train_df['Close'].iloc[-1]] * len(test_df))
        arima_model = None
    arima_test_series = pd.Series(arima_test_forecast, index=test_df.index)

    # ---------------- LSTM train -> predict test window
    lstm_model = train_lstm_model(X_train, y_train, lstm_units=lstm_units)
    lstm_test_pred_scaled = lstm_model.predict(X_test, verbose=0)
    lstm_test_pred = target_scaler.inverse_transform(lstm_test_pred_scaled).flatten()
    lstm_test_series = pd.Series(lstm_test_pred, index=test_df.index)

    # ---------------- Evaluation metrics on test
    y_test_prices = test_df['Close'].values
    valid_arima_idx = ~np.isnan(arima_test_series.values)
    if len(y_test_prices[valid_arima_idx]) == 0:
        rmse_arima = float('nan'); mae_arima = float('nan'); r2_arima = float('nan'); acc_arima = float('nan')
    else:
        rmse_arima = rmse(y_test_prices[valid_arima_idx], arima_test_series.values[valid_arima_idx])
        mae_arima = mean_absolute_error(y_test_prices[valid_arima_idx], arima_test_series.values[valid_arima_idx])
        r2_arima  = r2_score(y_test_prices[valid_arima_idx], arima_test_series.values[valid_arima_idx])
        acc_arima = accuracy_from_mape(y_test_prices[valid_arima_idx], arima_test_series.values[valid_arima_idx])

    rmse_lstm = rmse(y_test_prices, lstm_test_series.values)
    mae_lstm = mean_absolute_error(y_test_prices, lstm_test_series.values)
    r2_lstm = r2_score(y_test_prices, lstm_test_series.values)
    acc_lstm = accuracy_from_mape(y_test_prices, lstm_test_series.values)

    metrics = {
        'arima': {'rmse': rmse_arima, 'mae': mae_arima, 'r2': r2_arima, 'accuracy_pct': acc_arima},
        'lstm': {'rmse': rmse_lstm, 'mae': mae_lstm, 'r2': r2_lstm, 'accuracy_pct': acc_lstm}
    }

    # ---------------- Meta learner (train on reserved test-set)
    meta_X = np.column_stack((arima_test_series.values, lstm_test_series.values))
    meta_y = test_df['Close'].values
    # small train/val split for meta model
    meta_X_train, meta_X_val, meta_y_train, meta_y_val = train_test_split(meta_X, meta_y, test_size=0.2, random_state=SEED)
    meta_model = train_meta_nn(meta_X_train, meta_y_train, meta_X_val, meta_y_val)
    meta_val_pred = meta_model.predict(meta_X_val).flatten()
    rmse_meta_val = rmse(meta_y_val, meta_val_pred)
    mae_meta_val = mean_absolute_error(meta_y_val, meta_val_pred)

    meta_test_pred = meta_model.predict(meta_X).flatten()
    rmse_meta = rmse(meta_y, meta_test_pred)
    mae_meta = mean_absolute_error(meta_y, meta_test_pred)
    r2_meta = r2_score(meta_y, meta_test_pred)
    acc_meta = accuracy_from_mape(meta_y, meta_test_pred)

    metrics['meta'] = {'rmse': rmse_meta, 'mae': mae_meta, 'r2': r2_meta, 'accuracy_pct': acc_meta}

    # ---------------- FINAL FORECAST: retrain on full dataset & forecast `days`
    # ARIMA on full
    arima_full_model, future_arima = train_arima_on_series(df['Close'], steps=days, order=arima_order)
    if future_arima is None:
        future_arima = np.array([df['Close'].iloc[-1]] * days)

    # LSTM retrain on full
    feature_scaler_full = MinMaxScaler(); target_scaler_full = MinMaxScaler()
    feature_scaler_full.fit(df[['Close', 'Volume']])
    target_scaler_full.fit(df[['Close']].values.reshape(-1, 1))
    features_scaled_full = feature_scaler_full.transform(df[['Close', 'Volume']])
    target_scaled_full = target_scaler_full.transform(df[['Close']].values.reshape(-1, 1))
    X_full, y_full = build_sequences(features_scaled_full, target_scaled_full, time_step)
    lstm_full = train_lstm_model(X_full, y_full, lstm_units=lstm_units, lr=1e-3, epochs=50, batch_size=32, val_split=0.0)
    last_seq = features_scaled_full[-time_step:]
    future_lstm = recursive_lstm_forecast(lstm_full, last_seq, steps=days, target_scaler=target_scaler_full)

    # Meta ensemble for future
    meta_input_future = np.column_stack((future_arima, future_lstm))
    meta_final_future = meta_model.predict(meta_input_future).flatten()

    # Build business-day dates
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=days + 10, freq='B')[:days]

    # Historical snippet (last 60 days)
    historical_data = df.tail(60)

    def format_predictions(dates, arr):
        return [{'date': d.strftime('%Y-%m-%d'), 'value': float(v)} for d, v in zip(dates, arr)]

    return {
        'meta': {
            'predictions': format_predictions(future_dates, meta_final_future),
            'metrics': metrics['meta'],
            'historical': [{'date': i.strftime('%Y-%m-%d'), 'value': float(v)} for i, v in historical_data['Close'].items()]
        },
        'arima': {
            'predictions': format_predictions(future_dates, future_arima),
            'metrics': metrics['arima'],
            'historical': [{'date': i.strftime('%Y-%m-%d'), 'value': float(v)} for i, v in historical_data['Close'].items()]
        },
        'lstm': {
            'predictions': format_predictions(future_dates, future_lstm),
            'metrics': metrics['lstm'],
            'historical': [{'date': i.strftime('%Y-%m-%d'), 'value': float(v)} for i, v in historical_data['Close'].items()]
        }
    }

# -------------------------
# Flask endpoints (mirrors original)
@app.route('/api/forecast', methods=['POST'])
def forecast():
    print("=" * 80)
    print(" REAL FORECAST API CALLED")
    print("=" * 80)
    try:
        data = request.get_json(force=True, silent=True) or {}
        print("Request:", {k: v for k, v in data.items() if k != 'password'})

        # parse request
        totp = data.get('totp')
        symbol = data.get('symbol', 'RELIANCE-EQ')
        days = int(data.get('forecast_days', 7) or 7)

        df = None
        source = 'yfinance'
        live = None

        # Try Angel One if creds exist
        can_try_angel = all([ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD])

        # Auto-generate TOTP if secret present and totp not supplied
        if can_try_angel and not totp and ANGEL_TOTP_SECRET:
            try:
                totp = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
                print("Auto-generated TOTP using ANGEL_TOTP_SECRET")
            except Exception as e:
                print("TOTP generation error:", e)

        if can_try_angel and totp:
            obj, err = connect_angelone(totp)
            if obj:
                # request historical in Angel One format
                from_date = (datetime.now() - timedelta(days=365 * 15)).strftime("%Y-%m-%d 09:15")
                to_date = datetime.now().strftime("%Y-%m-%d %H:%M")
                df_a = get_angelone_data(obj, symbol, from_date=from_date, to_date=to_date)
                if df_a is not None and len(df_a) > 200:
                    df = df_a[['Close', 'Volume']].dropna()
                    source = 'angel_one'
                    # try to fetch live quote
                    live = get_live_quote(obj, symbol)
                else:
                    print("Angel One returned insufficient historical data; falling back to yfinance.")
            else:
                print("Angel One connection failed:", err)

        # fallback: yfinance
        if df is None or len(df) < 200:
            sym_yf = symbol.replace('-EQ', '.NS')
            print(f"Downloading {sym_yf} from yfinance...")
            df = yf.download(sym_yf, start='2010-01-01', progress=False)
            if df is None or df.empty:
                return jsonify({'success': False, 'error': 'Failed to download data from yfinance'}), 500
            df = df[['Close', 'Volume']].dropna()

        if len(df) < 200:
            return jsonify({'success': False, 'error': 'Insufficient historical data (need >200 rows)'}), 400

        # Run the heavy pipeline (synchronous)
        results = train_and_forecast(df.copy(), days=days)

        historical_df = df.tail(60)
        historical_data = {
            'dates': historical_df.index.strftime('%Y-%m-%d').tolist(),
            'prices': [float(x) for x in historical_df['Close'].tolist()]
        }

        response = {
            'success': True,
            'data_source': source,
            'live_quote': live,
            'results': results,
            'latest_close': float(df['Close'].iloc[-1]),
            'data_range': {
                'start': df.index[0].strftime('%Y-%m-%d'),
                'end': df.index[-1].strftime('%Y-%m-%d'),
                'records': len(df)
            },
            'historical': historical_data
        }
        return jsonify(response)
    except Exception as e:
        print("Forecast API error:", e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stocks', methods=['GET'])
def get_stocks():
    stocks = [
        {'symbol': 'RELIANCE-EQ', 'name': 'Reliance Industries'},
        {'symbol': 'TCS-EQ', 'name': 'Tata Consultancy Services'},
        {'symbol': 'INFY-EQ', 'name': 'Infosys'},
        {'symbol': 'HDFCBANK-EQ', 'name': 'HDFC Bank'},
        {'symbol': 'ICICIBANK-EQ', 'name': 'ICICI Bank'},
        {'symbol': 'SBIN-EQ', 'name': 'State Bank of India'},
        {'symbol': 'TATAMOTORS-EQ', 'name': 'Tata Motors'},
        {'symbol': 'WIPRO-EQ', 'name': 'Wipro'},
        {'symbol': 'ITC-EQ', 'name': 'ITC'},
        {'symbol': 'BHARTIARTL-EQ', 'name': 'Bharti Airtel'}
    ]
    return jsonify({'success': True, 'stocks': stocks})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/api/forecast-mock', methods=['POST'])
def forecast_mock():
    try:
        data = request.get_json(force=True, silent=True) or {}
        symbol = data.get('symbol', 'RELIANCE-EQ')
        days = int(data.get('forecast_days', 7) or 7)

        last_close = 1000.0
        arima_f = [round(last_close + i * 1.5, 2) for i in range(1, days + 1)]
        lstm_f = [round(last_close + i * 1.2, 2) for i in range(1, days + 1)]
        meta_f = [round((a + l) / 2.0, 2) for a, l in zip(arima_f, lstm_f)]
        dates = pd.date_range(datetime.now().date() + timedelta(1), periods=days, freq='B')

        return jsonify({
            'success': True,
            'data_source': 'mock',
            'live_quote': None,
            'latest_close': last_close,
            'data_range': {'start': '2010-01-01', 'end': datetime.now().strftime('%Y-%m-%d'), 'records': 1000},
            'results': {
                'metrics': {
                    'arima': {'rmse': 0.0, 'mae': 0.0, 'r2': 1.0, 'accuracy_pct': 100.0},
                    'lstm': {'rmse': 0.0, 'mae': 0.0, 'r2': 1.0, 'accuracy_pct': 100.0},
                    'meta': {'rmse': 0.0, 'mae': 0.0, 'r2': 1.0, 'accuracy_pct': 100.0}
                },
                'forecast': {
                    'dates': [d.strftime('%Y-%m-%d') for d in dates],
                    'arima': arima_f,
                    'lstm': lstm_f,
                    'meta': meta_f
                }
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# -------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
