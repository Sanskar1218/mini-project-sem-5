# stock_forecast_api.py - Secure Backend API
import os
import sys
import importlib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import pyotp

# Startup diagnostic: attempt to import common optional dependencies and print any missing ones
_required = ['flask', 'flask_cors', 'yfinance', 'tensorflow', 'SmartApi', 'logzero', 'websocket']
_missing = []
for _mod in _required:
    try:
        importlib.import_module(_mod)
    except Exception:
        _missing.append(_mod)
if _missing:
    print(f"Warning: the following optional modules are missing and may cause import errors: {_missing}")
# load .env file into environment (if present)
load_dotenv()
import yfinance as yf
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.arima.model import ARIMA
from SmartApi import SmartConnect
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import warnings
import tensorflow as tf
import math

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

CORS = CORS
# Serve the static frontend from the 'frontend' folder so the app can run as a single web process
app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)
CORS(app)

# Load from environment variables
ANGEL_API_KEY = os.getenv('ANGEL_API_KEY')
ANGEL_CLIENT_ID = os.getenv('ANGEL_CLIENT_ID')
ANGEL_PASSWORD = os.getenv('ANGEL_PASSWORD')
ANGEL_TOTP_SECRET = os.getenv('ANGEL_TOTP_SECRET')

# Warn if Angel One env vars are not configured (do not print values)
if not (ANGEL_API_KEY and ANGEL_CLIENT_ID and ANGEL_PASSWORD):
    print("Warning: Angel One credentials (ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD) are not set in environment.\nIf you intend to use Angel One, create a .env file or set these environment variables.")
else:
    if ANGEL_TOTP_SECRET:
        print("Angel One TOTP secret is configured (will generate codes automatically).")

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

def connect_angelone(totp):
    try:
        obj = SmartConnect(api_key=ANGEL_API_KEY)
        data = obj.generateSession(ANGEL_CLIENT_ID, ANGEL_PASSWORD, totp)
        if data['status']:
            return obj, None
        return None, data.get('message')
    except Exception as e:
        return None, str(e)

def get_angelone_data(obj, symbol, from_date, to_date):
    token = STOCK_TOKENS.get(symbol)
    if not token:
        return None
    
    param = {
        "exchange": "NSE",
        "symboltoken": token,
        "interval": "ONE_DAY",
        "fromdate": from_date,
        "todate": to_date
    }
    
    hist = obj.getCandleData(param)
    if hist['status']:
        df = pd.DataFrame(hist['data'], columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    return None

def train_and_forecast(df, days=7):
    n_test, time_step = 100, 60
    train_df = df.iloc[:-n_test]
    test_df = df.iloc[-n_test:]
    
    f_scaler, t_scaler = MinMaxScaler(), MinMaxScaler()
    f_scaler.fit(train_df[['Close', 'Volume']])
    t_scaler.fit(train_df[['Close']])
    
    features = f_scaler.transform(df[['Close', 'Volume']])
    target = t_scaler.transform(df[['Close']].values.reshape(-1,1))
    
    X, y = [], []
    for i in range(time_step, len(df)):
        X.append(features[i-time_step:i])
        y.append(target[i, 0])
    X, y = np.array(X), np.array(y).reshape(-1, 1)
    
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]
    
    # ARIMA
    try:
        arima = ARIMA(train_df['Close'], order=(5,1,0)).fit()
        arima_pred = pd.Series(arima.forecast(n_test).values, index=test_df.index)
    except:
        arima_pred = pd.Series([train_df['Close'].iloc[-1]]*n_test, index=test_df.index)
    
    # LSTM
    lstm = Sequential([
        LSTM(64, input_shape=(time_step, 2)),
        Dropout(0.2),
        Dense(1)
    ])
    lstm.compile(loss='mse', optimizer=Adam(0.001, clipnorm=1.0))
    lstm.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=32,
             callbacks=[EarlyStopping('val_loss', 8, restore_best_weights=True)], verbose=0)
    
    lstm_pred = t_scaler.inverse_transform(lstm.predict(X_test, verbose=0)).flatten()
    lstm_pred = pd.Series(lstm_pred, index=test_df.index)
    
    # Meta
    meta_X = np.column_stack((arima_pred, lstm_pred))
    meta_y = test_df['Close'].values
    X_tr, X_val, y_tr, y_val = train_test_split(meta_X, meta_y, test_size=0.2, random_state=42)
    
    meta = Sequential([
        Dense(32, 'relu', input_shape=(2,)),
        Dropout(0.2),
        Dense(8, 'relu'),
        Dense(1)
    ])
    meta.compile(Adam(1e-3, clipnorm=1.0), 'mse')
    meta.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=200, batch_size=16,
             callbacks=[EarlyStopping('val_loss', 10, restore_best_weights=True)], verbose=0)
    
    meta_pred = meta.predict(meta_X).flatten()
    
    # Metrics
    actual = test_df['Close'].values
    valid_idx = ~arima_pred.isnull()
    
    def calc_metrics(a, p):
        try:
            a = np.asarray(a).ravel()
            p = np.asarray(p).ravel()
            if a.size == 0 or p.size == 0:
                return {'rmse': None, 'mae': None, 'accuracy': None}

            # align lengths
            min_len = min(a.size, p.size)
            a = a[:min_len]
            p = p[:min_len]

            rmse = float(math.sqrt(mean_squared_error(a, p)))
            mae = float(mean_absolute_error(a, p))

            # compute a safe relative error -> accuracy, avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_err = np.abs((a - p) / a)
                rel_err = rel_err[np.isfinite(rel_err)]
                accuracy = float(100 - np.mean(rel_err) * 100) if rel_err.size > 0 else None

            return {'rmse': rmse, 'mae': mae, 'accuracy': accuracy}
        except Exception as e:
            # return None metrics instead of crashing the whole endpoint
            return {'rmse': None, 'mae': None, 'accuracy': None, 'error': str(e)}
    
    metrics = {
        'arima': calc_metrics(actual[valid_idx], arima_pred[valid_idx]),
        'lstm': calc_metrics(actual, lstm_pred),
        'meta': calc_metrics(actual, meta_pred)
    }
    
    # Future forecast
    try:
        arima_f = ARIMA(df['Close'], (5,1,0)).fit().forecast(days).flatten()
    except:
        arima_f = np.array([df['Close'].iloc[-1]]*days)
    
    f_sc, t_sc = MinMaxScaler(), MinMaxScaler()
    f_sc.fit(df[['Close','Volume']])
    t_sc.fit(df[['Close']])
    
    feat_full = f_sc.transform(df[['Close','Volume']])
    targ_full = t_sc.transform(df[['Close']].values.reshape(-1,1))
    
    X_full, y_full = [], []
    for i in range(time_step, len(df)):
        X_full.append(feat_full[i-time_step:i])
        y_full.append(targ_full[i, 0])
    X_full = np.array(X_full)
    y_full = np.array(y_full).reshape(-1,1)
    
    lstm_f = Sequential([LSTM(64, input_shape=(time_step, 2)), Dropout(0.2), Dense(1)])
    lstm_f.compile(Adam(0.001, clipnorm=1.0), 'mse')
    lstm_f.fit(X_full, y_full, epochs=50, batch_size=32,
               callbacks=[EarlyStopping('loss', 8, restore_best_weights=True)], verbose=0)
    
    seq = feat_full[-time_step:].reshape(1, time_step, 2)
    lstm_preds = []
    for _ in range(days):
        p = lstm_f.predict(seq, verbose=0)[0][0]
        lstm_preds.append(p)
        seq = np.append(seq[:, 1:, :], [[[p, seq[0, -1, 1]]]], axis=1)
    
    lstm_f_vals = t_sc.inverse_transform(np.array(lstm_preds).reshape(-1,1)).flatten()
    meta_f = meta.predict(np.column_stack((arima_f, lstm_f_vals))).flatten()
    
    dates = pd.date_range(df.index[-1] + timedelta(1), periods=days+5, freq='B')[:days]
    
    return {
        'metrics': metrics,
        'forecast': {
            'dates': [d.strftime('%Y-%m-%d') for d in dates],
            'arima': arima_f.tolist(),
            'lstm': lstm_f_vals.tolist(),
            'meta': meta_f.tolist()
        }
    }

@app.route('/api/forecast', methods=['POST'])
def forecast():
    try:
        data = request.json
        totp = data.get('totp')
        symbol = data.get('symbol', 'RELIANCE-EQ')
        use_angel = data.get('use_angelone', True)
        days = data.get('forecast_days', 7)
        
        df = None
        live = None
        source = 'yfinance'
        
        # If use_angel is requested but no TOTP provided, try to auto-generate
        if use_angel and not totp and ANGEL_TOTP_SECRET:
            try:
                totp = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
                print('Auto-generated TOTP using ANGEL_TOTP_SECRET')
            except Exception as e:
                print('Failed to generate TOTP from ANGEL_TOTP_SECRET:', e)

        if use_angel and totp:
            obj, err = connect_angelone(totp)
            if obj:
                df = get_angelone_data(obj, symbol, 
                    (datetime.now() - timedelta(days=365*15)).strftime("%Y-%m-%d 09:15"),
                    datetime.now().strftime("%Y-%m-%d %H:%M"))
                
                if df is not None and len(df) > 200:
                    df = df[['Close', 'Volume']].dropna()
                    source = 'angel_one'
                    
                    # Get live quote
                    token = STOCK_TOKENS.get(symbol)
                    if token:
                        try:
                            q = obj.getMarketData({"mode": "FULL", "exchangeTokens": {"NSE": [token]}})
                            if q['status']:
                                d = q['data']['fetched'][0]
                                live = {
                                    'ltp': d['ltp'], 'open': d['open'], 'high': d['high'],
                                    'low': d['low'], 'volume': d['volume'], 
                                    'change': d['netChange'], 'change_pct': d['percentChange']
                                }
                        except:
                            pass
            else:
                return jsonify({'success': False, 'error': f'Angel One failed: {err}'}), 401
        
        if df is None or len(df) < 200:
            sym_yf = symbol.replace('-EQ', '.NS')
            df = yf.download(sym_yf, start='2010-01-01', progress=False)
            df = df[['Close', 'Volume']].dropna()
        
        if len(df) < 200:
            return jsonify({'success': False, 'error': 'Insufficient data'}), 400
        
        results = train_and_forecast(df, days)
        
        return jsonify({
            'success': True,
            'data_source': source,
            'live_quote': live,
            'results': results,
            'latest_close': float(df['Close'].iloc[-1]),
            'data_range': {
                'start': df.index[0].strftime('%Y-%m-%d'),
                'end': df.index[-1].strftime('%Y-%m-%d'),
                'records': len(df)
            }
        })
        
    except Exception as e:
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
    """Quick mock forecast for frontend/dev use. Returns deterministic simple series."""
    try:
        data = request.json or {}
        symbol = data.get('symbol', 'RELIANCE-EQ')
        days = int(data.get('forecast_days', 7) or 7)

        # create simple deterministic forecasts
        last_close = 1000.0
        arima_f = [round(last_close + i * 1.5, 2) for i in range(1, days + 1)]
        lstm_f = [round(last_close + i * 1.2, 2) for i in range(1, days + 1)]
        meta_f = [round((a + l) / 2.0, 2) for a, l in zip(arima_f, lstm_f)]

        dates = pd.date_range(datetime.now().date() + timedelta(1), periods=days, freq='B')

        return jsonify({
            'success': True,
            'data_source': 'mock',
            'results': {
                'metrics': {
                    'arima': {'rmse': 0.0, 'mae': 0.0, 'accuracy': 100.0},
                    'lstm': {'rmse': 0.0, 'mae': 0.0, 'accuracy': 100.0},
                    'meta': {'rmse': 0.0, 'mae': 0.0, 'accuracy': 100.0}
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

@app.route('/')
def index():
    # Serve frontend/index.html
    return app.send_static_file('index.html')


if __name__ == '__main__':
    # Bind to 0.0.0.0 so the server is reachable from other machines in the network
    app.run(debug=True, host='0.0.0.0', port=5000)