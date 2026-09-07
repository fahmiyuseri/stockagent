#!/usr/bin/env python3
"""
🚀 Stock Trading Predictor - Streamlit Web App
Multi-Stock Dashboard with On-Demand Training
Bursa Malaysia Stocks (5347.KL, 1155.KL, 6012.KL, 0066.KL, etc)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import heavy libraries FIRST to avoid XGBoost deadlock
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import pickle as pkl

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="📈 Stock Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Gemini model for the AI analysis panel.
# gemini-2.5-flash  - stable, free tier, laju (shutdown dijadualkan 16 Okt 2026)
# gemini-3-flash-preview - generasi baru, tukar ke sini bila 2.5 ditutup
GEMINI_MODEL = "gemini-2.5-flash"

# Prediction horizons.
#
# Threshold scales with horizon. ±1% is a meaningful daily move, but
# over a month almost every day clears it -- HOLD would vanish and the
# model would just alternate BUY/SELL on noise.
HORIZON_CONFIG = {
    1:  {'threshold': 1.0, 'label': '1 hari  (harian)'},
    5:  {'threshold': 3.0, 'label': '5 hari  (mingguan)'},
    20: {'threshold': 6.0, 'label': '20 hari (bulanan)'},
}

KLCI30 = {
    "1155.KL": "Malayan Banking",
    "1295.KL": "Public Bank",
    "1023.KL": "CIMB Group",
    "1066.KL": "RHB Bank",
    "5819.KL": "Hong Leong Bank",
    "1082.KL": "Hong Leong Financial",
    "5347.KL": "Tenaga Nasional",
    "6742.KL": "YTL Power",
    "4677.KL": "YTL Corporation",
    "6947.KL": "CelcomDigi",
    "6012.KL": "Maxis",
    "4863.KL": "Telekom Malaysia",
    "6888.KL": "Axiata",
    "5225.KL": "IHH Healthcare",
    "1961.KL": "IOI Corporation",
    "2445.KL": "Kuala Lumpur Kepong",
    "5285.KL": "SD Guthrie",
    "4065.KL": "PPB Group",
    "4197.KL": "Sime Darby",
    "5183.KL": "Petronas Chemicals",
    "5681.KL": "Petronas Dagangan",
    "6033.KL": "Petronas Gas",
    "3816.KL": "MISC",
    "8869.KL": "Press Metal",
    "4707.KL": "Nestle Malaysia",
    "7084.KL": "QL Resources",
    "5296.KL": "Mr DIY",
    "5326.KL": "99 Speed Mart",
    "5398.KL": "Gamuda",
    "5211.KL": "Sunway",
}

STOCK_CONFIG = {
    # KLCI30 STOCKS
    "1155.KL": {"name": "Maybank", "csv_path": "/mnt/e/StockistAgent/data/raw/1155.KL.csv", "description": "Banking"},
    "1295.KL": {"name": "Public Bank", "csv_path": "/mnt/e/StockistAgent/data/raw/1295.KL.csv", "description": "Banking"},
    "1023.KL": {"name": "CIMB Group", "csv_path": "/mnt/e/StockistAgent/data/raw/1023.KL.csv", "description": "Banking"},
    "1066.KL": {"name": "RHB Bank", "csv_path": "/mnt/e/StockistAgent/data/raw/1066.KL.csv", "description": "Banking"},
    "5819.KL": {"name": "Hong Leong Bank", "csv_path": "/mnt/e/StockistAgent/data/raw/5819.KL.csv", "description": "Banking"},
    "1082.KL": {"name": "Hong Leong Financial", "csv_path": "/mnt/e/StockistAgent/data/raw/1082.KL.csv", "description": "Financial"},
    "5347.KL": {"name": "Tenaga Nasional", "csv_path": "/mnt/e/StockistAgent/data/raw/5347.KL.csv", "description": "Utility"},
    "6742.KL": {"name": "YTL Power", "csv_path": "/mnt/e/StockistAgent/data/raw/6742.KL.csv", "description": "Utility"},
    "4677.KL": {"name": "YTL Corporation", "csv_path": "/mnt/e/StockistAgent/data/raw/4677.KL.csv", "description": "Conglomerate"},
    "6947.KL": {"name": "CelcomDigi", "csv_path": "/mnt/e/StockistAgent/data/raw/6947.KL.csv", "description": "Telecom"},
    "6012.KL": {"name": "Maxis", "csv_path": "/mnt/e/StockistAgent/data/raw/6012.KL.csv", "description": "Telecom"},
    "4863.KL": {"name": "Telekom Malaysia", "csv_path": "/mnt/e/StockistAgent/data/raw/4863.KL.csv", "description": "Telecom"},
    "6888.KL": {"name": "Axiata", "csv_path": "/mnt/e/StockistAgent/data/raw/6888.KL.csv", "description": "Telecom"},
    "5225.KL": {"name": "IHH Healthcare", "csv_path": "/mnt/e/StockistAgent/data/raw/5225.KL.csv", "description": "Healthcare"},
    "1961.KL": {"name": "IOI Corporation", "csv_path": "/mnt/e/StockistAgent/data/raw/1961.KL.csv", "description": "Plantation"},
    "2445.KL": {"name": "Kuala Lumpur Kepong", "csv_path": "/mnt/e/StockistAgent/data/raw/2445.KL.csv", "description": "Plantation"},
    "5285.KL": {"name": "SD Guthrie", "csv_path": "/mnt/e/StockistAgent/data/raw/5285.KL.csv", "description": "Plantation"},
    "4065.KL": {"name": "PPB Group", "csv_path": "/mnt/e/StockistAgent/data/raw/4065.KL.csv", "description": "F&B"},
    "4197.KL": {"name": "Sime Darby", "csv_path": "/mnt/e/StockistAgent/data/raw/4197.KL.csv", "description": "Conglomerate"},
    "5183.KL": {"name": "Petronas Chemicals", "csv_path": "/mnt/e/StockistAgent/data/raw/5183.KL.csv", "description": "Chemical"},
    "5681.KL": {"name": "Petronas Dagangan", "csv_path": "/mnt/e/StockistAgent/data/raw/5681.KL.csv", "description": "Energy"},
    "6033.KL": {"name": "Petronas Gas", "csv_path": "/mnt/e/StockistAgent/data/raw/6033.KL.csv", "description": "Energy"},
    "3816.KL": {"name": "MISC", "csv_path": "/mnt/e/StockistAgent/data/raw/3816.KL.csv", "description": "Shipping"},
    "8869.KL": {"name": "Press Metal", "csv_path": "/mnt/e/StockistAgent/data/raw/8869.KL.csv", "description": "Metals"},
    "4707.KL": {"name": "Nestle Malaysia", "csv_path": "/mnt/e/StockistAgent/data/raw/4707.KL.csv", "description": "F&B"},
    "7084.KL": {"name": "QL Resources", "csv_path": "/mnt/e/StockistAgent/data/raw/7084.KL.csv", "description": "F&B"},
    "5296.KL": {"name": "Mr DIY", "csv_path": "/mnt/e/StockistAgent/data/raw/5296.KL.csv", "description": "Retail"},
    "5326.KL": {"name": "99 Speed Mart", "csv_path": "/mnt/e/StockistAgent/data/raw/5326.KL.csv", "description": "Retail"},
    "5398.KL": {"name": "Gamuda", "csv_path": "/mnt/e/StockistAgent/data/raw/5398.KL.csv", "description": "Construction"},
    "5211.KL": {"name": "Sunway", "csv_path": "/mnt/e/StockistAgent/data/raw/5211.KL.csv", "description": "Real Estate"},
    
    # OTHER POPULAR STOCKS
    "4456.KL": {"name": "DNEX", "csv_path": "/mnt/e/StockistAgent/data/raw/4456.KL.csv", "description": "Tech"},
    "0181.KL": {"name": "AEMULUS", "csv_path": "/mnt/e/StockistAgent/data/raw/0181.KL.csv", "description": "Tech"},
    "0275.KL": {"name": "OPPSTAR", "csv_path": "/mnt/e/StockistAgent/data/raw/0275.KL.csv", "description": "Tech"},
    "8993.KL": {"name": "Wah Seong", "csv_path": "/mnt/e/StockistAgent/data/raw/8993.KL.csv", "description": "Industrial"},
    "0175.KL": {"name": "Perak Daya", "csv_path": "/mnt/e/StockistAgent/data/raw/0175.KL.csv", "description": "Property"},
    "0127.KL": {"name": "Tropicana", "csv_path": "/mnt/e/StockistAgent/data/raw/0127.KL.csv", "description": "Property"},
    "6599.KL": {"name": "Sapura Energy", "csv_path": "/mnt/e/StockistAgent/data/raw/6599.KL.csv", "description": "Oil & Gas"},
    "0098.KL": {"name": "AMMB", "csv_path": "/mnt/e/StockistAgent/data/raw/0098.KL.csv", "description": "Banking"},
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_yfinance():
    import yfinance as yf
    return yf

def get_latest_price(ticker):
    """Fetch latest price from Yahoo Finance"""
    try:
        yf = load_yfinance()
        stock = yf.Ticker(ticker)
        data = stock.history(period='5d')
        
        if len(data) == 0:
            return None
        
        latest = data.iloc[-1]
        return {
            'price': latest['Close'],
            'high': latest['High'],
            'low': latest['Low'],
            'volume': int(latest['Volume']),
            'date': data.index[-1].strftime('%Y-%m-%d'),
            'change': ((latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close'] * 100) if len(data) > 1 else 0
        }
    except:
        return None

@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(ticker):
    """
    Pull full price history straight from Yahoo.

    Cached for 15 minutes because Streamlit reruns the whole script on
    every widget interaction -- without this, each checkbox click would
    fire a fresh download and Yahoo would start rate-limiting.
    """
    yf = load_yfinance()

    df = yf.download(
        ticker,
        start="2015-01-01",
        progress=False,
        auto_adjust=True
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    if df is None or len(df) == 0:
        return None

    df = df.dropna(subset=['Close'])
    df.reset_index(inplace=True)
    df.columns = [col.lower() for col in df.columns]

    if 'adjclose' in df.columns:
        df = df.rename(columns={'adjclose': 'close'})
    elif 'adj close' in df.columns:
        df = df.rename(columns={'adj close': 'close'})

    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    available = [c for c in required_cols if c in df.columns]

    if len(available) < 5:
        return None

    df = df[available].rename(columns=str.capitalize)
    df['Date'] = pd.to_datetime(df['Date'])

    return df.sort_values('Date').reset_index(drop=True)


def load_csv_data(csv_path, ticker):
    """
    Get price history, freshest source first.

    API is the primary source so indicators are always computed on
    current data. The CSV is written on every successful fetch and read
    only when Yahoo is unreachable -- stale data beats no data.
    """
    try:
        df = fetch_history(ticker)

        if df is not None and len(df) > 0:
            last = df['Date'].iloc[-1].date()
            print(f"✅ {ticker}: {len(df)} baris dari Yahoo (hingga {last})")

            # Refresh the local copy so the fallback stays useful.
            try:
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                df.to_csv(csv_path, index=False)
            except Exception as e:
                print(f"⚠️ Tak dapat simpan CSV: {e}")

            return df

        print(f"⚠️ {ticker}: Yahoo pulangkan kosong")

    except Exception as e:
        print(f"⚠️ {ticker}: Yahoo gagal — {str(e)[:80]}")

    # Fallback: whatever was cached on disk last time.
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            last = df['Date'].iloc[-1].date()
            print(f"📂 {ticker}: guna CSV simpanan (hingga {last})")
            return df
        except Exception as e:
            print(f"❌ Tak dapat baca CSV: {e}")
            return None

    print(f"❌ {ticker}: tiada data langsung")
    return None

def create_indicators(df, with_market=True):
    """
    Create technical indicators.

    with_market: attach KLCI context features. Set False only if you
        need the bare per-stock indicator set.
    """
    df_ind = df.copy()
    
    # Moving Averages
    df_ind['MA5'] = df_ind['Close'].rolling(window=5).mean()
    df_ind['MA10'] = df_ind['Close'].rolling(window=10).mean()
    df_ind['MA20'] = df_ind['Close'].rolling(window=20).mean()
    df_ind['MA50'] = df_ind['Close'].rolling(window=50).mean()
    
    # RSI
    def calculate_rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df_ind['RSI'] = calculate_rsi(df_ind['Close'], 14)
    
    # MACD
    exp1 = df_ind['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_ind['Close'].ewm(span=26, adjust=False).mean()
    df_ind['MACD'] = exp1 - exp2
    df_ind['MACD_Signal'] = df_ind['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    sma = df_ind['Close'].rolling(window=20).mean()
    std = df_ind['Close'].rolling(window=20).std()
    df_ind['BB_Upper'] = sma + (std * 2)
    df_ind['BB_Lower'] = sma - (std * 2)
    df_ind['BB_Position'] = (df_ind['Close'] - df_ind['BB_Lower']) / (df_ind['BB_Upper'] - df_ind['BB_Lower'])
    
    # Price changes
    df_ind['Price_Change'] = df_ind['Close'].pct_change() * 100
    df_ind['High_Low_Ratio'] = df_ind['High'] / df_ind['Low']
    df_ind['Volume_MA'] = df_ind['Volume'].rolling(window=20).mean()
    df_ind['Volume_Ratio'] = df_ind['Volume'] / df_ind['Volume_MA']

    df_ind = df_ind.dropna().reset_index(drop=True)

    if with_market:
        df_ind, _ = add_market_features(df_ind)

    return df_ind

@st.cache_data(ttl=900, show_spinner=False)
def fetch_index():
    """
    FTSE Bursa Malaysia KLCI daily history.

    Cached separately from stocks so switching tickers doesn't refetch
    the index every time.
    """
    yf = load_yfinance()
    try:
        df = yf.download("^KLSE", start="2015-01-01",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df is None or len(df) == 0:
            return None
        df = df.dropna(subset=['Close']).reset_index()
        df.columns = [c.lower() for c in df.columns]
        df = df[['date', 'close']].rename(
            columns={'date': 'Date', 'close': 'KLCI'}
        )
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values('Date').reset_index(drop=True)
    except Exception as e:
        print(f"⚠️ KLCI fetch failed: {e}")
        return None


def add_market_features(df_ind):
    """
    Attach market context.

    Without this the model sees one stock in a vacuum. Most Bursa
    counters move with the index -- a stock falling 2% on a day the
    KLCI fell 2% means something completely different from the same
    fall on a flat market. Relative strength captures that difference.

    Returns the frame unchanged if the index can't be fetched, so a
    Yahoo outage degrades the model rather than breaking it.
    """
    klci = fetch_index()
    if klci is None:
        print("⚠️ KLCI unavailable — training without market features")
        return df_ind, False

    d = df_ind.merge(klci, on='Date', how='left')
    d['KLCI'] = d['KLCI'].ffill()

    if d['KLCI'].isna().all():
        return df_ind, False

    # Index momentum over several lookbacks
    d['KLCI_Change'] = d['KLCI'].pct_change() * 100
    d['KLCI_Mom_5'] = d['KLCI'].pct_change(5) * 100
    d['KLCI_Mom_20'] = d['KLCI'].pct_change(20) * 100

    # Is the market itself in an uptrend?
    klci_ma50 = d['KLCI'].rolling(50).mean()
    d['KLCI_vs_MA50'] = (d['KLCI'] - klci_ma50) / klci_ma50 * 100

    # Relative strength: is this stock outpacing the market or lagging?
    d['RelStrength_1'] = d['Price_Change'] - d['KLCI_Change']
    d['RelStrength_5'] = (
        d['Close'].pct_change(5) * 100 - d['KLCI_Mom_5']
    )
    d['RelStrength_20'] = (
        d['Close'].pct_change(20) * 100 - d['KLCI_Mom_20']
    )

    # Raw index level would let the model memorise calendar periods,
    # so it is dropped -- only the derived changes are kept.
    d = d.drop(columns=['KLCI'])

    return d.dropna().reset_index(drop=True), True


def check_model_exists(ticker, horizon=1):
    """
    Check if a usable model exists for this ticker and horizon.

    Returns False for packages saved before the regressor was added,
    so those get retrained instead of silently losing price targets.
    """
    model_path, trades_path = get_model_paths(ticker, horizon)

    if not (os.path.exists(model_path) and os.path.exists(trades_path)):
        return False

    try:
        with open(model_path, 'rb') as f:
            package = pkl.load(f)
        if package.get('regressor') is None:
            return False
        # Files saved before the OOS arrays were stored can't drive the
        # confidence filter, and predate the KLCI features. Retrain.
        with open(trades_path, 'rb') as f:
            return isinstance(pkl.load(f), dict)
    except Exception:
        return False

def get_model_paths(ticker, horizon=1):
    """Return versioned, horizon-specific model paths."""
    prefix = ticker.replace('.', '_')
    return (
        f"{prefix}_h{horizon}_xgb_model_v2.pkl",
        f"{prefix}_h{horizon}_backtest_trades.pkl"
    )


def build_training_dataset(df_ind, horizon=1, threshold=1.0):
    """
    Build the supervised learning dataset.

    Target, looking `horizon` trading days ahead:
      - BUY  = forward return > +threshold%
      - HOLD = forward return within ±threshold%
      - SELL = forward return < -threshold%

    IMPORTANT:
      Future_Close / Daily_Return / Target are never used as features.
    """
    data = df_ind.copy()

    data['Next_Close'] = data['Close'].shift(-horizon)
    data['Daily_Return'] = (
        (data['Next_Close'] - data['Close'])
        / data['Close'] * 100
    )

    data['Target'] = 0
    data.loc[data['Daily_Return'] > threshold, 'Target'] = 1
    data.loc[data['Daily_Return'] < -threshold, 'Target'] = -1

    # The final `horizon` rows have no forward return, so they cannot
    # be used for training.
    data = data.iloc[:-horizon].copy().reset_index(drop=True)

    # XGBoost classes must be 0, 1, 2.
    target_mapping = {-1: 0, 0: 1, 1: 2}
    data['Target'] = data['Target'].map(target_mapping).astype(int)

    exclude_cols = [
        'Date',
        'Close',
        'Next_Close',
        'Target',
        'Daily_Return'
    ]

    feature_cols = [
        col for col in data.columns
        if col not in exclude_cols
    ]

    X = data[feature_cols].astype(float).values
    y = data['Target'].astype(int).values

    # Regression target: next-day percentage change, NOT raw price.
    #
    # Predicting raw price looks impressive (R² near 0.99) but is
    # meaningless -- tomorrow's price is almost today's price, so a model
    # that just echoes today scores well. Percentage change removes that
    # free ride, so the R² you see is real signal or nothing.
    y_return = data['Daily_Return'].astype(float).values

    return data, X, y, y_return, feature_cols


def create_xgb_model():
    """Create a conservative XGBoost classifier to reduce overfitting."""
    return XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=42,
        tree_method='hist',
        device='cuda',
        eval_metric='mlogloss',
        verbosity=0
    )


def create_xgb_regressor():
    """
    Create the price-target regressor.

    Deliberately shallower than the classifier. Percentage change is
    mostly noise, so a deep tree will memorise the training set and
    return confident nonsense out of sample.
    """
    return XGBRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.02,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=3.0,
        random_state=42,
        tree_method='hist',
        device='cuda',
        verbosity=0
    )


def generate_backtest_trades(
    data,
    y_pred,
    proba,
    split_idx,
    min_confidence=0.0
):
    """
    Convert OOS BUY/HOLD/SELL predictions into long-only trades.

    Rules:
      BUY  when not in a position -> ENTRY
      HOLD while in a position    -> keep holding
      SELL while in a position    -> EXIT

    min_confidence: skip entries the model isn't sure about. With three
        classes, 33% is a coin flip -- entering on a 40% signal is
        trading on almost nothing, and each entry costs ~1.1% round trip.
        Raising this cuts trade count, which is where the cost damage is.

        Exits always fire regardless. Refusing to close a losing position
        because confidence dipped would be the worst of both worlds.

    Prices are CLOSE prices. No transaction fees are included here --
    they're applied separately in cost_adjusted_summary().
    """
    signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

    trades = []
    in_position = False
    entry_price = None
    entry_date = None
    entry_confidence = None

    test_data = data.iloc[split_idx:].copy().reset_index(drop=True)

    for i in range(len(test_data)):
        row = test_data.iloc[i]
        pred = int(y_pred[i])
        signal = signal_map[pred]
        confidence = float(proba[i][pred] * 100)
        price = float(row['Close'])
        date = row['Date']

        if signal == 'BUY' and not in_position:
            if confidence < min_confidence:
                continue
            in_position = True
            entry_price = price
            entry_date = date
            entry_confidence = confidence

        elif signal == 'SELL' and in_position:
            exit_price = price
            exit_date = date

            return_pct = (
                (exit_price - entry_price)
                / entry_price * 100
            )

            trades.append({
                'Entry Date': entry_date,
                'Entry Price': entry_price,
                'Exit Date': exit_date,
                'Exit Price': exit_price,
                'Return %': return_pct,
                'Entry Confidence %': entry_confidence,
                'Exit Confidence %': confidence,
                'Result': (
                    'WIN' if return_pct > 0
                    else 'LOSS' if return_pct < 0
                    else 'BREAKEVEN'
                )
            })

            in_position = False
            entry_price = None
            entry_date = None
            entry_confidence = None

    # Close an open trade at the last available test close.
    #
    # If the entry landed on that very same bar, there is no holding
    # period and nothing was risked -- recording it would add a fake
    # trade to the statistics.
    if in_position and len(test_data) > 0:
        row = test_data.iloc[-1]
        exit_price = float(row['Close'])
        exit_date = row['Date']

        if exit_date != entry_date:
            return_pct = (
                (exit_price - entry_price)
                / entry_price * 100
            )

            trades.append({
                'Entry Date': entry_date,
                'Entry Price': entry_price,
                'Exit Date': exit_date,
                'Exit Price': exit_price,
                'Return %': return_pct,
                'Entry Confidence %': entry_confidence,
                'Exit Confidence %': None,
                'Result': (
                    'WIN' if return_pct > 0
                    else 'LOSS' if return_pct < 0
                    else 'BREAKEVEN'
                ),
                'Exit Type': 'End of test period'
            })

    return pd.DataFrame(trades)


def calculate_backtest_stats(trades):
    """Calculate simple price-only trading statistics."""
    if trades is None or trades.empty:
        return {
            'trades': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0
        }

    returns = trades['Return %'].astype(float)

    # Compounded gross return. No transaction fee.
    equity = (1 + returns / 100).cumprod()
    total_return = (equity.iloc[-1] - 1) * 100

    running_max = equity.cummax()
    drawdown = (equity / running_max - 1) * 100
    max_drawdown = float(drawdown.min())

    return {
        'trades': int(len(trades)),
        'win_rate': float((returns > 0).mean() * 100),
        'avg_return': float(returns.mean()),
        'total_return': float(total_return),
        'max_drawdown': max_drawdown
    }


def train_model(ticker, horizon=1, progress_bar=None, status_text=None):
    """
    Train XGBoost models for a stock at a given horizon.

    Two models are used:
      1. OOS model: train on first 80%, evaluate on final 20%.
      2. Final model: retrain on 100% of historical data for live prediction.

    The OOS model is used ONLY to calculate honest historical
    signal/trade performance.
    """
    try:
        threshold = HORIZON_CONFIG[horizon]['threshold']
        csv_path = STOCK_CONFIG[ticker]['csv_path']

        if status_text:
            status_text.text("📥 Loading data...")

        df = load_csv_data(csv_path, ticker)

        if df is None:
            return False, "CSV file not found"

        if progress_bar:
            progress_bar.progress(10)

        if status_text:
            status_text.text("📊 Creating indicators...")

        df_ind = create_indicators(df)

        if progress_bar:
            progress_bar.progress(25)

        # ------------------------------------------------------------------
        # Build target + features
        # ------------------------------------------------------------------
        data, X, y, y_return, feature_cols = build_training_dataset(
            df_ind, horizon=horizon, threshold=threshold
        )

        if len(X) < 200:
            return False, "Not enough data for training"

        if progress_bar:
            progress_bar.progress(35)

        # ------------------------------------------------------------------
        # Chronological 80/20 split, with an embargo gap.
        #
        # The last `horizon` training rows are labelled using prices that
        # fall inside the test window -- so without a gap the model gets
        # a peek at the future it is meant to be judged on. Dropping those
        # rows costs a handful of samples and buys an honest score.
        # ------------------------------------------------------------------
        split_idx = int(len(X) * 0.80)
        embargo = horizon

        train_end = split_idx - embargo

        if train_end < 200:
            return False, "Not enough data for training after embargo"

        X_train_raw = X[:train_end]
        y_train = y[:train_end]
        y_return_train = y_return[:train_end]

        X_test_raw = X[split_idx:]
        y_test = y[split_idx:]
        y_return_test = y_return[split_idx:]

        # ------------------------------------------------------------------
        # OOS scaler - FIT ONLY ON TRAINING DATA
        # This removes the previous data leakage.
        # ------------------------------------------------------------------
        oos_scaler = StandardScaler()

        X_train = oos_scaler.fit_transform(X_train_raw)
        X_test = oos_scaler.transform(X_test_raw)

        if status_text:
            status_text.text("🤖 Training OOS XGBoost model...")

        # ------------------------------------------------------------------
        # OOS model
        # ------------------------------------------------------------------
        oos_model = create_xgb_model()

        oos_model.fit(
            X_train,
            y_train,
           # eval_set=[(X_test, y_test)],
            verbose=False
        )

        if progress_bar:
            progress_bar.progress(70)

        # ------------------------------------------------------------------
        # OOS evaluation
        # ------------------------------------------------------------------
        y_pred = oos_model.predict(X_test)
        proba = oos_model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        # ------------------------------------------------------------------
        # OOS trade simulation
        # ------------------------------------------------------------------
        trades = generate_backtest_trades(
            data=data,
            y_pred=y_pred,
            proba=proba,
            split_idx=split_idx
        )

        stats = calculate_backtest_stats(trades)

        # ------------------------------------------------------------------
        # OOS regressor: how far off is the price target, honestly?
        # ------------------------------------------------------------------
        if status_text:
            status_text.text("📉 Training OOS regressor...")

        oos_regressor = create_xgb_regressor()
        oos_regressor.fit(X_train, y_return_train, verbose=False)

        pred_return = oos_regressor.predict(X_test)

        # MAE in percentage points: typical miss on the daily move.
        reg_mae = float(mean_absolute_error(y_return_test, pred_return))

        # R² on percentage change. Near zero (or negative) means the
        # regressor adds nothing beyond guessing the average move.
        reg_r2 = float(r2_score(y_return_test, pred_return))

        if progress_bar:
            progress_bar.progress(80)

        # ------------------------------------------------------------------
        # Final model for LIVE prediction
        #
        # We intentionally retrain using all historical data AFTER
        # the honest OOS evaluation is complete.
        # ------------------------------------------------------------------
        if status_text:
            status_text.text("🔄 Retraining final model on all data...")

        final_scaler = StandardScaler()
        X_all = final_scaler.fit_transform(X)

        final_model = create_xgb_model()
        final_model.fit(
            X_all,
            y,
            verbose=False
        )

        final_regressor = create_xgb_regressor()
        final_regressor.fit(X_all, y_return, verbose=False)

        # ------------------------------------------------------------------
        # Save model package.
        #
        # Feature names are saved together with the model so prediction
        # always uses EXACTLY the same columns as training.
        # ------------------------------------------------------------------
        model_path, trades_path = get_model_paths(ticker, horizon)

        model_package = {
            'model': final_model,
            'regressor': final_regressor,
            'scaler': final_scaler,
            'feature_cols': feature_cols,
            'target_mapping': {
                0: 'SELL',
                1: 'HOLD',
                2: 'BUY'
            },
            'threshold': threshold,
            'horizon_days': horizon,
            'oos_accuracy': accuracy,
            'oos_reg_mae': reg_mae,
            'oos_reg_r2': reg_r2,
            'oos_split_idx': split_idx,
            'oos_embargo': embargo,
            'oos_train_rows': int(train_end),
            'oos_test_rows': int(len(X) - split_idx),
            'trained_until': str(data['Date'].iloc[-1])
        }

        with open(model_path, 'wb') as f:
            pkl.dump(model_package, f)

        # Save trades plus the raw OOS output, so confidence and
        # trend filters can be re-tested without retraining.
        with open(trades_path, 'wb') as f:
            pkl.dump({
                'trades': trades,
                'oos': {
                    'y_pred': y_pred,
                    'proba': proba,
                    'split_idx': split_idx,
                    'data': data[['Date', 'Close']].copy(),
                },
            }, f)

        if progress_bar:
            progress_bar.progress(100)

        return True, (
            f"✅ Trained! "
            f"OOS Accuracy: {accuracy * 100:.1f}% | "
            f"Trades: {stats['trades']} | "
            f"Win Rate: {stats['win_rate']:.1f}% | "
            f"Total Return: {stats['total_return']:+.2f}% | "
            f"Regressor MAE: {reg_mae:.2f}pp, R²: {reg_r2:.3f}"
        )

    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def get_model_freshness(ticker, horizon, latest_data_date):
    """
    Compare when the model was trained against the newest data available.

    The last `horizon` rows can never be trained on -- their labels need
    prices that haven't happened yet. So a freshly trained model always
    sits a horizon behind the data. That gap is subtracted before judging
    staleness, otherwise a 20-day model would look a month out of date
    the moment it finished training.
    """
    try:
        model_path, _ = get_model_paths(ticker, horizon)
        if not os.path.exists(model_path):
            return None

        with open(model_path, 'rb') as f:
            pkg = pkl.load(f)

        trained_until = pkg.get('trained_until')
        if not trained_until:
            return None

        trained_dt = pd.to_datetime(str(trained_until)[:10])
        latest_dt = pd.to_datetime(str(latest_data_date)[:10])
        gap = (latest_dt - trained_dt).days

        # Trading days aren't calendar days, so approximate the
        # unavoidable portion generously (7/5 plus a holiday allowance).
        expected_gap = int(horizon * 1.4) + 2
        real_lag = max(0, gap - expected_gap)

        return {
            'trained_until': trained_dt.date(),
            'latest_data': latest_dt.date(),
            'gap_days': gap,
            'expected_gap': expected_gap,
            'real_lag': real_lag,
            # 30 days is a deliberate floor. Retraining more often than
            # this mostly re-fits the same patterns, and invites tuning
            # the model until the backtest looks good -- which makes the
            # numbers prettier and the model worse.
            'stale': real_lag >= 30
        }
    except Exception:
        return None


def predict_signal(ticker, horizon=1):
    """Make the latest live prediction for a ticker at a given horizon."""
    try:
        csv_path = STOCK_CONFIG[ticker]['csv_path']
        df = load_csv_data(csv_path, ticker)

        if df is None:
            return None

        df_ind = create_indicators(df)

        model_path, _ = get_model_paths(ticker, horizon)

        with open(model_path, 'rb') as f:
            model_package = pkl.load(f)

        model = model_package['model']
        scaler = model_package['scaler']
        feature_cols = model_package['feature_cols']

        # Exact same feature order used during training.
        missing_cols = [
            col for col in feature_cols
            if col not in df_ind.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Missing feature columns: {missing_cols}"
            )

        X = df_ind[feature_cols].astype(float).values

        X_normalized = scaler.transform(X)

        latest_features = X_normalized[-1:].reshape(1, -1)

        pred = int(model.predict(latest_features)[0])
        proba = model.predict_proba(latest_features)[0]

        # Price target from the regressor, if this model package has one.
        # Older packages predate the regressor, so degrade gracefully
        # rather than crashing.
        current_price = float(df_ind['Close'].iloc[-1])
        regressor = model_package.get('regressor')

        if regressor is not None:
            pred_return = float(regressor.predict(latest_features)[0])
            target_price = current_price * (1 + pred_return / 100)
        else:
            pred_return = None
            target_price = None

        signal_map = {
            0: 'SELL',
            1: 'HOLD',
            2: 'BUY'
        }

        emoji_map = {
            0: '🔴',
            1: '🟡',
            2: '🟢'
        }

        return {
            'signal': signal_map[pred],
            'emoji': emoji_map[pred],
            'confidence': float(proba[pred] * 100),
            'proba': proba,
            'price': current_price,
            'date': df_ind['Date'].iloc[-1],
            'pred_return': pred_return,
            'target_price': target_price,
            'oos_accuracy': float(
                model_package.get('oos_accuracy', 0) * 100
            ),
            'oos_reg_mae': model_package.get('oos_reg_mae'),
            'oos_reg_r2': model_package.get('oos_reg_r2'),
            'horizon': model_package.get('horizon_days', 1),
            'threshold': model_package.get('threshold', 1.0)
        }

    except Exception as e:
        print(f"Prediction error for {ticker}: {e}")
        return None


def load_backtest_trades(ticker, horizon=1):
    """Load previously generated OOS trade history."""
    try:
        _, trades_path = get_model_paths(ticker, horizon)

        if not os.path.exists(trades_path):
            return pd.DataFrame()

        with open(trades_path, 'rb') as f:
            obj = pkl.load(f)

        if obj is None:
            return pd.DataFrame()

        # New format bundles trades with the OOS arrays.
        if isinstance(obj, dict):
            return obj.get('trades', pd.DataFrame())

        return obj

    except Exception as e:
        print(f"Backtest load error for {ticker}: {e}")
        return pd.DataFrame()


def load_oos_bundle(ticker, horizon=1):
    """
    Load trades plus the raw OOS arrays.

    Returns (trades, oos) where oos is None for model files saved
    before the arrays were stored -- those need a retrain before the
    confidence filter can work.
    """
    try:
        _, trades_path = get_model_paths(ticker, horizon)
        if not os.path.exists(trades_path):
            return pd.DataFrame(), None

        with open(trades_path, 'rb') as f:
            obj = pkl.load(f)

        if isinstance(obj, dict):
            return obj.get('trades', pd.DataFrame()), obj.get('oos')

        return (obj if obj is not None else pd.DataFrame()), None

    except Exception:
        return pd.DataFrame(), None


def get_latest_trade_status(ticker, horizon=1):
    """
    Return the latest open/closed trade state from OOS backtest.

    This is historical OOS information, not a new live trade.
    """
    trades = load_backtest_trades(ticker, horizon)

    if trades.empty:
        return None

    latest = trades.iloc[-1]

    exit_type = latest.get('Exit Type') if 'Exit Type' in trades.columns else None
    if pd.isna(exit_type):
        exit_type = None

    return {
        'entry_date': latest.get('Entry Date'),
        'entry_price': float(latest.get('Entry Price')),
        'exit_date': latest.get('Exit Date'),
        'exit_price': float(latest.get('Exit Price')),
        'return_pct': float(latest.get('Return %')),
        'result': latest.get('Result', ''),
        'exit_type': exit_type or 'SELL signal',
        'forced_exit': exit_type is not None
    }


def build_indicator_snapshot(ticker):
    """Collect the latest indicator readings for the AI prompt."""
    csv_path = STOCK_CONFIG[ticker]['csv_path']
    df = load_csv_data(csv_path, ticker)

    if df is None:
        return None

    df_ind = create_indicators(df)
    row = df_ind.iloc[-1]

    # 20-day context so the AI can see direction, not just a snapshot
    recent = df_ind.tail(20)

    return {
        'date': str(row['Date'])[:10],
        'close': float(row['Close']),
        'ma5': float(row['MA5']),
        'ma20': float(row['MA20']),
        'ma50': float(row['MA50']),
        'rsi': float(row['RSI']),
        'macd': float(row['MACD']),
        'macd_signal': float(row['MACD_Signal']),
        'bb_upper': float(row['BB_Upper']),
        'bb_lower': float(row['BB_Lower']),
        'bb_position': float(row['BB_Position']),
        'volume_ratio': float(row['Volume_Ratio']),
        'change_20d': float(
            (row['Close'] - recent['Close'].iloc[0])
            / recent['Close'].iloc[0] * 100
        ),
        'high_20d': float(recent['High'].max()),
        'low_20d': float(recent['Low'].min()),
    }


def build_analysis_prompt(ticker, snapshot, prediction, trade_stats=None,
                          chart_summary=None):
    """
    Build the analysis prompt.

    Kept separate from the API call so the same text can be shown
    to the user for manual pasting into any chat interface.

    The prompt now carries the model's own diagnostics -- accuracy
    against the 33% random baseline, regressor R², trade count,
    drawdown, and the buy-and-hold comparison. Without those the AI
    would treat every signal as equally credible, which is exactly
    the mistake to avoid.
    """
    name = STOCK_CONFIG[ticker]['name']
    dp = 4 if snapshot['close'] < 5 else 2
    horizon = prediction.get('horizon', 1)
    threshold = prediction.get('threshold', 1.0)

    # --- Model quality block ---
    acc = prediction.get('oos_accuracy', 0)
    r2 = prediction.get('oos_reg_r2')
    mae = prediction.get('oos_reg_mae')

    quality_lines = [
        f"- Accuracy out-of-sample: {acc:.1f}% "
        f"(tekaan rawak untuk 3 kelas = 33.3%)"
    ]

    if r2 is not None:
        verdict = (
            "tidak berguna, lebih teruk dari meneka purata" if r2 < 0
            else "hampir sifar, tidak lebih baik dari tekaan" if r2 < 0.05
            else "lemah" if r2 < 0.15
            else "sederhana"
        )
        quality_lines.append(
            f"- Regressor R² pada % perubahan: {r2:.3f} ({verdict})"
        )
    if mae is not None:
        quality_lines.append(
            f"- Regressor purata tersasar: {mae:.2f} mata peratusan"
        )

    quality_block = "\n".join(quality_lines)

    # --- Backtest block ---
    backtest_block = "Tiada backtest tersedia."
    if trade_stats and trade_stats.get('trades', 0) > 0:
        n = trade_stats['trades']
        reliability = (
            "terlalu sedikit untuk apa-apa kesimpulan" if n < 10
            else "sedikit, layan sebagai petunjuk kasar" if n < 25
            else "mencukupi untuk petunjuk awal"
        )
        overlap = ""
        if horizon > 1:
            overlap = (
                f"\n- NOTA: pada horizon {horizon} hari, tetingkap ramalan "
                f"bertindih, jadi trade-trade ini kurang bebas antara satu "
                f"sama lain berbanding horizon harian"
            )
        backtest_block = (
            f"- Bilangan trade: {n} ({reliability})\n"
            f"- Win rate: {trade_stats['win_rate']:.1f}%\n"
            f"- Purata setiap trade: {trade_stats['avg_return']:+.2f}%\n"
            f"- Jumlah pulangan: {trade_stats['total_return']:+.2f}%\n"
            f"- Penurunan maksimum: {trade_stats['max_drawdown']:.2f}%"
            f"{overlap}"
        )

    # --- Baseline block: the comparison that decides everything ---
    baseline_block = ""
    if chart_summary and 'error' not in chart_summary:
        edge = chart_summary['edge']
        judgement = (
            "MODEL KALAH pada strategi paling malas" if edge < 0
            else "kelebihan tipis, mungkin lenyap selepas kos" if edge < 5
            else "model mengatasi beli & simpan"
        )
        baseline_block = f"""

PERBANDINGAN DENGAN BELI & SIMPAN ({chart_summary['start']} → {chart_summary['end']}):
- Ikut model: {chart_summary['model']:+.2f}%
- Beli & simpan sahaja: {chart_summary['hold']:+.2f}%
- Kelebihan model: {edge:+.2f}% ({judgement})"""

    # --- Price target block ---
    target_line = ""
    if prediction.get('target_price') is not None:
        target_line = (
            f"\n- Sasaran regressor selepas {horizon} hari: "
            f"RM {prediction['target_price']:.{dp}f} "
            f"({prediction['pred_return']:+.2f}%)"
        )

    return f"""Kau penganalisis teknikal yang berhati-hati. Analisa data saham Bursa Malaysia ini.

═══════════════════════════════════════════
SAHAM: {ticker} ({name})
TARIKH DATA: {snapshot['date']}
═══════════════════════════════════════════

HARGA:
- Penutup: RM {snapshot['close']:.{dp}f}
- Perubahan 20 hari: {snapshot['change_20d']:+.2f}%
- Julat 20 hari: RM {snapshot['low_20d']:.{dp}f} - RM {snapshot['high_20d']:.{dp}f}

PURATA BERGERAK:
- MA5: RM {snapshot['ma5']:.{dp}f}
- MA20: RM {snapshot['ma20']:.{dp}f}
- MA50: RM {snapshot['ma50']:.{dp}f}

MOMENTUM:
- RSI(14): {snapshot['rsi']:.1f}
- MACD: {snapshot['macd']:.4f}
- MACD Signal: {snapshot['macd_signal']:.4f}

BOLLINGER BANDS:
- Atas: RM {snapshot['bb_upper']:.{dp}f}
- Bawah: RM {snapshot['bb_lower']:.{dp}f}
- Kedudukan: {snapshot['bb_position']:.2f} (0 = band bawah, 1 = band atas)

VOLUME:
- Nisbah vs purata 20 hari: {snapshot['volume_ratio']:.2f}x

═══════════════════════════════════════════
RAMALAN MODEL ML (XGBoost)
═══════════════════════════════════════════
- Signal: {prediction['signal']} pada keyakinan {prediction['confidence']:.1f}%
- Taburan: SELL {prediction['proba'][0]*100:.1f}% · HOLD {prediction['proba'][1]*100:.1f}% · BUY {prediction['proba'][2]*100:.1f}%
- Horizon: {horizon} hari dagangan ke depan
- Threshold: ±{threshold:.0f}% (BUY jika naik melebihi ini, SELL jika jatuh){target_line}

KUALITI MODEL (out-of-sample, ada embargo di sempadan latihan/ujian):
{quality_block}

PRESTASI BACKTEST:
{backtest_block}{baseline_block}

Nota: backtest long-only, harga penutup, tiada brokerage atau duti setem.

═══════════════════════════════════════════
TUGAS
═══════════════════════════════════════════

Cari berita terkini tentang {name} ({ticker}), terutamanya 3 bulan lepas:
- Skandal, siasatan, tindakan penguatkuasaan (SC, Bursa, MACC)
- Suspension saham, query UMA, notis Bursa
- Perletakan jawatan CEO/CFO/pengarah, masalah audit
- Amaran keuntungan, kerugian besar, isu going concern
- Merger, akuisisi, rights issue, private placement
- Keputusan kewangan suku tahun terkini

Jawab dalam Bahasa Melayu, format ini:

**Berita & risiko syarikat**
Apa yang kau jumpa, dengan tarikh. Kalau tiada apa-apa material,
tulis "Tiada berita material dijumpai" — jangan reka. Kalau sumber
tak pasti atau rumor forum, sebut "belum disahkan".

**Bolehkah model ini dipercayai?**
Nilai nombor kualiti di atas secara jujur. Ingat:
- Accuracy dekat 33% bermakna model tak belajar apa-apa
- Accuracy tinggi boleh menipu kalau kebanyakan hari adalah HOLD —
  model yang asyik jawab HOLD pun dapat markah tinggi
- R² negatif bermakna regressor lebih teruk dari meneka purata
- Kalau model kalah beli & simpan, signalnya tiada nilai praktikal
- Trade sedikit bermakna win rate mungkin nasib semata
Beri satu ayat kesimpulan: signal ini patut diambil serius atau tidak.

**Bacaan teknikal**
2-3 ayat tentang apa yang indikator tunjuk.

**Yang menyokong BUY**
Bullet pendek. Kalau tiada, tulis "Tiada isyarat kukuh."

**Yang menyokong SELL**
Bullet pendek. Kalau tiada, tulis "Tiada isyarat kukuh."

**Percanggahan**
Indikator yang bercanggah sesama sendiri, dengan model ML, atau
dengan berita. Contoh: teknikal bullish tapi ada siasatan berjalan,
atau model kata BUY tapi backtest tunjukkan ia kalah beli-simpan.
Kalau tiada, tulis "Indikator sejajar."

**Pandangan**
Satu perenggan. Nyatakan kecenderungan kau dan sebabnya. Kalau ada
risiko berita serius, kata terus walaupun teknikal nampak elok —
carta tak nampak fraud. Kalau nombor model lemah, kata terus yang
signal itu tak patut dipakai. Kalau gambaran tak jelas, cakap tak
jelas — jangan paksa jawapan.

Ringkas dan terus terang. Jangan tambah disclaimer, sistem dah ada."""


def get_ai_analysis(ticker, snapshot, prediction, trade_stats=None,
                    use_search=True, chart_summary=None):
    """
    Ask Gemini to interpret the technical picture.

    Requires GEMINI_API_KEY in the environment.
    Get one free at https://aistudio.google.com/apikey

    use_search: enable Google Search grounding for company news.
                Falls back to technical-only if the search quota
                is exhausted.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None, "Package tak dijumpai. Jalankan: pip install google-genai"

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        return None, "GEMINI_API_KEY tak di-set dalam environment"

    prompt = build_analysis_prompt(
        ticker, snapshot, prediction, trade_stats, chart_summary
    )

    def _extract(response):
        """Pull text plus any grounding sources out of a response."""
        text = response.text or ""
        sources = []
        try:
            for cand in response.candidates or []:
                meta = getattr(cand, 'grounding_metadata', None)
                if not meta:
                    continue
                for chunk in getattr(meta, 'grounding_chunks', []) or []:
                    web = getattr(chunk, 'web', None)
                    if web and getattr(web, 'uri', None):
                        title = getattr(web, 'title', '') or web.uri
                        pair = (title, web.uri)
                        if pair not in sources:
                            sources.append(pair)
        except Exception:
            pass

        if sources:
            text += "\n\n**Sumber**\n"
            for title, uri in sources[:8]:
                text += f"- [{title}]({uri})\n"

        return text

    client = genai.Client(api_key=api_key)

    # Attempt 1: with Google Search grounding.
    # Grounding has its own, much tighter free-tier quota than plain
    # generation, so it is the first thing to fail with 429.
    if use_search:
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            )
            return _extract(response), None
        except Exception as e:
            msg = str(e)
            is_quota = '429' in msg or 'RESOURCE_EXHAUSTED' in msg
            if not is_quota:
                return None, f"Panggilan API gagal: {msg[:200]}"
            # Fall through and retry without search.

    # Attempt 2: no search. Technical analysis only.
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        text = _extract(response)
        if use_search:
            text = (
                "> ⚠️ Kuota Google Search habis. Analisis ini berdasarkan "
                "indikator teknikal sahaja — bahagian berita tidak disahkan "
                "dengan carian sebenar, jadi abaikan.\n\n"
            ) + text
        return text, None
    except Exception as e:
        msg = str(e)
        if '429' in msg or 'RESOURCE_EXHAUSTED' in msg:
            return None, (
                "Kuota Gemini habis (429). Free tier ada had per minit "
                "dan per hari. Tunggu seminit dan cuba lagi, atau tukar "
                "GEMINI_MODEL ke 'gemini-2.5-flash-lite' yang ada kuota "
                "lebih longgar."
            )
        return None, f"Panggilan API gagal: {msg[:200]}"


def get_backtest_stats_for(ticker, horizon=1):
    """Recompute stats from the saved trades file."""
    trades = load_backtest_trades(ticker, horizon)
    if trades.empty:
        return None
    return calculate_backtest_stats(trades)


# Bursa Malaysia retail transaction costs, per side unless noted.
#
# These are typical online-broker rates. A backtest that ignores them
# will show a thin positive edge on strategies that lose money in
# practice -- which is the most common way a trading system fools its
# own author.
COST_CONFIG = {
    'brokerage_pct': 0.0042,   # 0.42% typical online rate
    'brokerage_min': 8.00,     # RM 8 minimum per side
    'stamp_duty_per_1000': 1.00,   # RM 1 per RM 1,000, capped
    'stamp_duty_cap': 1000.00,
    'clearing_pct': 0.0003,    # 0.03%, capped
    'clearing_cap': 1000.00,
}


def trade_cost(position_value):
    """
    Round-trip cost for one trade at a given position size, in RM.

    Brokerage has a floor, so small positions are punished hardest:
    a RM 2,000 trade pays the same RM 8 as a RM 1,900 one, which is
    0.4% before anything else.
    """
    c = COST_CONFIG

    brokerage = max(position_value * c['brokerage_pct'], c['brokerage_min'])
    stamp = min(
        np.ceil(position_value / 1000) * c['stamp_duty_per_1000'],
        c['stamp_duty_cap']
    )
    clearing = min(position_value * c['clearing_pct'], c['clearing_cap'])

    one_side = brokerage + stamp + clearing
    return one_side * 2   # buy and sell


def cost_adjusted_summary(summary, trade_stats, position_value):
    """
    Restate the model's edge after transaction costs.

    Buy-and-hold pays for exactly one round trip no matter how long it
    is held, so it is barely affected. The model pays per trade, which
    is where a thin edge disappears.
    """
    if not summary or 'error' in summary or not trade_stats:
        return None

    n = trade_stats.get('trades', 0)
    if n == 0:
        return None

    per_trade = trade_cost(position_value)
    per_trade_pct = per_trade / position_value * 100

    model_cost_pct = per_trade_pct * n
    hold_cost_pct = per_trade_pct   # one entry, one exit, that's all

    model_net = summary['model'] - model_cost_pct
    hold_net = summary['hold'] - hold_cost_pct

    return {
        'trades': n,
        'cost_per_trade_rm': per_trade,
        'cost_per_trade_pct': per_trade_pct,
        'model_gross': summary['model'],
        'model_cost': model_cost_pct,
        'model_net': model_net,
        'hold_net': hold_net,
        'edge_gross': summary['edge'],
        'edge_net': model_net - hold_net,
    }


def build_backtest_charts(ticker, horizon, price_df):
    """
    Two views of the same backtest.

    1. Price with entry/exit markers -- shows *where* the model acted,
       which is usually more revealing than any summary number. Buying
       near tops or selling near bottoms is visible instantly.

    2. Equity curve against buy-and-hold -- shows whether any of it
       was worth the effort.
    """
    trades = load_backtest_trades(ticker, horizon)

    if price_df is None:
        return None, None, {'error': 'Tiada data harga dimuatkan.'}

    if trades is None or trades.empty:
        return None, None, {
            'error': 'Fail backtest kosong — tiada trade direkodkan. '
                     'Model mungkin tak pernah bagi pasangan BUY→SELL '
                     'dalam tempoh ujian. Cuba horizon lain.'
        }

    needed = {'Entry Date', 'Exit Date', 'Entry Price', 'Exit Price', 'Return %'}
    if not needed.issubset(trades.columns):
        return None, None, {
            'error': f'Fail backtest format lama. Lajur ada: '
                     f'{list(trades.columns)}. Latih semula.'
        }

    t = trades.copy()
    t['Entry Date'] = pd.to_datetime(t['Entry Date'])
    t['Exit Date'] = pd.to_datetime(t['Exit Date'])

    start = t['Entry Date'].min()
    end = t['Exit Date'].max()

    window = price_df[
        (price_df['Date'] >= start) & (price_df['Date'] <= end)
    ]
    if len(window) < 2:
        return None, None, {
            'error': f'Tempoh trade terlalu pendek untuk dilukis '
                     f'({start.date()} → {end.date()}, {len(window)} bar).'
        }

    # --- Chart 1: price with markers ---
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(
        x=window['Date'], y=window['Close'],
        mode='lines', name='Harga',
        line=dict(color='#5f6caf', width=1.6)
    ))
    price_fig.add_trace(go.Scatter(
        x=t['Entry Date'], y=t['Entry Price'],
        mode='markers', name='Masuk (BUY)',
        marker=dict(symbol='triangle-up', size=11, color='#2ed573',
                    line=dict(width=1, color='#1e7e34'))
    ))
    price_fig.add_trace(go.Scatter(
        x=t['Exit Date'], y=t['Exit Price'],
        mode='markers', name='Keluar (SELL)',
        marker=dict(symbol='triangle-down', size=11, color='#ff4757',
                    line=dict(width=1, color='#a4161a'))
    ))
    price_fig.update_layout(
        height=380, margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title='Harga (RM)',
        legend=dict(orientation='h', y=1.12, x=0),
        hovermode='x unified'
    )

    # --- Chart 2: equity vs buy-and-hold ---
    r = t['Return %'].astype(float)
    model_equity = (1 + r / 100).cumprod()
    model_curve = pd.DataFrame({
        'Date': t['Exit Date'],
        'Value': (model_equity - 1) * 100
    })

    first_close = float(window['Close'].iloc[0])
    hold_curve = pd.DataFrame({
        'Date': window['Date'],
        'Value': (window['Close'] / first_close - 1) * 100
    })

    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(
        x=hold_curve['Date'], y=hold_curve['Value'],
        mode='lines', name='Beli & simpan',
        line=dict(color='#9aa0a6', width=1.6, dash='dot')
    ))
    eq_fig.add_trace(go.Scatter(
        x=model_curve['Date'], y=model_curve['Value'],
        mode='lines+markers', name='Ikut model',
        line=dict(color='#2ed573', width=2),
        marker=dict(size=5)
    ))
    eq_fig.add_hline(y=0, line=dict(color='#ced4da', width=1))
    eq_fig.update_layout(
        height=340, margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title='Pulangan terkumpul (%)',
        legend=dict(orientation='h', y=1.14, x=0),
        hovermode='x unified'
    )

    hold_total = float(hold_curve['Value'].iloc[-1])
    model_total = float(model_curve['Value'].iloc[-1])

    return price_fig, eq_fig, {
        'model': model_total,
        'hold': hold_total,
        'edge': model_total - hold_total,
        'start': start.date(),
        'end': end.date(),
    }


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.markdown("""
    <style>
        .main {
            padding-top: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("📈 Stock Trading Predictor")
st.markdown("**Multi-Stock Dashboard with XGBoost AI** | Bursa Malaysia")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    selected_stock = st.selectbox(
        "📊 Select Stock",
        list(STOCK_CONFIG.keys()),
        format_func=lambda x: f"{x} - {STOCK_CONFIG[x]['name']}"
    )
    
    st.markdown("---")
    selected_horizon = st.selectbox(
        "⏱️ Horizon ramalan",
        list(HORIZON_CONFIG.keys()),
        format_func=lambda h: HORIZON_CONFIG[h]['label'],
        help="Setiap horizon ada model sendiri. Pergerakan harian "
             "lebih banyak bunyi; horizon panjang selalunya lebih "
             "mudah diramal tapi trade jadi kurang."
    )
    st.caption(
        f"Threshold: ±{HORIZON_CONFIG[selected_horizon]['threshold']:.0f}% "
        f"dalam {selected_horizon} hari"
    )

    st.markdown("---")
    if st.button("🔄 Muat semula data"):
        fetch_history.clear()
        st.rerun()
    st.caption(
        "Data segar ditarik setiap kali kau tukar saham. "
        "Tekan butang di atas untuk paksa muat semula."
    )

    st.markdown("---")
    st.subheader("📋 Stock Info")
    if selected_stock in STOCK_CONFIG:
        stock_info = STOCK_CONFIG[selected_stock]
        st.write(f"**Name:** {stock_info['name']}")
        st.write(f"**Sector:** {stock_info['description']}")

# Main content
col1, col2, col3 = st.columns(3)

# Pull fresh data whenever the user switches stock.
#
# The cache still guards against Streamlit's rerun-on-every-widget
# behaviour -- ticking a checkbox or changing horizon reuses what was
# already fetched. Only an actual change of stock forces a new download.
if st.session_state.get('_last_stock') != selected_stock:
    fetch_history.clear()
    st.session_state['_last_stock'] = selected_stock

# Check if model exists
model_exists = check_model_exists(selected_stock, selected_horizon)

if not model_exists:
    st.warning(
        f"⚠️ Model {selected_stock} untuk horizon "
        f"{selected_horizon} hari belum ada. Melatih sekarang..."
    )

    progress_bar = st.progress(0)
    status_text = st.empty()

    success, message = train_model(
        selected_stock, selected_horizon, progress_bar, status_text
    )
    
    if success:
        st.success(message)
        st.rerun()
    else:
        st.error(f"❌ Training failed: {message}")
        st.stop()

# Get latest price
with st.spinner("📡 Fetching latest data..."):
    latest_data = get_latest_price(selected_stock)
    prediction = predict_signal(selected_stock, selected_horizon)

if latest_data and prediction:
    # Display metrics
    with col1:
        st.metric(
            "💰 Current Price",
            f"RM {latest_data['price']:.2f}",
            f"{latest_data['change']:+.2f}%",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "📊 High/Low",
            f"RM {latest_data['high']:.2f}",
            f"L: RM {latest_data['low']:.2f}"
        )
    
    with col3:
        st.metric(
            "📈 Volume",
            f"{latest_data['volume']:,.0f}",
            "shares"
        )
    
    st.markdown("---")

    # Is the saved model still working from recent data?
    freshness = get_model_freshness(
        selected_stock, selected_horizon, prediction['date']
    )

    if freshness:
        if freshness['stale']:
            fresh_col1, fresh_col2 = st.columns([3, 1])
            with fresh_col1:
                st.warning(
                    f"🕐 Model ketinggalan **{freshness['real_lag']} hari** "
                    f"(dilatih hingga {freshness['trained_until']}, data "
                    f"hingga {freshness['latest_data']}). Signal harian "
                    f"masih guna data terkini, tapi trade backtest di bawah "
                    f"adalah gambaran lama."
                )
            with fresh_col2:
                st.write("")
                if st.button("🔄 Latih semula", type="primary",
                             key=f"retrain_{selected_stock}_{selected_horizon}"):
                    mp, tp = get_model_paths(selected_stock, selected_horizon)
                    for p in (mp, tp):
                        if os.path.exists(p):
                            os.remove(p)
                    st.rerun()
        else:
            msg = (
                f"✅ Model terkini — dilatih hingga "
                f"{freshness['trained_until']}, data hingga "
                f"{freshness['latest_data']}"
            )
            if selected_horizon > 1:
                # Without this, the gap looks like staleness when it is
                # simply the future not having happened yet.
                msg += (
                    f" · jurang {freshness['gap_days']} hari adalah normal "
                    f"— {selected_horizon} baris terakhir tak boleh dilatih "
                    f"sebab harga {selected_horizon} hari ke depan belum wujud"
                )
            else:
                msg += f" ({freshness['gap_days']} hari beza)"
            st.caption(msg)

    st.markdown("---")

    # Prediction Section
    st.subheader("🤖 AI Prediction")
    
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    
    with pred_col1:
        signal_color = {
            'BUY': '🟢',
            'HOLD': '🟡',
            'SELL': '🔴'
        }
        st.metric(
            "Signal",
            f"{signal_color[prediction['signal']]} {prediction['signal']}",
            f"Confidence: {prediction['confidence']:.1f}%"
        )

    with pred_col2:
        if prediction.get('target_price') is not None:
            dp = 4 if prediction['price'] < 5 else 2
            st.metric(
                f"🎯 Sasaran ({selected_horizon}h)",
                f"RM {prediction['target_price']:.{dp}f}",
                f"{prediction['pred_return']:+.2f}%"
            )
        else:
            st.metric("🎯 Sasaran Harga", "—", "Latih semula")

    with pred_col3:
        r2 = prediction.get('oos_reg_r2')
        mae = prediction.get('oos_reg_mae')
        if r2 is not None:
            st.metric(
                "📉 Ketepatan Sasaran",
                f"R² {r2:.3f}",
                f"Purata tersasar {mae:.2f}pp"
            )

    # The regressor is the part most likely to mislead, so say so plainly.
    r2 = prediction.get('oos_reg_r2')
    if r2 is not None:
        if r2 < 0.05:
            st.warning(
                f"⚠️ R² regressor {r2:.3f} — hampir sifar. Sasaran harga "
                f"ini tidak lebih baik dari tekaan. Guna signal sahaja, "
                f"abaikan nombor sasaran."
            )
        elif r2 < 0.15:
            st.info(
                f"ℹ️ R² regressor {r2:.3f} — lemah. Anggap sasaran harga "
                f"sebagai arah kasar, bukan angka untuk letak order."
            )
    
    # Confidence breakdown
    st.markdown("#### Confidence Breakdown")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔴 SELL", f"{prediction['proba'][0]*100:.1f}%")
    with col2:
        st.metric("🟡 HOLD", f"{prediction['proba'][1]*100:.1f}%")
    with col3:
        st.metric("🟢 BUY", f"{prediction['proba'][2]*100:.1f}%")
    
    # Confidence chart
    fig = go.Figure(data=[
        go.Bar(
            y=['SELL', 'HOLD', 'BUY'],
            x=prediction['proba'] * 100,
            orientation='h',
            marker=dict(
                color=['#ff4757', '#ffa502', '#2ed573'],
            )
        )
    ])
    fig.update_layout(
        title="Prediction Confidence %",
        xaxis_title="Confidence (%)",
        height=300,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recommendation
    st.subheader("💡 Recommendation")
    
    if prediction['confidence'] > 75:
        conf_level = "🟢 STRONG"
    elif prediction['confidence'] > 60:
        conf_level = "🟡 MODERATE"
    else:
        conf_level = "🔴 WEAK"
    
    st.write(f"**Confidence Level:** {conf_level}")
    
    if prediction['signal'] == 'BUY' and prediction['confidence'] > 75:
        tp = prediction.get('target_price')
        tp_line = (
            f"- Sasaran model: RM {tp:.2f}"
            if tp else
            f"- Take Profit: RM {latest_data['price'] * 1.05:.2f} (+5%)"
        )
        st.success(f"""
        ✅ BUY Signal (Strong)
        - Entry: RM {latest_data['price']:.2f}
        - Stop Loss: RM {latest_data['price'] * 0.98:.2f} (-2%)
        {tp_line}
        """)
    elif prediction['signal'] == 'SELL' and prediction['confidence'] > 75:
        st.warning(f"""
        ⚠️ SELL Signal (Strong)
        - Exit: RM {latest_data['price']:.2f}
        - Don't short (risky)
        """)
    elif prediction['signal'] == 'HOLD':
        st.info("""
        ℹ️ HOLD Signal
        - Tiada tindakan diperlukan
        - Tunggu signal lebih jelas
        """)
    else:
        # Signal points somewhere, but not confidently enough to act on.
        # Saying "HOLD" here would misreport what the model produced.
        st.info(f"""
        ℹ️ {prediction['signal']} Signal — keyakinan terlalu rendah
        - Model condong ke {prediction['signal']} pada
          {prediction['confidence']:.1f}%, di bawah ambang 75%
        - Tiada tindakan dicadangkan
        - Tunggu signal lebih kukuh
        """)
    
    st.markdown("---")

    # Latest backtest trade: entry & exit price
    st.subheader("📊 Latest Backtest Trade")

    latest_trade = get_latest_trade_status(selected_stock, selected_horizon)
    all_trades = load_backtest_trades(selected_stock, selected_horizon)

    if latest_trade:
        # A stale-looking date usually means the model simply stopped
        # trading, not that the data is old. Say which, before the
        # numbers get misread.
        try:
            last_exit = pd.to_datetime(latest_trade['exit_date'])
            data_end = pd.to_datetime(str(prediction['date'])[:10])
            months_idle = (data_end - last_exit).days / 30.44
        except Exception:
            months_idle = 0

        st.caption(
            f"Trade ke-{len(all_trades)} daripada {len(all_trades)} — "
            f"ini yang terakhir model buat."
        )

        if months_idle >= 3:
            st.info(
                f"ℹ️ Model tidak buat sebarang trade selama "
                f"**{months_idle:.0f} bulan** selepas tarikh ini "
                f"(hingga {data_end.date()}). Ia kekal HOLD sepanjang "
                f"tempoh tersebut — bukan data yang lapuk, tapi model "
                f"yang tidak aktif. Untuk saham yang jarang bergerak "
                f"melebihi ±{prediction['threshold']:.0f}%, ini biasa. "
                f"Cuba horizon lebih pendek kalau nak lebih banyak signal."
            )

        # Penny stocks need more decimals or the numbers look inconsistent
        dp = 4 if latest_trade['entry_price'] < 5 else 2

        t1, t2, t3, t4 = st.columns(4)

        t1.metric(
            "📍 Entry Price",
            f"RM {latest_trade['entry_price']:.{dp}f}",
            str(latest_trade['entry_date'])[:10]
        )
        t2.metric(
            "🚪 Exit Price",
            f"RM {latest_trade['exit_price']:.{dp}f}",
            f"{str(latest_trade['exit_date'])[:10]} · {latest_trade['exit_type']}"
        )
        t3.metric(
            "💹 Return",
            f"{latest_trade['return_pct']:+.2f}%",
            latest_trade['result']
        )
        t4.metric(
            "📈 vs Current",
            f"RM {latest_data['price']:.{dp}f}",
            f"{((latest_data['price'] - latest_trade['entry_price']) / latest_trade['entry_price'] * 100):+.2f}% from entry"
        )

        if latest_trade['forced_exit']:
            st.caption(
                "⚠️ Trade ini ditutup sebab data habis, bukan sebab model bagi signal SELL. "
                "Model masih nak HOLD. Abaikan return ini bila nilai prestasi model."
            )
    else:
        st.info("ℹ️ No backtest trades recorded yet")

    st.markdown("---")

    # Visual backtest: where the model acted, and whether it paid off
    st.subheader("📈 Carta Backtest")

    price_df = load_csv_data(
        STOCK_CONFIG[selected_stock]['csv_path'], selected_stock
    )
    price_fig, eq_fig, summary = build_backtest_charts(
        selected_stock, selected_horizon, price_df
    )

    # Referenced by the confidence filter below. Kept in sync with the
    # cost panel's own input further down.
    position_value_default = 10000
    _base_stats = get_backtest_stats_for(selected_stock, selected_horizon)
    summary_trade_count = _base_stats['trades'] if _base_stats else 0

    if price_fig is not None:
        st.caption(
            f"Tempoh ujian {summary['start']} → {summary['end']}. "
            f"Segitiga hijau = model masuk, merah = model keluar."
        )
        st.plotly_chart(price_fig, use_container_width=True)

        # --- Live confidence filter ---
        _, oos_bundle = load_oos_bundle(selected_stock, selected_horizon)

        if oos_bundle is not None:
            st.markdown("**Tapis pada keyakinan**")
            st.caption(
                "Model masuk trade walaupun keyakinan hampir sama dengan "
                "tekaan rawak (33%). Setiap kemasukan kos ~1.1%, jadi "
                "menapis signal lemah adalah cara paling terus untuk "
                "kurangkan kos. Laras dan tengok kesannya — model tidak "
                "dilatih semula."
            )

            min_conf = st.slider(
                "Keyakinan minimum untuk masuk (%)",
                33, 80, 33, step=1,
                key=f"conf_{selected_stock}_{selected_horizon}"
            )

            if min_conf > 33:
                filt_trades = generate_backtest_trades(
                    oos_bundle['data'], oos_bundle['y_pred'],
                    oos_bundle['proba'], oos_bundle['split_idx'],
                    min_confidence=float(min_conf)
                )
                filt_stats = calculate_backtest_stats(filt_trades)

                if filt_stats['trades'] > 0:
                    filt_summary = {
                        'model': filt_stats['total_return'],
                        'hold': summary['hold'],
                        'edge': filt_stats['total_return'] - summary['hold'],
                    }
                    filt_net = cost_adjusted_summary(
                        filt_summary, filt_stats, position_value_default
                    )

                    f1, f2, f3, f4 = st.columns(4)
                    f1.metric(
                        "Trade", filt_stats['trades'],
                        f"dari {summary_trade_count}"
                    )
                    f2.metric("Win rate", f"{filt_stats['win_rate']:.1f}%")
                    f3.metric(
                        "Kasar", f"{filt_stats['total_return']:+.2f}%",
                        f"kelebihan {filt_summary['edge']:+.2f}%"
                    )
                    if filt_net:
                        f4.metric(
                            "Bersih selepas kos",
                            f"{filt_net['edge_net']:+.2f}%",
                            f"kos {filt_net['model_cost']:.1f}%"
                        )
                else:
                    st.info(
                        f"Tiada trade pada keyakinan ≥{min_conf}%. "
                        f"Model tidak pernah cukup yakin — itu sendiri "
                        f"satu jawapan."
                    )

        st.markdown("**Ikut model vs duduk diam**")
        st.plotly_chart(eq_fig, use_container_width=True)

        e1, e2, e3 = st.columns(3)
        e1.metric("🤖 Ikut model", f"{summary['model']:+.2f}%")
        e2.metric("💤 Beli & simpan", f"{summary['hold']:+.2f}%",
                  "tak buat apa-apa")
        e3.metric("📐 Kelebihan", f"{summary['edge']:+.2f}%",
                  "model tolak beli-simpan")

        # This is the comparison that decides whether the whole exercise
        # was worth running.
        if summary['edge'] < 0:
            st.error(
                f"❌ Model kalah {abs(summary['edge']):.1f}% pada strategi "
                f"paling malas. Beli dan simpan sahaja lebih menguntungkan "
                f"dalam tempoh ini."
            )
        elif summary['edge'] < 5:
            st.warning(
                f"⚠️ Model menang {summary['edge']:.1f}% sahaja — kelebihan "
                f"setipis ini perlu diuji terhadap kos. Lihat di bawah."
            )
        else:
            st.success(
                f"✅ Model mengatasi beli & simpan sebanyak "
                f"{summary['edge']:.1f}% (sebelum kos)."
            )

        # --- Cost reality check ---
        st.markdown("**Selepas kos transaksi**")
        st.caption(
            "Angka di atas kasar. Setiap trade bayar brokerage, duti "
            "setem dan clearing fee. Beli & simpan bayar sekali sahaja; "
            "model bayar setiap kali ia masuk dan keluar."
        )

        position_value = st.number_input(
            "Saiz posisi setiap trade (RM)",
            min_value=1000, max_value=500000, value=10000, step=1000,
            help="Brokerage ada minimum RM 8 sekali jalan, jadi posisi "
                 "kecil dihukum jauh lebih teruk secara peratusan."
        )

        cost_stats = get_backtest_stats_for(selected_stock, selected_horizon)
        net = cost_adjusted_summary(summary, cost_stats, position_value)

        if net:
            n1, n2, n3 = st.columns(3)
            n1.metric(
                "🤖 Model (bersih)",
                f"{net['model_net']:+.2f}%",
                f"kasar {net['model_gross']:+.2f}%"
            )
            n2.metric(
                "💤 Beli & simpan (bersih)",
                f"{net['hold_net']:+.2f}%",
                "1 trade sahaja"
            )
            n3.metric(
                "📐 Kelebihan bersih",
                f"{net['edge_net']:+.2f}%",
                f"kasar {net['edge_gross']:+.2f}%"
            )

            st.caption(
                f"Kos: RM {net['cost_per_trade_rm']:.2f} setiap trade "
                f"pergi-balik ({net['cost_per_trade_pct']:.2f}% pada posisi "
                f"RM {position_value:,}) × {net['trades']} trade = "
                f"{net['model_cost']:.2f}% ditelan."
            )

            if net['edge_net'] < 0 <= net['edge_gross']:
                st.error(
                    f"❌ Kelebihan kasar +{net['edge_gross']:.2f}% berubah "
                    f"jadi **{net['edge_net']:+.2f}%** selepas kos. Model "
                    f"ini rugi dalam realiti walaupun backtest nampak "
                    f"positif. Inilah sebab kos mesti dikira."
                )
            elif net['edge_net'] < 0:
                st.error(
                    f"❌ Kelebihan bersih {net['edge_net']:+.2f}%. "
                    f"Model rugi sebelum dan selepas kos."
                )
            elif net['edge_net'] < 3:
                st.warning(
                    f"⚠️ Kelebihan bersih {net['edge_net']:+.2f}% sahaja "
                    f"selepas kos. Terlalu tipis untuk bergantung padanya."
                )
            else:
                st.success(
                    f"✅ Kelebihan bersih {net['edge_net']:+.2f}% selepas "
                    f"kos. Ini yang berbaloi disiasat lebih lanjut."
                )
    else:
        reason = (summary or {}).get('error', 'Sebab tidak diketahui.')
        st.info(f"Carta tidak dapat dijana — {reason}")

    st.markdown("---")

    # Compare horizons that have already been trained
    st.subheader("⏱️ Perbandingan Horizon")

    rows = []
    for h in HORIZON_CONFIG:
        if not check_model_exists(selected_stock, h):
            rows.append({
                'Horizon': HORIZON_CONFIG[h]['label'],
                'Threshold': f"±{HORIZON_CONFIG[h]['threshold']:.0f}%",
                'Accuracy': '—',
                'Regressor R²': '—',
                'Trade': '—',
                'Win Rate': '—',
                'Total Return': 'belum dilatih',
            })
            continue

        try:
            mp, _ = get_model_paths(selected_stock, h)
            with open(mp, 'rb') as f:
                pkg = pkl.load(f)
            s = get_backtest_stats_for(selected_stock, h)

            rows.append({
                'Horizon': HORIZON_CONFIG[h]['label'],
                'Threshold': f"±{pkg.get('threshold', 0):.0f}%",
                'Accuracy': f"{pkg.get('oos_accuracy', 0) * 100:.1f}%",
                'Regressor R²': f"{pkg.get('oos_reg_r2', 0):.3f}",
                'Trade': str(s['trades']) if s else '0',
                'Win Rate': f"{s['win_rate']:.1f}%" if s else '—',
                'Total Return': f"{s['total_return']:+.1f}%" if s else '—',
            })
        except Exception:
            continue

    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                 hide_index=True)

    st.caption(
        "Tukar horizon di sidebar untuk melatih yang belum ada. "
        "Accuracy tinggi tak semestinya bagus — kalau kebanyakan hari "
        "adalah HOLD, model yang asyik jawab HOLD pun dapat markah "
        "tinggi. Tengok win rate dan total return."
    )

    # Long horizons produce overlapping forecast windows, so their trades
    # are less independent than the count suggests. Say so where the
    # number is actually being read.
    cur_stats = get_backtest_stats_for(selected_stock, selected_horizon)
    if cur_stats and cur_stats['trades'] > 0:
        n = cur_stats['trades']
        if n < 10:
            st.error(
                f"❌ Horizon {selected_horizon} hari hanya hasilkan {n} trade. "
                f"Terlalu sedikit untuk apa-apa kesimpulan — win rate pada "
                f"sampel sekecil ini hampir semuanya nasib."
            )
        elif n < 25:
            st.warning(
                f"⚠️ Hanya {n} trade pada horizon {selected_horizon} hari. "
                f"Layan win rate sebagai petunjuk kasar, bukan bukti."
            )

        if selected_horizon > 1:
            st.caption(
                f"ℹ️ Pada horizon {selected_horizon} hari, setiap ramalan "
                f"bertindih dengan {selected_horizon - 1} ramalan sebelumnya "
                f"— tetingkap masa berkongsi hari yang sama. Trade-trade ini "
                f"kurang bebas antara satu sama lain berbanding horizon 1 "
                f"hari, jadi perlukan lebih banyak trade sebelum boleh "
                f"percaya nombornya."
            )

    st.markdown("---")

    # AI second opinion
    st.subheader("🧠 Analisis AI")
    st.caption(
        "Gemini membaca indikator teknikal yang sama, cari berita syarikat "
        "melalui Google Search, dan bagi pandangan bebas. "
        "Ini tafsiran konteks, bukan model ramalan kedua."
    )

    use_search = st.checkbox(
        "Cari berita syarikat (guna kuota Google Search)",
        value=True,
        help="Matikan kalau selalu kena error kuota 429. "
             "Analisis akan guna indikator teknikal sahaja."
    )

    if st.button("Jalankan analisis AI", type="primary"):
        with st.spinner("Menganalisa & mencari berita..."):
            snapshot = build_indicator_snapshot(selected_stock)

            if snapshot is None:
                st.error("Tak dapat baca data indikator")
            else:
                stats = get_backtest_stats_for(selected_stock, selected_horizon)
                analysis, err = get_ai_analysis(
                    selected_stock, snapshot, prediction, stats,
                    use_search=use_search,
                    chart_summary=summary
                )

                if err:
                    st.error(err)
                    st.warning(
                        "Tak apa — salin prompt di bawah dan tampal ke "
                        "ChatGPT, Claude, atau Gemini web. Semua tu percuma "
                        "dan boleh cari berita sendiri."
                    )
                    st.session_state[f'prompt_{selected_stock}_h{selected_horizon}'] = (
                        build_analysis_prompt(
                            selected_stock, snapshot, prediction, stats,
                            summary
                        )
                    )
                else:
                    st.session_state[f'ai_{selected_stock}_h{selected_horizon}'] = analysis
                    # Keep the prompt around too, so it can be
                    # re-run elsewhere for a second opinion.
                    st.session_state[f'prompt_{selected_stock}_h{selected_horizon}'] = (
                        build_analysis_prompt(
                            selected_stock, snapshot, prediction, stats,
                            summary
                        )
                    )

    # Persist across reruns so the answer doesn't vanish
    cached = st.session_state.get(f'ai_{selected_stock}_h{selected_horizon}')
    if cached:
        st.markdown(cached)

    # Manual fallback: the same prompt, ready to paste anywhere
    cached_prompt = st.session_state.get(f'prompt_{selected_stock}_h{selected_horizon}')
    if cached_prompt:
        with st.expander("📋 Salin prompt untuk ChatGPT / Claude / lain"):
            st.caption(
                "Tekan ikon salin di penjuru kanan atas kotak. "
                "Tampal ke mana-mana chat AI yang boleh buat carian web."
            )
            st.code(cached_prompt, language=None)
    else:
        # Let the prompt be generated without spending an API call
        if st.button("Papar prompt sahaja (tanpa panggil API)"):
            snap = build_indicator_snapshot(selected_stock)
            if snap:
                st.session_state[f'prompt_{selected_stock}_h{selected_horizon}'] = (
                    build_analysis_prompt(
                        selected_stock, snap, prediction,
                        get_backtest_stats_for(selected_stock, selected_horizon),
                        summary
                    )
                )
                st.rerun()
            else:
                st.error("Tak dapat baca data indikator")

    st.markdown("---")
    
    # Info section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Stock Details")
        st.write(f"**Price:** RM {latest_data['price']:.2f}")
        st.write(f"**24h Change:** {latest_data['change']:+.2f}%")
        st.write(f"**Volume:** {latest_data['volume']:,.0f}")
        st.write(f"**Date:** {latest_data['date']}")

        # The model reads history; the price box reads the live quote.
        # If those disagree, the prediction is not about today.
        pred_date = str(prediction['date'])[:10]
        st.write(f"**Data ramalan hingga:** {pred_date}")
        if pred_date != latest_data['date']:
            st.caption(
                f"⚠️ Harga semasa dari {latest_data['date']} tapi indikator "
                f"dikira hingga {pred_date}. Yahoo belum siarkan penutup "
                f"terkini — ramalan ini untuk hari berikutnya selepas "
                f"{pred_date}."
            )
    
    with col2:
        st.subheader("🤖 Model Info")
        st.write(f"**Model:** XGBoost Classifier + Regressor")
        st.write(
            f"**OOS Accuracy:** "
            f"{prediction['oos_accuracy']:.1f}%"
        )
        if prediction.get('oos_reg_r2') is not None:
            st.write(
                f"**Regressor R²:** {prediction['oos_reg_r2']:.3f} "
                f"(pada % perubahan)"
            )
            st.write(
                f"**Regressor MAE:** "
                f"{prediction['oos_reg_mae']:.2f} mata peratusan"
            )
        st.write(
            f"**Target:** Pergerakan {prediction['horizon']} hari ke depan"
        )
        st.write(f"**Threshold:** ±{prediction['threshold']:.0f}%")
        st.write(f"**Entry:** Close price")
        st.write(f"**Exit:** SELL signal close")
        st.write(f"**Fees:** Not included")
        st.write(f"**Mode:** Long-only")
    
    st.markdown("---")
    
    # Disclaimer
    st.warning("""
    ⚠️ **DISCLAIMER**
    - This is NOT financial advice
    - Use at your own risk
    - Past performance ≠ future results
    - Always do your own research
    - Never invest money you can't afford to lose
    """)

else:
    st.error("❌ Could not fetch data. Please check CSV file path.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>📈 Stock Predictor v2.0 | Powered by XGBoost & Streamlit</p>
        <p>Bursa Malaysia Stocks | Real-time Predictions</p>
    </div>
""", unsafe_allow_html=True)