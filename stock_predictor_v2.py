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
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
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

def load_csv_data(csv_path, ticker):
    """
    Load CSV data, or download from Yahoo Finance using download_bursa logic if not found
    Uses yfinance with verification (similar to download_bursa.py)
    """
    
    # Try to load from CSV first
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            print(f"✅ Loaded from CSV: {csv_path}")
            return df
        except Exception as e:
            print(f"⚠️ Error reading CSV: {e}")
            return None
    
    # If CSV not found, download from Yahoo Finance (using download_bursa logic)
    print(f"📥 CSV not found. Downloading {ticker} from Yahoo Finance...")
    
    START = "2015-01-01"
    END = "2026-09-04"
    
    try:
        yf = load_yfinance()
        
        # Download data
        df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
        
        # Flatten MultiIndex columns if present (yfinance sometimes does this)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        if df is None or len(df) == 0:
            print(f"❌ No data found for {ticker}")
            return None
        
        # Clean data
        df = df.dropna(subset=['Close'])
        
        # Reset index to get Date as column
        df.reset_index(inplace=True)
        
        # Ensure correct column names
        df.columns = [col.lower() for col in df.columns]
        
        # Rename 'adjclose' or 'adj close' to 'close' if needed
        if 'adjclose' in df.columns:
            df = df.rename(columns={'adjclose': 'close'})
        elif 'adj close' in df.columns:
            df = df.rename(columns={'adj close': 'close'})
        
        # Keep only required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < 5:
            print(f"❌ Missing required columns for {ticker}")
            return None
        
        df = df[available_cols]
        
        # Rename to match expected format (lowercase)
        df = df.rename(columns=str.capitalize)
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"✅ Downloaded {len(df)} rows for {ticker} ({df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()})")
        
        # Save to CSV for future use
        try:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path, index=False)
            print(f"💾 Data saved to {csv_path}")
        except Exception as e:
            print(f"⚠️ Could not save CSV: {e}")
        
        return df
    
    except Exception as e:
        print(f"❌ Error downloading {ticker}: {str(e)[:60]}")
        return None

def create_indicators(df):
    """Create technical indicators"""
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
    
    return df_ind.dropna().reset_index(drop=True)

def check_model_exists(ticker):
    """Check if the new v2 model exists for ticker."""
    model_path, trades_path = get_model_paths(ticker)
    return os.path.exists(model_path) and os.path.exists(trades_path)

def get_model_paths(ticker):
    """Return versioned model paths."""
    prefix = ticker.replace('.', '_')
    return (
        f"{prefix}_xgb_model_v2.pkl",
        f"{prefix}_backtest_trades.pkl"
    )


def build_training_dataset(df_ind):
    """
    Build the supervised learning dataset.

    Target:
      - BUY  = next-day return > +1%
      - HOLD = next-day return between -1% and +1%
      - SELL = next-day return < -1%

    IMPORTANT:
      Next_Close / Daily_Return / Target are never used as features.
    """
    THRESHOLD = 1.0

    data = df_ind.copy()

    data['Next_Close'] = data['Close'].shift(-1)
    data['Daily_Return'] = (
        (data['Next_Close'] - data['Close'])
        / data['Close'] * 100
    )

    data['Target'] = 0
    data.loc[data['Daily_Return'] > THRESHOLD, 'Target'] = 1
    data.loc[data['Daily_Return'] < -THRESHOLD, 'Target'] = -1

    # Last row has no next-day return, so it cannot be used for training.
    data = data.iloc[:-1].copy().reset_index(drop=True)

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

    return data, X, y, feature_cols


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


def generate_backtest_trades(
    data,
    y_pred,
    proba,
    split_idx
):
    """
    Convert OOS BUY/HOLD/SELL predictions into long-only trades.

    Rules:
      BUY  when not in a position -> ENTRY
      HOLD while in a position    -> keep holding
      SELL while in a position    -> EXIT

    Prices are CLOSE prices. No transaction fees are included.
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
                'Result': 'WIN' if return_pct > 0 else 'LOSS'
            })

            in_position = False
            entry_price = None
            entry_date = None
            entry_confidence = None

    # Close an open trade at the last available test close.
    if in_position and len(test_data) > 0:
        row = test_data.iloc[-1]
        exit_price = float(row['Close'])
        exit_date = row['Date']

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
            'Result': 'WIN' if return_pct > 0 else 'LOSS',
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


def train_model(ticker, progress_bar=None, status_text=None):
    """
    Train XGBoost model for a stock.

    Two models are used:
      1. OOS model: train on first 80%, evaluate on final 20%.
      2. Final model: retrain on 100% of historical data for live prediction.

    The OOS model is used ONLY to calculate honest historical
    signal/trade performance.
    """
    try:
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
        data, X, y, feature_cols = build_training_dataset(df_ind)

        if len(X) < 200:
            return False, "Not enough data for training"

        if progress_bar:
            progress_bar.progress(35)

        # ------------------------------------------------------------------
        # Chronological 80/20 split
        # ------------------------------------------------------------------
        split_idx = int(len(X) * 0.80)

        X_train_raw = X[:split_idx]
        y_train = y[:split_idx]

        X_test_raw = X[split_idx:]
        y_test = y[split_idx:]

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
            eval_set=[(X_test, y_test)],
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

        # ------------------------------------------------------------------
        # Save model package.
        #
        # Feature names are saved together with the model so prediction
        # always uses EXACTLY the same columns as training.
        # ------------------------------------------------------------------
        model_path, trades_path = get_model_paths(ticker)

        model_package = {
            'model': final_model,
            'scaler': final_scaler,
            'feature_cols': feature_cols,
            'target_mapping': {
                0: 'SELL',
                1: 'HOLD',
                2: 'BUY'
            },
            'threshold': 1.0,
            'horizon_days': 1,
            'oos_accuracy': accuracy,
            'oos_split_idx': split_idx,
            'trained_until': str(data['Date'].iloc[-1])
        }

        with open(model_path, 'wb') as f:
            pkl.dump(model_package, f)

        # Save OOS trades separately.
        with open(trades_path, 'wb') as f:
            pkl.dump(trades, f)

        if progress_bar:
            progress_bar.progress(100)

        return True, (
            f"✅ Trained! "
            f"OOS Accuracy: {accuracy * 100:.1f}% | "
            f"Trades: {stats['trades']} | "
            f"Win Rate: {stats['win_rate']:.1f}% | "
            f"Total Return: {stats['total_return']:+.2f}%"
        )

    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def predict_signal(ticker):
    """Make the latest live prediction for a ticker."""
    try:
        csv_path = STOCK_CONFIG[ticker]['csv_path']
        df = load_csv_data(csv_path, ticker)

        if df is None:
            return None

        df_ind = create_indicators(df)

        model_path, _ = get_model_paths(ticker)

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
            'price': float(df_ind['Close'].iloc[-1]),
            'date': df_ind['Date'].iloc[-1],
            'oos_accuracy': float(
                model_package.get('oos_accuracy', 0) * 100
            )
        }

    except Exception as e:
        print(f"Prediction error for {ticker}: {e}")
        return None


def load_backtest_trades(ticker):
    """Load previously generated OOS trade history."""
    try:
        _, trades_path = get_model_paths(ticker)

        if not os.path.exists(trades_path):
            return pd.DataFrame()

        with open(trades_path, 'rb') as f:
            trades = pkl.load(f)

        if trades is None:
            return pd.DataFrame()

        return trades

    except Exception as e:
        print(f"Backtest load error for {ticker}: {e}")
        return pd.DataFrame()


def get_latest_trade_status(ticker):
    """
    Return the latest open/closed trade state from OOS backtest.

    This is historical OOS information, not a new live trade.
    """
    trades = load_backtest_trades(ticker)

    if trades.empty:
        return None

    latest = trades.iloc[-1]

    return {
        'entry_date': latest.get('Entry Date'),
        'entry_price': float(latest.get('Entry Price')),
        'exit_date': latest.get('Exit Date'),
        'exit_price': float(latest.get('Exit Price')),
        'return_pct': float(latest.get('Return %')),
        'result': latest.get('Result', '')
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
    st.subheader("📋 Stock Info")
    if selected_stock in STOCK_CONFIG:
        stock_info = STOCK_CONFIG[selected_stock]
        st.write(f"**Name:** {stock_info['name']}")
        st.write(f"**Sector:** {stock_info['description']}")

# Main content
col1, col2, col3 = st.columns(3)

# Check if model exists
model_exists = check_model_exists(selected_stock)

if not model_exists:
    st.warning(f"⚠️ Model for {selected_stock} not found. Training now...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    success, message = train_model(selected_stock, progress_bar, status_text)
    
    if success:
        st.success(message)
        st.rerun()
    else:
        st.error(f"❌ Training failed: {message}")
        st.stop()

# Get latest price
with st.spinner("📡 Fetching latest data..."):
    latest_data = get_latest_price(selected_stock)
    prediction = predict_signal(selected_stock)

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
        st.success(f"""
        ✅ BUY Signal (Strong)
        - Entry: RM {latest_data['price']:.2f}
        - Stop Loss: RM {latest_data['price'] * 0.98:.2f} (-2%)
        - Take Profit: RM {latest_data['price'] * 1.05:.2f} (+5%)
        """)
    elif prediction['signal'] == 'SELL' and prediction['confidence'] > 75:
        st.warning(f"""
        ⚠️ SELL Signal (Strong)
        - Exit: RM {latest_data['price']:.2f}
        - Don't short (risky)
        """)
    else:
        st.info(f"""
        ℹ️ HOLD Signal
        - No action needed
        - Wait for stronger signal
        """)
    
    st.markdown("---")
    
    # Info section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Stock Details")
        st.write(f"**Price:** RM {latest_data['price']:.2f}")
        st.write(f"**24h Change:** {latest_data['change']:+.2f}%")
        st.write(f"**Volume:** {latest_data['volume']:,.0f}")
        st.write(f"**Date:** {latest_data['date']}")
    
    with col2:
        st.subheader("🤖 Model Info")
        st.write(f"**Model:** XGBoost Classifier")
        st.write(
            f"**OOS Accuracy:** "
            f"{prediction['oos_accuracy']:.1f}%"
        )
        st.write(f"**Target:** Next-day movement")
        st.write(f"**Threshold:** ±1.0%")
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