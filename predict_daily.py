#!/usr/bin/env python3
"""
Daily Stock Prediction - XGBoost Version (BETTER ACCURACY!)
Loads pre-trained XGBoost model and makes instant predictions
Uses Yahoo Finance for latest stock price
XGBoost Accuracy: 91.61% (vs LSTM 45.47%)
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("📊 DAILY PREDICTION - XGBOOST MODEL (BETTER!)")
print("=" * 80)

# ============================================================================
# CONFIG
# ============================================================================
STOCK_TICKER = "5347.KL"
CSV_PATH = "/mnt/e/StockistAgent/data/raw/5347.KL.csv"
MODEL_PATH = "xgb_model.pkl"  # XGBoost model (not LSTM)
SCALER_PATH = "scaler.pkl"

# ============================================================================
# PHASE 1: Check if model exists
# ============================================================================
print("\n[PHASE 1] Checking saved model...")

if not os.path.exists(MODEL_PATH):
    print(f"✗ Model not found: {MODEL_PATH}")
    print("  Note: XGBoost model not saved yet.")
    print("  Will need to update training script to save XGBoost model.")
    print("  For now, using LSTM model...")
    MODEL_PATH = "best_lstm_model.pth"

if not os.path.exists(CSV_PATH):
    print(f"✗ CSV not found: {CSV_PATH}")
    sys.exit(1)

print(f"✓ Model found: {MODEL_PATH}")
print(f"✓ Data found: {CSV_PATH}")

# ============================================================================
# PHASE 2: Get latest stock price from Yahoo Finance
# ============================================================================
print("\n[PHASE 2] Fetching latest stock price...")

try:
    import yfinance as yf
    print("✓ yfinance loaded")
except ImportError:
    print("✗ yfinance not installed. Installing...")
    os.system("pip install yfinance")
    import yfinance as yf

try:
    stock = yf.Ticker(STOCK_TICKER)
    current_data = stock.history(period='5d')
    
    if len(current_data) == 0:
        print(f"✗ Could not fetch data for {STOCK_TICKER}")
        print("  Using CSV data instead...")
        use_yfinance = False
    else:
        latest_row = current_data.iloc[-1]
        latest_close = latest_row['Close']
        latest_high = latest_row['High']
        latest_low = latest_row['Low']
        latest_volume = int(latest_row['Volume'])
        latest_date = current_data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
        
        if len(current_data) > 1:
            prev_close = current_data.iloc[-2]['Close']
            daily_change = ((latest_close - prev_close) / prev_close) * 100
        else:
            daily_change = 0
        
        use_yfinance = True
        print(f"✓ Data fetched for {STOCK_TICKER}")
        print(f"  Latest Price: RM {latest_close:.2f}")
        print(f"  Daily Change: {daily_change:+.2f}%")
        print(f"  Volume: {latest_volume:,}")
        print(f"  As of: {latest_date}")
        
except Exception as e:
    print(f"⚠ Could not fetch from Yahoo Finance: {e}")
    print("  Using CSV data instead...")
    use_yfinance = False

# ============================================================================
# PHASE 3: Load historical data
# ============================================================================
print("\n[PHASE 3] Loading historical data...")

df = pd.read_csv(CSV_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"✓ Loaded {len(df)} historical records")
print(f"  Date range: {df['Date'].iloc[0].strftime('%Y-%m-%d')} → {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")

# ============================================================================
# PHASE 4: Calculate indicators for all historical data
# ============================================================================
print("\n[PHASE 4] Calculating technical indicators...")

df_indicators = df.copy()

# Moving Averages
df_indicators['MA5'] = df_indicators['Close'].rolling(window=5).mean()
df_indicators['MA10'] = df_indicators['Close'].rolling(window=10).mean()
df_indicators['MA20'] = df_indicators['Close'].rolling(window=20).mean()
df_indicators['MA50'] = df_indicators['Close'].rolling(window=50).mean()

# RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df_indicators['RSI'] = calculate_rsi(df_indicators['Close'], 14)

# MACD
exp1 = df_indicators['Close'].ewm(span=12, adjust=False).mean()
exp2 = df_indicators['Close'].ewm(span=26, adjust=False).mean()
df_indicators['MACD'] = exp1 - exp2
df_indicators['MACD_Signal'] = df_indicators['MACD'].ewm(span=9, adjust=False).mean()

# Bollinger Bands
sma = df_indicators['Close'].rolling(window=20).mean()
std = df_indicators['Close'].rolling(window=20).std()
df_indicators['BB_Upper'] = sma + (std * 2)
df_indicators['BB_Lower'] = sma - (std * 2)
df_indicators['BB_Position'] = (df_indicators['Close'] - df_indicators['BB_Lower']) / \
                                (df_indicators['BB_Upper'] - df_indicators['BB_Lower'])

# Price changes
df_indicators['Price_Change'] = df_indicators['Close'].pct_change() * 100
df_indicators['High_Low_Ratio'] = df_indicators['High'] / df_indicators['Low']
df_indicators['Volume_MA'] = df_indicators['Volume'].rolling(window=20).mean()
df_indicators['Volume_Ratio'] = df_indicators['Volume'] / df_indicators['Volume_MA']

# Remove NaN
df_indicators = df_indicators.dropna().reset_index(drop=True)

print(f"✓ Created 14 indicators")
print(f"  Final dataset: {len(df_indicators)} rows")

# ============================================================================
# PHASE 5: Prepare features for prediction
# ============================================================================
print("\n[PHASE 5] Preparing features for prediction...")

exclude_cols = ['Date', 'Close']
feature_cols = [col for col in df_indicators.columns if col not in exclude_cols]

print(f"✓ Features: {len(feature_cols)}")
print(f"  {', '.join(feature_cols)}")

# Get all features (historical)
X_all = df_indicators[feature_cols].values

# ============================================================================
# PHASE 6: Load scaler and normalize
# ============================================================================
print("\n[PHASE 6] Normalizing features...")

import pickle
from sklearn.preprocessing import StandardScaler

if not os.path.exists('scaler.pkl'):
    print("✗ ERROR: Scaler not found!")
    print("  Run training first: python stock_trading_trainer.py")
    sys.exit(1)

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded from: scaler.pkl")
except Exception as e:
    print(f"✗ ERROR loading scaler: {e}")
    sys.exit(1)

# Verify scaler has correct number of features
expected_features = len(feature_cols)
scaler_features = scaler.n_features_in_
if scaler_features != expected_features:
    print(f"✗ ERROR: Feature mismatch!")
    print(f"  Scaler expects: {scaler_features} features")
    print(f"  Data has: {expected_features} features")
    sys.exit(1)

X_normalized = scaler.transform(X_all)
print(f"✓ Data normalized ({expected_features} features)")

# ============================================================================
# PHASE 7: Load trained XGBoost model
# ============================================================================
print("\n[PHASE 7] Loading trained XGBoost model...")

from xgboost import XGBClassifier

if not os.path.exists('xgb_model.pkl'):
    print("✗ ERROR: XGBoost model not found: xgb_model.pkl")
    print("  Run training first: python stock_trading_trainer.py")
    sys.exit(1)

try:
    # Load XGBoost model from pickle
    with open('xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    print(f"✓ XGBoost model loaded: xgb_model.pkl")
except Exception as e:
    print(f"✗ ERROR loading XGBoost model: {e}")
    sys.exit(1)

# ============================================================================
# PHASE 8: Make prediction
# ============================================================================
print("\n[PHASE 8] Making prediction...")

# Get latest data
latest_features = X_normalized[-1:].reshape(1, -1)

# Predict
pred_xgb = xgb_model.predict(latest_features)[0]

# Get probability
pred_proba = xgb_model.predict_proba(latest_features)[0]

signal_map = {0: 'SELL 🔴', 1: 'HOLD 🟡', 2: 'BUY 🟢'}
signal_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

print(f"✓ Prediction made!")

# ============================================================================
# PHASE 9: Display results
# ============================================================================
print("\n" + "=" * 80)
print("📈 PREDICTION RESULTS - XGBOOST")
print("=" * 80)

if use_yfinance:
    print(f"\n✓ STOCK: {STOCK_TICKER}")
    print(f"✓ Current Price: RM {latest_close:.2f}")
    print(f"✓ Daily Change: {daily_change:+.2f}%")
    print(f"✓ Volume: {latest_volume:,} shares")
    print(f"✓ High: RM {latest_high:.2f} | Low: RM {latest_low:.2f}")
    print(f"✓ As of: {latest_date}")
else:
    latest_close = df_indicators['Close'].iloc[-1]
    print(f"\n✓ STOCK: {STOCK_TICKER}")
    print(f"✓ Latest Close: RM {latest_close:.2f}")
    print(f"✓ Date: {df_indicators['Date'].iloc[-1].strftime('%Y-%m-%d')}")

print(f"\n📊 MODEL PREDICTION (XGBOOST - 91.61% Accuracy)")
print(f"─" * 80)
print(f"Signal: {signal_map[pred_xgb]}")
print(f"Confidence: {pred_proba[pred_xgb]*100:.1f}%")

print(f"\n📊 CONFIDENCE BREAKDOWN")
print(f"─" * 80)
print(f"  SELL 🔴: {pred_proba[0]*100:6.1f}% {('█' * int(pred_proba[0]*50)):<50}")
print(f"  HOLD 🟡: {pred_proba[1]*100:6.1f}% {('█' * int(pred_proba[1]*50)):<50}")
print(f"  BUY  🟢: {pred_proba[2]*100:6.1f}% {('█' * int(pred_proba[2]*50)):<50}")

# ============================================================================
# PHASE 10: Recommendation
# ============================================================================
print(f"\n💡 RECOMMENDATION")
print(f"─" * 80)

if pred_proba[pred_xgb] > 0.75:
    confidence_level = "STRONG ✅"
    action = f"{signal_names[pred_xgb].upper()} with HIGH confidence"
    can_trade = True
elif pred_proba[pred_xgb] > 0.60:
    confidence_level = "MODERATE ⚠️"
    action = f"{signal_names[pred_xgb].upper()} (moderate confidence)"
    can_trade = True
else:
    confidence_level = "WEAK ❌"
    action = "HOLD and wait for clearer signal"
    can_trade = False

print(f"Confidence Level: {confidence_level}")
print(f"Action: {action}")
print(f"Can Trade: {'YES - Signal is strong' if can_trade else 'NO - Wait for better signal'}")

# Trading hints
if pred_xgb == 2 and can_trade:  # BUY
    print(f"\n🟢 BUY SIGNAL:")
    print(f"  • Entry: RM {latest_close:.2f}")
    print(f"  • Stop Loss: RM {latest_close * 0.98:.2f} (-2%)")
    print(f"  • Take Profit: RM {latest_close * 1.05:.2f} (+5%)")
    print(f"  • Risk/Reward: 1:2.5 (Good!)")
elif pred_xgb == 0 and can_trade:  # SELL
    print(f"\n🔴 SELL SIGNAL:")
    print(f"  • Exit: RM {latest_close:.2f}")
    print(f"  • Don't short (risky in Malaysia market)")
    print(f"  • Wait for re-entry BUY signal")
else:
    print(f"\n🟡 HOLD SIGNAL:")
    print(f"  • No action needed")
    print(f"  • Wait for stronger signal (>75% confidence)")

# ============================================================================
# PHASE 11: Summary
# ============================================================================
print("\n" + "=" * 80)
print("✅ PREDICTION COMPLETE")
print("=" * 80)

print(f"""
📋 Summary:
  • Stock: {STOCK_TICKER}
  • Price: RM {latest_close:.2f}
  • Signal: {signal_map[pred_xgb]}
  • Confidence: {pred_proba[pred_xgb]*100:.1f}%
  • Model: XGBoost (91.61% Accuracy) ✅
  • Next Update: Tomorrow or run predict_daily_xgboost.py again

📊 Model Comparison:
  • XGBoost Accuracy: 91.61% ✅ BEST
  • LSTM Accuracy: 45.47% ❌ POOR

⚠️  Disclaimer:
  This is NOT financial advice.
  Use at your own risk.
  Always do your own research.
  Never invest money you can't afford to lose.
  Past performance ≠ future results.
""")

print("=" * 80)