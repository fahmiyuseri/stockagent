#!/usr/bin/env python3
"""
Stock Trading Model Trainer - Complete Pipeline
Bursa Malaysia Stock Prediction (XGBoost + LSTM)
RTX 5070 Ti GPU Optimized
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("📈 STOCK TRADING MODEL TRAINER - COMPLETE PIPELINE")
print("=" * 80)

# ============================================================================
# PHASE 1: CHECK ENVIRONMENT & INSTALL DEPENDENCIES
# ============================================================================
print("\n[PHASE 1] Checking environment...")

try:
    import torch
    print(f"✓ PyTorch installed: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ CUDA not available - CPU mode (slower)")
except ImportError:
    print("✗ PyTorch not installed. Installing...")
    os.system("pip install torch --index-url https://download.pytorch.org/whl/cu118")

try:
    import xgboost
    print(f"✓ XGBoost installed: {xgboost.__version__}")
except ImportError:
    print("✗ XGBoost not installed. Installing...")
    os.system("pip install xgboost")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    print("✓ Scikit-learn installed")
except ImportError:
    print("✗ Scikit-learn not installed. Installing...")
    os.system("pip install scikit-learn")

try:
    import ta
    print("✓ TA (Technical Analysis) installed")
except ImportError:
    print("✗ TA not installed. Installing...")
    os.system("pip install ta")

# ============================================================================
# PHASE 2: LOAD & EXPLORE DATA
# ============================================================================
print("\n[PHASE 2] Loading & exploring data...")

CSV_PATH = "/mnt/e/StockistAgent/data/raw/1155.KL.csv"

# Try to find the file
if not os.path.exists(CSV_PATH):
    print("✗ CSV file not found!")
    print(f"  Expected: {CSV_PATH}")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print(f"✓ Loaded {len(df)} rows from {CSV_PATH}")
print(f"  Columns: {', '.join(df.columns.tolist())}")
print(f"  Date range: {df['Date'].iloc[0]} → {df['Date'].iloc[-1]}")

# Data cleaning
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Remove zero volume (data quality issue)
df_clean = df[df['Volume'] > 0].reset_index(drop=True)
removed = len(df) - len(df_clean)
if removed > 0:
    print(f"  Removed {removed} rows with zero volume")
    df = df_clean

print(f"\n✓ Data Summary:")
print(f"  Shape: {df.shape}")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  Price range: RM {df['Close'].min():.2f} → RM {df['Close'].max():.2f}")
print(f"  Avg Volume: {df['Volume'].mean():,.0f} shares")

# ============================================================================
# PHASE 3: FEATURE ENGINEERING
# ============================================================================
print("\n[PHASE 3] Creating technical indicators...")

# Add all technical indicators
df_features = df.copy()

# Moving Averages
df_features['MA5'] = df_features['Close'].rolling(window=5).mean()
df_features['MA10'] = df_features['Close'].rolling(window=10).mean()
df_features['MA20'] = df_features['Close'].rolling(window=20).mean()
df_features['MA50'] = df_features['Close'].rolling(window=50).mean()

# RSI (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df_features['RSI'] = calculate_rsi(df_features['Close'], 14)

# MACD
exp1 = df_features['Close'].ewm(span=12, adjust=False).mean()
exp2 = df_features['Close'].ewm(span=26, adjust=False).mean()
df_features['MACD'] = exp1 - exp2
df_features['MACD_Signal'] = df_features['MACD'].ewm(span=9, adjust=False).mean()

# Bollinger Bands
sma = df_features['Close'].rolling(window=20).mean()
std = df_features['Close'].rolling(window=20).std()
df_features['BB_Upper'] = sma + (std * 2)
df_features['BB_Lower'] = sma - (std * 2)
df_features['BB_Position'] = (df_features['Close'] - df_features['BB_Lower']) / \
                              (df_features['BB_Upper'] - df_features['BB_Lower'])

# Price changes
df_features['Price_Change'] = df_features['Close'].pct_change() * 100
df_features['High_Low_Ratio'] = df_features['High'] / df_features['Low']
df_features['Volume_MA'] = df_features['Volume'].rolling(window=20).mean()
df_features['Volume_Ratio'] = df_features['Volume'] / df_features['Volume_MA']

# Remove NaN rows
df_features = df_features.dropna().reset_index(drop=True)

print(f"✓ Created {len(df_features.columns) - 6} indicators")
print(f"  Final dataset: {len(df_features)} rows (removed {len(df) - len(df_features)} NaN)")

# ============================================================================
# PHASE 3B: CREATE TARGET VARIABLE
# ============================================================================
print("\n[PHASE 3B] Creating target variable (BUY/SELL/HOLD)...")

# Calculate next day's return
df_features['Next_Close'] = df_features['Close'].shift(-1)
df_features['Daily_Return'] = ((df_features['Next_Close'] - df_features['Close']) / df_features['Close'] * 100)

# Define signal: BUY (+1), HOLD (0), SELL (-1)
THRESHOLD = 2.0  # 2% threshold

df_features['Target'] = 0  # Default: HOLD

# BUY signal: Next close > current close + threshold
df_features.loc[df_features['Daily_Return'] > THRESHOLD, 'Target'] = 1

# SELL signal: Next close < current close - threshold
df_features.loc[df_features['Daily_Return'] < -THRESHOLD, 'Target'] = -1

# Remove last row (no next day data)
df_features = df_features[:-1].reset_index(drop=True)

signal_counts = df_features['Target'].value_counts().sort_index()
print(f"✓ Signal distribution (BEFORE encoding):")
print(f"  BUY (+1):  {signal_counts.get(1, 0):4d} ({signal_counts.get(1, 0)/len(df_features)*100:5.1f}%)")
print(f"  HOLD (0):  {signal_counts.get(0, 0):4d} ({signal_counts.get(0, 0)/len(df_features)*100:5.1f}%)")
print(f"  SELL (-1): {signal_counts.get(-1, 0):4d} ({signal_counts.get(-1, 0)/len(df_features)*100:5.1f}%)")

# Map target values to [0, 1, 2] for XGBoost/LSTM compatibility
# SELL (-1) → 0, HOLD (0) → 1, BUY (1) → 2
target_mapping = {-1: 0, 0: 1, 1: 2}
df_features['Target'] = df_features['Target'].map(target_mapping).astype(int)

# Verify mapping worked
unique_targets = sorted(df_features['Target'].unique())
print(f"✓ Target encoded to {unique_targets}")

if set(unique_targets) != {0, 1, 2}:
    print("⚠ Warning: Target values not properly encoded!")
    print(f"   Got: {unique_targets}, Expected: [0, 1, 2]")

# ============================================================================
# PHASE 4: PREPARE DATA FOR ML
# ============================================================================
print("\n[PHASE 4] Preparing data for machine learning...")

# Select features (exclude date, price, target)
exclude_cols = ['Date', 'Close', 'Next_Close', 'Target', 'Daily_Return']
feature_cols = [col for col in df_features.columns if col not in exclude_cols]

X = df_features[feature_cols].values
y = df_features['Target'].values.astype(int)  # Use encoded target [0, 1, 2]

# Verify classes are correct
unique_y = np.unique(y)
print(f"✓ Features selected: {len(feature_cols)}")
print(f"  Shape: X={X.shape}, y={y.shape}")
print(f"  Target classes: {unique_y}")

if not set(unique_y).issubset({0, 1, 2}):
    print("✗ ERROR: Target values not properly encoded!")
    print(f"   Got: {unique_y}, Expected: subset of [0, 1, 2]")
    sys.exit(1)

# Normalize features
from sklearn.preprocessing import StandardScaler
import pickle

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for daily predictions
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"✓ Data normalized (StandardScaler)")
print(f"✓ Scaler saved: scaler.pkl")

# Train/Test split (80/20, time-series aware)
split_idx = int(len(X_scaled) * 0.80)
X_train = X_scaled[:split_idx]
y_train = y[:split_idx]
X_test = X_scaled[split_idx:]
y_test = y[split_idx:]

print(f"✓ Train/Test split:")
print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X_scaled)*100:.1f}%)")
print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X_scaled)*100:.1f}%)")

# ============================================================================
# PHASE 5A: TRAIN XGBOOST
# ============================================================================
print("\n[PHASE 5A] Training XGBoost model...")
print("  (Estimated time: 1-2 minutes)")

from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist',  # hist works for both CPU and GPU
    device='cuda' if torch.cuda.is_available() else 'cpu',
    eval_metric='mlogloss',
    verbosity=0
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"\n✓ XGBoost trained successfully!")
print(f"  Accuracy: {xgb_accuracy*100:.2f}%")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_xgb, 
                           target_names=['SELL', 'HOLD', 'BUY'],
                           digits=3))

# ============================================================================
# PHASE 5B: TRAIN LSTM
# ============================================================================
print("\n[PHASE 5B] Training LSTM model...")
print("  (Estimated time: 5-15 minutes on GPU)")

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

# Create sequences
def create_sequences(X, y, seq_length=30):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

seq_length = 30
X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

print(f"  Sequences created:")
print(f"    Train: {X_train_seq.shape}")
print(f"    Test:  {X_test_seq.shape}")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
y_train_tensor = torch.LongTensor(y_train_seq).to(device)
X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
y_test_tensor = torch.LongTensor(y_test_seq).to(device)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.2
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Initialize
input_size = X_train_seq.shape[2]
lstm_model = LSTMModel(input_size, hidden_size=64, num_layers=2, num_classes=3).to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
epochs = 100
patience = 15
best_loss = float('inf')
patience_counter = 0

print(f"  Training started (max {epochs} epochs with early stopping)...")

# Training loop
for epoch in range(epochs):
    lstm_model.train()
    train_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = lstm_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    lstm_model.eval()
    with torch.no_grad():
        val_outputs = lstm_model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)
    
    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        torch.save(lstm_model.state_dict(), 'best_lstm_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
    
    if (epoch + 1) % 20 == 0:
        print(f"    Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}")

# Load best model
lstm_model.load_state_dict(torch.load('best_lstm_model.pth'))

# Test LSTM
lstm_model.eval()
with torch.no_grad():
    y_pred_lstm_tensor = lstm_model(X_test_tensor)
    y_pred_lstm = torch.argmax(y_pred_lstm_tensor, dim=1).cpu().numpy()

lstm_accuracy = accuracy_score(y_test_seq, y_pred_lstm)
print(f"\n✓ LSTM trained successfully!")
print(f"  Accuracy: {lstm_accuracy*100:.2f}%")
print(f"\n  Classification Report:")
print(classification_report(y_test_seq, y_pred_lstm,
                           target_names=['SELL', 'HOLD', 'BUY'],
                           digits=3))

# ============================================================================
# PHASE 6: COMPARE MODELS
# ============================================================================
print("\n[PHASE 6] Model Comparison")
print("=" * 60)

models = ['XGBoost', 'LSTM']
accuracies = [xgb_accuracy * 100, lstm_accuracy * 100]

print(f"\n  {'Model':<15} {'Accuracy':<15} {'Training Time':<15}")
print("  " + "-" * 45)
print(f"  {'XGBoost':<15} {xgb_accuracy*100:>6.2f}%{'':8} {'~2 min':>15}")
print(f"  {'LSTM':<15} {lstm_accuracy*100:>6.2f}%{'':8} {'~15 min':>15}")
print()

best_idx = np.argmax(accuracies)
best_model_name = models[best_idx]
best_accuracy = accuracies[best_idx]

print(f"  ✓ BEST MODEL: {best_model_name} ({best_accuracy:.2f}%)")

# Save comparison chart
plt.figure(figsize=(10, 6))
colors = ['#FF6B6B' if i != best_idx else '#4ECDC4' for i in range(len(models))]
bars = plt.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylim([50, 75])

for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n  📊 Comparison chart saved: model_comparison.png")

# ============================================================================
# PHASE 7: MAKE PREDICTIONS ON LATEST DATA
# ============================================================================
print("\n[PHASE 7] Making predictions on latest data...")

# Latest data for prediction
latest_close = df_features['Close'].iloc[-1]

# XGBoost prediction (use last row - it's already correct shape)
try:
    pred_xgb = xgb_model.predict(X_test[-1:])[0]  # X_test[-1:] is already (1, 18)
except Exception as e:
    print(f"  ✗ XGBoost prediction error: {e}")
    print(f"    X_test shape: {X_test.shape}, X_test[-1:] shape: {X_test[-1:].shape}")
    pred_xgb = 1  # Default to HOLD

# LSTM prediction (use last sequence from test set which is most recent)
if len(X_test_seq) > 0:
    latest_seq = torch.FloatTensor(X_test_seq[-1:]).to(device)
else:
    print("⚠ Warning: No test sequences available for LSTM prediction")
    latest_seq = None

if latest_seq is not None:
    lstm_model.eval()
    with torch.no_grad():
        pred_lstm_logits = lstm_model(latest_seq)
        pred_lstm = torch.argmax(pred_lstm_logits, dim=1).item()

    # Get probabilities
    with torch.no_grad():
        probs = torch.softmax(pred_lstm_logits, dim=1)[0].cpu().numpy()
else:
    # Fallback to XGBoost prediction if LSTM sequence unavailable
    pred_lstm = pred_xgb
    probs = np.array([0.33, 0.33, 0.34])

# Map encoded labels [0, 1, 2] back to signals
# 0 = SELL (originally -1), 1 = HOLD (originally 0), 2 = BUY (originally 1)
signal_map = {0: 'SELL 🔴', 1: 'HOLD 🟡', 2: 'BUY 🟢'}
signal_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

print(f"\n✓ Current Stock Price: RM {latest_close:.2f}")
print(f"✓ Latest Trading Date: {df_features['Date'].iloc[-1].strftime('%Y-%m-%d')}")

print(f"\n  XGBoost Signal: {signal_map[pred_xgb]}")
print(f"  LSTM Signal:    {signal_map[pred_lstm]}")
print(f"    Confidence:")
print(f"      SELL:  {probs[0]*100:5.1f}%")
print(f"      HOLD:  {probs[1]*100:5.1f}%")
print(f"      BUY:   {probs[2]*100:5.1f}%")

# Ensemble prediction (average of both models' predictions)
# Coded as: 0=SELL, 1=HOLD, 2=BUY
try:
    ensemble_pred = int(np.round((pred_xgb + pred_lstm) / 2))
    ensemble_pred = np.clip(ensemble_pred, 0, 2)  # Ensure in range [0, 1, 2]
    print(f"\n  Ensemble Signal: {signal_map[ensemble_pred]}")
except Exception as e:
    print(f"\n  ⚠ Ensemble prediction failed: {e}")
    ensemble_pred = 1  # Default to HOLD

# ============================================================================
# PHASE 8: SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("📊 TRAINING COMPLETE - SUMMARY")
print("=" * 80)

print(f"""
✓ Dataset: {len(df_features)} trading days ({df_features['Date'].iloc[0].strftime('%Y-%m-%d')} → {df_features['Date'].iloc[-1].strftime('%Y-%m-%d')})
✓ Features: {len(feature_cols)} technical indicators
✓ Train/Test: {len(X_train)}/{len(X_test)} samples

MODEL RESULTS:
  • XGBoost Accuracy: {xgb_accuracy*100:.2f}%
  • LSTM Accuracy:    {lstm_accuracy*100:.2f}%
  • Best Model:       {best_model_name}

CURRENT RECOMMENDATION:
  • Signal: {signal_map[pred_lstm]}
  • Price:  RM {latest_close:.2f}
  • Action: {signal_names[pred_lstm]} position

NEXT STEPS:
  1. Review the predictions and signals
  2. If accuracy < 60%, try adding correlated stocks (MAYBANK, CIMB, RHB)
  3. Backtest strategies on historical data
  4. Use LSTM predictions for live trading signals

FILES CREATED:
  • model_comparison.png - Accuracy comparison chart
  • best_lstm_model.pth - Saved LSTM weights
""")

print("=" * 80)
print("✅ TRAINING PIPELINE COMPLETE!")
print("=" * 80)

# Save summary to file
with open('training_summary.txt', 'w') as f:
    f.write("STOCK TRADING MODEL TRAINING SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: {CSV_PATH}\n")
    f.write(f"Total Samples: {len(df_features)}\n")
    f.write(f"Features: {len(feature_cols)}\n\n")
    f.write(f"XGBoost Accuracy: {xgb_accuracy*100:.2f}%\n")
    f.write(f"LSTM Accuracy: {lstm_accuracy*100:.2f}%\n")
    f.write(f"Best Model: {best_model_name}\n\n")
    f.write(f"Latest Signal (LSTM): {signal_names[pred_lstm]}\n")
    f.write(f"Current Price: RM {latest_close:.2f}\n")

print("\n✓ Summary saved: training_summary.txt")
