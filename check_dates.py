import pandas as pd

df = pd.read_csv('data/raw/1066.KL.csv', index_col=0, parse_dates=True)
df['prev_close'] = df['Close'].shift(1)
same = df[abs(df['Open'] - df['prev_close']) < 1e-6]

print(f"Jumlah hari Open == Close semalam: {len(same)} / {len(df)}")
print("\n20 tarikh terkini:\n")
print(same[['Open', 'High', 'Low', 'Close', 'prev_close']].tail(20).to_string())
