"""
Backtest baseline - peraturan teknikal mudah, kos penuh.

Tujuan: dapatkan SATU nombor - kadar menang yang boleh dicapai pada
        universe ni. Tanpa nombor tu, keputusan LSTM nanti tiada rujukan.

Peraturan (sengaja mudah - ini tanda aras, bukan strategi):
    A. RSI mean-reversion : beli bila RSI(14) < 30
    B. MA momentum        : beli bila Close naik silang MA20
    C. Breakout           : beli bila Close > tertinggi 20 hari
    D. Random             : masuk rawak (kawalan - patut ~50%)

Peraturan D penting. Kalau strategi A/B/C tak mengalahkan D dengan
margin melebihi ralat piawai, ia tiada edge.

Masuk : Close[t+1]   (isyarat dikira dari Close[t])
Keluar: Close[t+4]   (pegang 3 hari)

Usage:
    python baseline_backtest.py --capital 25000
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

CLEANDIR = "data/clean"
UNIVERSE = "data/final_universe.csv"
HOLD = 3
SEED = 42

STAMP_DUTY_PCT = 0.10
CLEARING_PCT = 0.03
BROKERAGE_FLAT = 8.0

TRAIN_END = "2021-12-31"
VAL_END = "2023-12-31"


def tick_size(p):
    if p < 1.00:
        return 0.005
    if p < 10.00:
        return 0.01
    if p < 100.00:
        return 0.02
    return 0.10


def cost_pct(price, capital):
    shares = capital / price
    brok = BROKERAGE_FLAT * 2
    stamp = capital * STAMP_DUTY_PCT / 100 * 2
    clear = capital * CLEARING_PCT / 100 * 2
    spread = shares * tick_size(price)
    return (brok + stamp + clear + spread) / capital * 100


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def signals(df, rng):
    c = df["Close"]
    out = {}

    r = rsi(c)
    out["A_rsi_oversold"] = (r < 30) & (r.shift(1) >= 30)

    ma20 = c.rolling(20).mean()
    out["B_ma_cross"] = (c > ma20) & (c.shift(1) <= ma20.shift(1))

    hi20 = c.rolling(20).max()
    out["C_breakout"] = c >= hi20

    # kawalan: kekerapan sama dengan purata tiga di atas
    freq = np.mean([out[k].mean() for k in out])
    out["D_random"] = pd.Series(rng.random(len(c)) < freq, index=c.index)

    return out


def evaluate(df, sig, capital):
    """Pulangkan senarai trade untuk satu isyarat."""
    c = df["Close"]
    entry = c.shift(-1)          # masuk Close[t+1]
    exit_ = c.shift(-1 - HOLD)   # keluar Close[t+1+HOLD]

    gross = (exit_ / entry - 1) * 100
    trades = []

    idx = df.index[sig.fillna(False).values]
    for dt in idx:
        e, x, g = entry.get(dt), exit_.get(dt), gross.get(dt)
        if pd.isna(e) or pd.isna(x) or pd.isna(g):
            continue
        cst = cost_pct(float(e), capital)
        trades.append({"date": dt, "gross_pct": g, "cost_pct": cst,
                       "net_pct": g - cst})
    return trades


def summarise(trades, label):
    if len(trades) < 20:
        return {"strategy": label, "n": len(trades)}
    t = pd.DataFrame(trades)
    n = len(t)
    win = (t["net_pct"] > 0).mean()
    se = np.sqrt(win * (1 - win) / n) * 100
    return {
        "strategy": label,
        "n": n,
        "win_rate_pct": round(win * 100, 1),
        "se_pp": round(se, 2),
        "ci_low": round(win * 100 - 1.96 * se, 1),
        "ci_high": round(win * 100 + 1.96 * se, 1),
        "avg_gross_pct": round(t["gross_pct"].mean(), 3),
        "avg_cost_pct": round(t["cost_pct"].mean(), 3),
        "avg_net_pct": round(t["net_pct"].mean(), 3),
        "total_net_pct": round(t["net_pct"].sum(), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=25000)
    ap.add_argument("--split", default="train",
                    choices=["train", "val", "test", "all"])
    a = ap.parse_args()

    if os.path.exists(UNIVERSE):
        tickers = pd.read_csv(UNIVERSE)["ticker"].tolist()
    else:
        tickers = [os.path.basename(f).replace(".csv", "")
                   for f in glob.glob(f"{CLEANDIR}/*.csv")
                   if not os.path.basename(f).startswith(("_", "IDX_"))]

    print(f"Universe : {len(tickers)} saham")
    print(f"Modal    : RM{a.capital:,.0f}")
    print(f"Split    : {a.split}")
    print(f"Pegang   : {HOLD} hari\n")

    rng = np.random.default_rng(SEED)
    by_strategy, buyhold = {}, []

    for t in tickers:
        p = f"{CLEANDIR}/{t}.csv"
        if not os.path.exists(p):
            print(f"  hilang: {t}")
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()

        if a.split == "train":
            df = df[df.index <= TRAIN_END]
        elif a.split == "val":
            df = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
        elif a.split == "test":
            df = df[df.index > VAL_END]
        if len(df) < 100:
            continue

        for name, s in signals(df, rng).items():
            by_strategy.setdefault(name, []).extend(
                evaluate(df, s, a.capital))

        # tanda aras beli-simpan atas tempoh sama
        c = df["Close"]
        buyhold.append((c.iloc[-1] / c.iloc[0] - 1) * 100)

    rows = [summarise(v, k) for k, v in sorted(by_strategy.items())]
    res = pd.DataFrame(rows)
    res.to_csv(f"data/baseline_{a.split}.csv", index=False)

    print(res.to_string(index=False))

    print("\n" + "=" * 74)
    print("BACA MACAM NI")
    print("=" * 74)

    if "win_rate_pct" in res.columns:
        rnd = res[res["strategy"] == "D_random"]
        if len(rnd):
            r = rnd.iloc[0]
            print(f"Kawalan rawak : {r['win_rate_pct']}%  "
                  f"(CI {r['ci_low']}-{r['ci_high']})")
            print("Mana-mana strategi yang CI-nya bertindih dengan rawak")
            print("belum terbukti ada edge.\n")

        real = res[(res["strategy"] != "D_random")
                   & res["win_rate_pct"].notna()]
        for _, r in real.iterrows():
            beats = r["ci_low"] > rnd.iloc[0]["win_rate_pct"] if len(rnd) else False
            mark = "MENGATASI RAWAK" if beats else "tidak berbeza dari rawak"
            print(f"{r['strategy']:<18} {r['win_rate_pct']:>5.1f}%  "
                  f"CI {r['ci_low']:.1f}-{r['ci_high']:.1f}   {mark}")

    if buyhold:
        print(f"\nBeli-dan-simpan sepanjang tempoh: "
              f"purata {np.mean(buyhold):+.1f}% setiap saham")
        print("Strategi apa pun kena mengalahkan ini, bukan sekadar")
        print("melebihi sifar.")

    print("\n" + "=" * 74)
    print("Kadar menang diperlukan (dari screening) : ~53-59% ikut saham")
    print("Bandingkan dengan nombor di atas. Kalau baseline jauh di")
    print("bawah, jurang yang LSTM kena tutup adalah besar.")


if __name__ == "__main__":
    main()
