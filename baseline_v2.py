"""
Backtest baseline v2 - semua bug dibetulkan.

PEMBETULAN:
 1. Definisi kadar menang dipisahkan:
      direction_rate = P(gross > 0)      <- banding dengan req_winrate
      net_win_rate   = P(net > 0)        <- keuntungan sebenar
    v1 mencampur kedua-duanya dan membandingkan skala berbeza.

 2. Ujian statistik dua-hala:
      v1 hanya uji "lebih baik dari rawak", jadi MA cross yang
      signifikan LEBIH TERUK dilabel "tidak berbeza".
    Guna ujian-z dua proporsi.

 3. Kos dikira atas harga TAK DISELARASKAN:
      auto_adjust menurunkan harga sejarah, jadi tick% melambung.
      Script turunkan Close mentah sekali, cache, guna untuk kos.
      Pulangan tetap guna harga diselaraskan.

 4. Median dilapor bersama purata (outlier mendominasi purata).

 5. Semakan survivorship: universe vs indeks KLCI.

Usage:
    python baseline_v2.py --capital 25000 --split train
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import stats

CLEANDIR = "data/clean"
UNADJDIR = "data/unadj"
UNIVERSE = "data/final_universe.csv"
HOLD = 3
SEED = 42

STAMP_DUTY_PCT = 0.10
CLEARING_PCT = 0.03
BROKERAGE_FLAT = 8.0

TRAIN_END = "2021-12-31"
VAL_END = "2023-12-31"


# ---------------------------------------------------------------- kos
def tick_size(p):
    if p < 1.00:
        return 0.005
    if p < 10.00:
        return 0.01
    if p < 100.00:
        return 0.02
    return 0.10


def cost_pct(true_price, capital):
    """Kos pusingan penuh. true_price mesti harga TAK diselaraskan."""
    shares = capital / true_price
    brok = BROKERAGE_FLAT * 2
    stamp = capital * STAMP_DUTY_PCT / 100 * 2
    clear = capital * CLEARING_PCT / 100 * 2
    spread = shares * tick_size(true_price)
    return (brok + stamp + clear + spread) / capital * 100


def fetch_unadjusted(tickers):
    """Turun & cache Close mentah. Hanya untuk kiraan kos."""
    os.makedirs(UNADJDIR, exist_ok=True)
    missing = [t for t in tickers
               if not os.path.exists(f"{UNADJDIR}/{t}.csv")]
    if not missing:
        return
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance tiada - kos akan guna harga diselaraskan (melambung)")
        return
    print(f"Menurunkan harga mentah untuk {len(missing)} saham...")
    for t in missing:
        try:
            d = yf.download(t, start="2014-01-01", end="2026-09-04",
                            progress=False, auto_adjust=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.droplevel(1)
            d[["Close"]].to_csv(f"{UNADJDIR}/{t}.csv")
            print(f"  {t} ok")
        except Exception as e:
            print(f"  {t} gagal: {str(e)[:40]}")


def load_unadj(t):
    p = f"{UNADJDIR}/{t}.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, index_col=0, parse_dates=True)
    return d["Close"]


# ---------------------------------------------------------- isyarat
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
    out["C_breakout"] = c >= c.rolling(20).max()
    freq = np.mean([out[k].mean() for k in out])
    out["D_random"] = pd.Series(rng.random(len(c)) < freq, index=c.index)
    return out


def evaluate(df, sig, capital, unadj):
    c = df["Close"]
    entry = c.shift(-1)
    exit_ = c.shift(-1 - HOLD)
    gross = (exit_ / entry - 1) * 100

    trades = []
    for dt in df.index[sig.fillna(False).values]:
        e, g = entry.get(dt), gross.get(dt)
        if pd.isna(e) or pd.isna(g):
            continue
        # harga sebenar pada tarikh masuk untuk kiraan kos
        if unadj is not None:
            nxt = df.index[df.index > dt]
            true_p = float(unadj.get(nxt[0], e)) if len(nxt) else float(e)
        else:
            true_p = float(e)
        cst = cost_pct(true_p, capital)
        trades.append({"gross": g, "cost": cst, "net": g - cst})
    return trades


# --------------------------------------------------------- ringkasan
def summarise(trades, label):
    if len(trades) < 20:
        return None
    t = pd.DataFrame(trades)
    n = len(t)
    dr = (t["gross"] > 0).mean()
    nw = (t["net"] > 0).mean()
    return {
        "strategy": label,
        "n": n,
        "direction_rate": round(dr * 100, 1),
        "dir_ci": f"{(dr-1.96*np.sqrt(dr*(1-dr)/n))*100:.1f}-"
                  f"{(dr+1.96*np.sqrt(dr*(1-dr)/n))*100:.1f}",
        "net_win_rate": round(nw * 100, 1),
        "gross_mean": round(t["gross"].mean(), 3),
        "gross_median": round(t["gross"].median(), 3),
        "cost_mean": round(t["cost"].mean(), 3),
        "net_mean": round(t["net"].mean(), 3),
        "net_median": round(t["net"].median(), 3),
        "_dr": dr, "_n": n,
    }


def two_prop_z(p1, n1, p2, n2):
    """Ujian dua-hala: adakah p1 berbeza dari p2?"""
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=25000)
    ap.add_argument("--split", default="train",
                    choices=["train", "val", "test", "all"])
    a = ap.parse_args()

    tickers = (pd.read_csv(UNIVERSE)["ticker"].tolist()
               if os.path.exists(UNIVERSE) else [])
    if not tickers:
        print("final_universe.csv tiada")
        return

    fetch_unadjusted(tickers)

    print(f"\nUniverse : {len(tickers)} saham")
    print(f"Modal    : RM{a.capital:,.0f}   Split: {a.split}   "
          f"Pegang: {HOLD} hari\n")

    rng = np.random.default_rng(SEED)
    by_strat, bh = {}, []

    for t in tickers:
        p = f"{CLEANDIR}/{t}.csv"
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
        unadj = load_unadj(t)

        if a.split == "train":
            df = df[df.index <= TRAIN_END]
        elif a.split == "val":
            df = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
        elif a.split == "test":
            df = df[df.index > VAL_END]
        if len(df) < 100:
            continue

        for name, s in signals(df, rng).items():
            by_strat.setdefault(name, []).extend(
                evaluate(df, s, a.capital, unadj))

        c = df["Close"]
        bh.append({"ticker": t, "ret": (c.iloc[-1] / c.iloc[0] - 1) * 100})

    rows = [r for r in (summarise(v, k) for k, v in sorted(by_strat.items()))
            if r]
    res = pd.DataFrame(rows)

    rnd = res[res["strategy"] == "D_random"].iloc[0]

    verdicts = []
    for _, r in res.iterrows():
        if r["strategy"] == "D_random":
            verdicts.append("—")
            continue
        z, pv = two_prop_z(r["_dr"], r["_n"], rnd["_dr"], rnd["_n"])
        if pv >= 0.05:
            verdicts.append(f"sama (p={pv:.2f})")
        elif z > 0:
            verdicts.append(f"LEBIH BAIK (p={pv:.3f})")
        else:
            verdicts.append(f"LEBIH TERUK (p={pv:.3f})")
    res["vs_random"] = verdicts

    show = ["strategy", "n", "direction_rate", "dir_ci", "net_win_rate",
            "gross_mean", "gross_median", "cost_mean", "net_mean",
            "net_median", "vs_random"]
    res[show].to_csv(f"data/baseline_v2_{a.split}.csv", index=False)
    print(res[show].to_string(index=False))

    # ---------------------------------------------- survivorship
    print("\n" + "=" * 76)
    print("SEMAKAN SURVIVORSHIP")
    print("=" * 76)
    b = pd.DataFrame(bh)
    print(f"Universe buy-hold  purata : {b['ret'].mean():+.1f}%")
    print(f"Universe buy-hold  median : {b['ret'].median():+.1f}%")
    print(f"Terbaik / terburuk        : {b['ret'].max():+.1f}% / "
          f"{b['ret'].min():+.1f}%")

    ip = f"{CLEANDIR}/IDX_KLSE.csv"
    if os.path.exists(ip):
        idx = pd.read_csv(ip, index_col=0, parse_dates=True).sort_index()
        if a.split == "train":
            idx = idx[idx.index <= TRAIN_END]
        elif a.split == "val":
            idx = idx[(idx.index > TRAIN_END) & (idx.index <= VAL_END)]
        elif a.split == "test":
            idx = idx[idx.index > VAL_END]
        if len(idx) > 10:
            ir = (idx["Close"].iloc[-1] / idx["Close"].iloc[0] - 1) * 100
            print(f"Indeks KLCI               : {ir:+.1f}%")
            print(f"Universe median lebih     : "
                  f"{b['ret'].median() - ir:+.1f} mata peratusan")
            print("\nJurang besar = universe dipilih dari pemenang yang")
            print("sudah diketahui. Saham yang runtuh tiada dalam senarai.")

    print("\nPer saham:")
    print(b.sort_values("ret", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
