"""
Eksperimen perbandingan horizon.

Soalan: horizon mana yang pergerakan median-nya MELEBIHI kos dengan
        margin bermakna? Itu sahaja calon untuk ML.

Untuk setiap horizon (1,3,5,10,20) dan setiap strategi:
    median & purata pulangan kasar
    median & purata pulangan bersih
    ketepatan arah + CI 95%
    ujian dua-hala vs rawak

Ditambah:
    - Baris ALL_DAYS : pergerakan setiap hari, bukan hari isyarat sahaja.
      Ini garis dasar sebenar - berapa banyak pergerakan WUJUD.
    - Baris BUY_HOLD : hanyut sepanjang tempoh.
      Pada horizon panjang hanyut mendominasi; tanpa lajur ni kau
      boleh salah tafsir hanyut sebagai isyarat.

Usage:
    python horizon_experiment.py --capital 25000 --split train
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats

CLEANDIR = "data/clean"
UNADJDIR = "data/unadj"
UNIVERSE = "data/final_universe.csv"

HORIZONS = [1, 3, 5, 10, 20]
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


def cost_pct(true_price, capital):
    shares = capital / true_price
    brok = BROKERAGE_FLAT * 2
    stamp = capital * STAMP_DUTY_PCT / 100 * 2
    clear = capital * CLEARING_PCT / 100 * 2
    spread = shares * tick_size(true_price)
    return (brok + stamp + clear + spread) / capital * 100


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def signals(df, rng):
    c = df["Close"]
    o = {}
    r = rsi(c)
    o["A_rsi"] = (r < 30) & (r.shift(1) >= 30)
    ma = c.rolling(20).mean()
    o["B_ma_cross"] = (c > ma) & (c.shift(1) <= ma.shift(1))
    o["C_breakout"] = c >= c.rolling(20).max()
    o["E_momentum"] = (c / c.shift(20) - 1) > 0.05
    freq = np.mean([o[k].mean() for k in ["A_rsi", "B_ma_cross", "C_breakout"]])
    o["D_random"] = pd.Series(rng.random(len(c)) < freq, index=c.index)
    return o


def load_unadj(t):
    p = f"{UNADJDIR}/{t}.csv"
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, index_col=0, parse_dates=True)["Close"]


def collect(df, unadj, h, capital, mask=None):
    """Pulangkan senarai dict trade untuk horizon h."""
    c = df["Close"]
    entry = c.shift(-1)
    exit_ = c.shift(-1 - h)
    gross = (exit_ / entry - 1) * 100

    idx = df.index if mask is None else df.index[mask.fillna(False).values]
    out = []
    for dt in idx:
        e, g = entry.get(dt), gross.get(dt)
        if pd.isna(e) or pd.isna(g):
            continue
        if unadj is not None:
            nxt = df.index[df.index > dt]
            tp = float(unadj.get(nxt[0], e)) if len(nxt) else float(e)
        else:
            tp = float(e)
        cst = cost_pct(tp, capital)
        out.append({"gross": g, "cost": cst, "net": g - cst})
    return out


def summarise(trades, label, h):
    if len(trades) < 20:
        return None
    t = pd.DataFrame(trades)
    n = len(t)
    dr = (t["gross"] > 0).mean()
    se = np.sqrt(dr * (1 - dr) / n)
    return {
        "horizon": h, "strategy": label, "n": n,
        "dir_acc": round(dr * 100, 1),
        "ci_lo": round((dr - 1.96 * se) * 100, 1),
        "ci_hi": round((dr + 1.96 * se) * 100, 1),
        "gross_med": round(t["gross"].median(), 3),
        "gross_mean": round(t["gross"].mean(), 3),
        "cost": round(t["cost"].mean(), 3),
        "net_med": round(t["net"].median(), 3),
        "net_mean": round(t["net"].mean(), 3),
        "_dr": dr, "_n": n,
    }


def two_prop_z(p1, n1, p2, n2):
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
                    choices=["train", "val", "all"])
    a = ap.parse_args()

    tickers = pd.read_csv(UNIVERSE)["ticker"].tolist()
    rng = np.random.default_rng(SEED)

    frames = {}
    for t in tickers:
        p = f"{CLEANDIR}/{t}.csv"
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
        if a.split == "train":
            d = d[d.index <= TRAIN_END]
        elif a.split == "val":
            d = d[(d.index > TRAIN_END) & (d.index <= VAL_END)]
        if len(d) > 100:
            frames[t] = (d, load_unadj(t))

    print(f"Universe: {len(frames)} saham   Split: {a.split}   "
          f"Modal: RM{a.capital:,.0f}\n")

    rows = []
    for h in HORIZONS:
        pool = {}
        for t, (d, u) in frames.items():
            # garis dasar: semua hari
            pool.setdefault("Z_all_days", []).extend(
                collect(d, u, h, a.capital, None))
            for name, s in signals(d, rng).items():
                pool.setdefault(name, []).extend(
                    collect(d, u, h, a.capital, s))

        summaries = {k: summarise(v, k, h) for k, v in pool.items()}
        rnd = summaries.get("D_random")
        for k in sorted(summaries):
            s = summaries[k]
            if s is None:
                continue
            if rnd and k not in ("D_random", "Z_all_days"):
                z, pv = two_prop_z(s["_dr"], s["_n"], rnd["_dr"], rnd["_n"])
                s["vs_rand"] = ("sama" if pv >= 0.05
                                else ("BAIK" if z > 0 else "TERUK"))
                s["p"] = round(pv, 3)
            else:
                s["vs_rand"], s["p"] = "—", np.nan
            rows.append(s)

    res = pd.DataFrame(rows)
    cols = ["horizon", "strategy", "n", "dir_acc", "ci_lo", "ci_hi",
            "gross_med", "gross_mean", "cost", "net_med", "net_mean",
            "vs_rand", "p"]
    res[cols].to_csv(f"data/horizon_experiment_{a.split}.csv", index=False)

    for h in HORIZONS:
        sub = res[res["horizon"] == h]
        print("=" * 92)
        print(f"HORIZON {h} HARI")
        print("=" * 92)
        print(sub[cols[1:]].to_string(index=False))
        print()

    # ---- jadual keputusan ----
    print("=" * 92)
    print("JADUAL KEPUTUSAN  (garis dasar semua-hari vs kos)")
    print("=" * 92)
    print(f"{'Horizon':>8} {'Gerak median':>14} {'Kos':>8} "
          f"{'Nisbah':>8} {'Ruang?':>10}")
    print("-" * 92)
    for h in HORIZONS:
        b = res[(res["horizon"] == h) & (res["strategy"] == "Z_all_days")]
        if not len(b):
            continue
        b = b.iloc[0]
        mv = abs(b["gross_med"])
        cs = b["cost"]
        ratio = mv / cs if cs else 0
        flag = ("tiada" if ratio < 1 else
                "sempit" if ratio < 2 else "ADA RUANG")
        print(f"{h:>8} {b['gross_med']:>14.3f} {cs:>8.3f} "
              f"{ratio:>8.2f} {flag:>10}")

    print("\nGerak median ialah pergerakan tipikal yang WUJUD pada horizon")
    print("itu, tanpa mengira strategi. Kalau ia bawah kos, tiada model")
    print("boleh menjadikannya menguntungkan pada trade biasa.")
    print("\nNisbah 2+ bermakna ada ruang untuk ML dicuba. Bawah 1")
    print("bermakna horizon itu mati sebelum model masuk.")


if __name__ == "__main__":
    main()
