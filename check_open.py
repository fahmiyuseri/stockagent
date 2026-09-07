"""
Diagnosis integriti data Open.

Kenapa perlu: kau masuk pada Open[t+1]. Kalau Open cuma salinan
Close semalam, backtest kau tak mengukur apa-apa yang wujud.

v2 guna atol=1e-9 yang terlalu ketat selepas auto_adjust.
Script ni guna toleransi relatif + banding dengan saiz tick.

Usage:
    python check_open.py
"""

import os
import glob
import numpy as np
import pandas as pd

RAWDIR = "data/raw"
LOOKBACK = 500


def tick_size(price):
    """Saiz tick Bursa mengikut julat harga (sen)."""
    if price < 1.00:
        return 0.005
    if price < 10.00:
        return 0.01
    return 0.02


def analyse(path):
    t = os.path.basename(path).replace(".csv", "")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.dropna(subset=["Close"]).sort_index().tail(LOOKBACK)
    if len(df) < 100:
        return None

    o = df["Open"].values
    pc = df["Close"].shift(1).values
    valid = ~np.isnan(pc)
    o, pc = o[valid], pc[valid]

    rel = np.abs(o / pc - 1)

    px = df["Close"].iloc[-1]
    tick_rel = tick_size(px) / px          # satu tick sebagai pecahan harga

    return {
        "ticker": t,
        "price": round(px, 3),
        "tick_pct": round(tick_rel * 100, 3),
        # sifar tepat mengikut toleransi longgar
        "open_eq_prev_pct": round((rel < 1e-6).mean() * 100, 1),
        # gap lebih kecil dari satu tick = mustahil secara fizikal
        "sub_tick_pct": round((rel < tick_rel * 0.5).mean() * 100, 1),
        "gap_median_pct": round(np.median(rel) * 100, 4),
        "gap_p75_pct": round(np.percentile(rel, 75) * 100, 3),
    }


def main():
    files = sorted(glob.glob(f"{RAWDIR}/*.csv"))
    files = [f for f in files
             if not os.path.basename(f).startswith(("_", "IDX_"))]

    rows = [r for r in (analyse(f) for f in files) if r]
    rep = pd.DataFrame(rows).sort_values("sub_tick_pct", ascending=False)
    rep.to_csv("data/open_integrity.csv", index=False)

    print("Kolum penting: sub_tick_pct")
    print("  = % hari gap lebih kecil dari separuh tick.")
    print("  Secara fizikal mustahil. Tinggi = Open sintetik.\n")
    print(rep.to_string(index=False))

    print("\n" + "=" * 60)
    for lab, lo, hi in [("TERUK  (>40%)", 40, 101),
                        ("RAGU   (15-40%)", 15, 40),
                        ("OK     (<15%)", -1, 15)]:
        sub = rep[(rep["sub_tick_pct"] > lo) & (rep["sub_tick_pct"] <= hi)]
        print(f"{lab}: {len(sub)}  {list(sub['ticker'])}")

    print("\nLaporan: data/open_integrity.csv")


if __name__ == "__main__":
    main()
