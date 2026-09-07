"""
Screening liquidity + volatility untuk data Bursa yang dah didownload.

Usage:
    python screen_bursa.py

Input:
    data/raw/*.csv          (dari download_bursa.py)
    data/raw/_manifest.csv

Output:
    data/screen_report.csv  - metrik penuh setiap counter
    printout                - taburan + cadangan keep/drop
"""

import os
import glob
import numpy as np
import pandas as pd

RAWDIR = "data/raw"
LOOKBACK_DAYS = 500      # guna data terkini je untuk nilai liquidity semasa
MIN_ROWS = 1200          # kena ada sejarah cukup panjang (~5 tahun)

# Threshold PERMULAAN. Tengok taburan dulu, pastu adjust.
MIN_TURNOVER_RM = 3_000_000    # median RM bertukar tangan sehari
MIN_RANGE_PCT = 1.2            # median (High-Low)/Close, dalam %
MAX_DEAD_PCT = 10.0            # max % hari tak bergerak / volume sifar


def load(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.dropna(subset=["Close"])
    return df.sort_index()


def metrics(df):
    """Kira metrik screening dari data terkini."""
    recent = df.tail(LOOKBACK_DAYS)
    if len(recent) < 100:
        return None

    close = recent["Close"]
    high = recent["High"]
    low = recent["Low"]
    vol = recent["Volume"]

    turnover = close * vol
    daily_range = (high - low) / close * 100
    ret = close.pct_change()

    dead = ((vol == 0) | (ret.abs() < 1e-9)).sum() / len(recent) * 100

    # anggaran gap overnight - berapa banyak pergerakan berlaku
    # sebelum kau sempat masuk pada Open
    gap = (recent["Open"] / close.shift(1) - 1).abs() * 100

    return {
        "rows_total": len(df),
        "first": str(df.index[0].date()),
        "last": str(df.index[-1].date()),
        "price": round(close.iloc[-1], 3),
        "turnover_median_rm": int(turnover.median()),
        "turnover_p25_rm": int(turnover.quantile(0.25)),
        "range_median_pct": round(daily_range.median(), 2),
        "vol_annual_pct": round(ret.std() * np.sqrt(252) * 100, 1),
        "gap_median_pct": round(gap.median(), 2),
        "dead_days_pct": round(dead, 1),
    }


def verdict(m):
    """Keputusan keep/drop + sebab."""
    reasons = []
    if m["rows_total"] < MIN_ROWS:
        reasons.append("sejarah pendek")
    if m["turnover_median_rm"] < MIN_TURNOVER_RM:
        reasons.append("turnover rendah")
    if m["range_median_pct"] < MIN_RANGE_PCT:
        reasons.append("tak cukup gerak")
    if m["dead_days_pct"] > MAX_DEAD_PCT:
        reasons.append("banyak hari mati")
    return ("DROP", "; ".join(reasons)) if reasons else ("KEEP", "")


def main():
    files = sorted(glob.glob(f"{RAWDIR}/*.csv"))
    files = [f for f in files if not os.path.basename(f).startswith("_")]
    files = [f for f in files if not os.path.basename(f).startswith("IDX_")]

    if not files:
        print(f"Tiada fail dalam {RAWDIR}/. Run download_bursa.py dulu.")
        return

    rows = []
    for f in files:
        ticker = os.path.basename(f).replace(".csv", "")
        try:
            m = metrics(load(f))
            if m is None:
                rows.append({"ticker": ticker, "keep": "DROP",
                             "why": "data tak cukup"})
                continue
            k, why = verdict(m)
            rows.append({"ticker": ticker, "keep": k, "why": why, **m})
        except Exception as e:
            rows.append({"ticker": ticker, "keep": "DROP",
                         "why": f"ralat: {str(e)[:30]}"})

    rep = pd.DataFrame(rows).sort_values(
        "turnover_median_rm", ascending=False, na_position="last")
    rep.to_csv("data/screen_report.csv", index=False)

    ok = rep[rep["keep"] == "KEEP"]

    # ---- taburan: tengok ni sebelum percaya threshold ----
    print("TABURAN (semua counter)")
    print("-" * 72)
    for col, label in [("turnover_median_rm", "Turnover harian (RM)"),
                       ("range_median_pct", "Range harian (%)"),
                       ("gap_median_pct", "Gap overnight (%)")]:
        s = rep[col].dropna()
        if len(s):
            print(f"{label:<24} min={s.min():>12,.1f}  "
                  f"p25={s.quantile(.25):>12,.1f}  "
                  f"median={s.median():>12,.1f}  "
                  f"max={s.max():>12,.1f}")

    print("\n" + "=" * 72)
    print(f"KEEP: {len(ok)}    DROP: {len(rep) - len(ok)}")
    print("=" * 72)

    cols = ["ticker", "price", "turnover_median_rm", "range_median_pct",
            "gap_median_pct", "vol_annual_pct", "rows_total"]
    if len(ok):
        print("\nLULUS:")
        print(ok[cols].to_string(index=False))

    bad = rep[rep["keep"] == "DROP"]
    if len(bad):
        print("\nGUGUR:")
        print(bad[["ticker", "why"]].to_string(index=False))

    # ---- perkara penting: gap vs range ----
    if len(ok):
        ratio = (ok["gap_median_pct"] / ok["range_median_pct"]).median()
        print(f"\nGap / range = {ratio:.2f}")
        print("Makna: lebih kurang bahagian ni daripada pergerakan harian")
        print("berlaku semasa market tutup - sebelum kau sempat masuk.")

    print(f"\nLaporan penuh: data/screen_report.csv")


if __name__ == "__main__":
    main()
