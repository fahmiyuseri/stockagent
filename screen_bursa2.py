"""
Screening v2 - betulkan metrik hari mati, tambah semakan integriti Open.

Perubahan dari v1:
  - "hari mati" = High==Low ATAU Volume==0  (bukan close tak berubah)
  - "flat close" jadi maklumat sahaja, bukan sebab gugur
  - TAMBAH: open_eq_prevclose_pct - kesan data Open sintetik

Usage:
    python screen_bursa2.py
"""

import os
import glob
import numpy as np
import pandas as pd

RAWDIR = "data/raw"
LOOKBACK_DAYS = 500
MIN_ROWS = 1200

MIN_TURNOVER_RM = 3_000_000
MIN_RANGE_PCT = 1.5        # dinaikkan - 1.2% terlalu longgar selepas kos
MAX_DEAD_PCT = 5.0         # definisi baru jauh lebih ketat, jadi threshold turun
MAX_SYNTH_OPEN_PCT = 30.0  # atas ni, Open tak boleh dipercayai


def load(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.dropna(subset=["Close"]).sort_index()


def metrics(df):
    recent = df.tail(LOOKBACK_DAYS)
    if len(recent) < 100:
        return None

    o, h, l, c = recent["Open"], recent["High"], recent["Low"], recent["Close"]
    v = recent["Volume"]

    prev_c = c.shift(1)
    ret = c.pct_change()

    # HARI MATI: betul-betul tiada dagangan berlaku
    dead = ((v == 0) | (h == l)).sum() / len(recent) * 100

    # FLAT CLOSE: maklumat sahaja - kesan tick size, bukan kecairan
    flat = (ret.abs() < 1e-9).sum() / len(recent) * 100

    # OPEN SINTETIK: Open sama TEPAT dengan Close semalam
    synth = (np.isclose(o, prev_c, rtol=0, atol=1e-9)).sum() / len(recent) * 100

    gap = (o / prev_c - 1).abs() * 100
    turnover = c * v
    rng = (h - l) / c * 100

    return {
        "rows_total": len(df),
        "last": str(df.index[-1].date()),
        "price": round(c.iloc[-1], 3),
        "turnover_median_rm": int(turnover.median()),
        "range_median_pct": round(rng.median(), 2),
        "gap_median_pct": round(gap.median(), 3),
        "vol_annual_pct": round(ret.std() * np.sqrt(252) * 100, 1),
        "dead_days_pct": round(dead, 1),
        "flat_close_pct": round(flat, 1),
        "synth_open_pct": round(synth, 1),
    }


def verdict(m):
    r = []
    if m["rows_total"] < MIN_ROWS:
        r.append("sejarah pendek")
    if m["turnover_median_rm"] < MIN_TURNOVER_RM:
        r.append("turnover rendah")
    if m["range_median_pct"] < MIN_RANGE_PCT:
        r.append("tak cukup gerak")
    if m["dead_days_pct"] > MAX_DEAD_PCT:
        r.append("hari tanpa dagangan")
    if m["synth_open_pct"] > MAX_SYNTH_OPEN_PCT:
        r.append("OPEN SINTETIK")
    return ("DROP", "; ".join(r)) if r else ("KEEP", "")


def main():
    files = sorted(glob.glob(f"{RAWDIR}/*.csv"))
    files = [f for f in files
             if not os.path.basename(f).startswith(("_", "IDX_"))]
    if not files:
        print(f"Tiada fail dalam {RAWDIR}/")
        return

    rows = []
    for f in files:
        t = os.path.basename(f).replace(".csv", "")
        try:
            m = metrics(load(f))
            if m is None:
                rows.append({"ticker": t, "keep": "DROP", "why": "data kurang"})
                continue
            k, why = verdict(m)
            rows.append({"ticker": t, "keep": k, "why": why, **m})
        except Exception as e:
            rows.append({"ticker": t, "keep": "DROP",
                         "why": f"ralat: {str(e)[:30]}"})

    rep = pd.DataFrame(rows).sort_values(
        "turnover_median_rm", ascending=False, na_position="last")
    rep.to_csv("data/screen_report_v2.csv", index=False)

    ok = rep[rep["keep"] == "KEEP"]
    print(f"KEEP: {len(ok)}   DROP: {len(rep)-len(ok)}\n")

    cols = ["ticker", "price", "turnover_median_rm", "range_median_pct",
            "gap_median_pct", "dead_days_pct", "flat_close_pct",
            "synth_open_pct"]
    if len(ok):
        print("LULUS:")
        print(ok[cols].to_string(index=False))

    # ---- amaran integriti Open, berasingan dari keputusan keep/drop ----
    susp = rep[rep["synth_open_pct"] > 20].dropna(subset=["synth_open_pct"])
    if len(susp):
        print("\n" + "!" * 66)
        print("DATA OPEN MENCURIGAKAN - Open == Close semalam terlalu kerap")
        print("!" * 66)
        print(susp[["ticker", "synth_open_pct", "turnover_median_rm"]]
              .to_string(index=False))
        print("\nKau masuk pada Open[t+1]. Kalau Open sintetik, backtest")
        print("counter ni tak mengukur apa-apa. Sahkan dengan sumber lain")
        print("sebelum guna.")

    bad = rep[rep["keep"] == "DROP"]
    if len(bad):
        print("\nGUGUR:")
        print(bad[["ticker", "why"]].to_string(index=False))

    print("\nLaporan: data/screen_report_v2.csv")


if __name__ == "__main__":
    main()
