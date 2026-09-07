"""
Pembersihan data Bursa - buang baris yang diisi (bukan didagangkan).

Latar belakang:
    Feed Yahoo untuk Bursa mengisi data yang hilang dengan nilai terakhir
    diketahui. Dua tanda:
      A) Open == Close semalam  (Open sintetik - biasa, ~30-60%)
      B) Open==High==Low==Close (hari hilang sepenuhnya)

    Jenis A: kita abaikan Open terus, guna Close-ke-Close. Baris masih guna.
    Jenis B: baris tu tiada maklumat langsung. Kena buang.

Yang script ni buat:
    1. Kira baris jenis B per counter
    2. Gugurkan counter yang terlalu tercemar
    3. Buang baris jenis B dari counter yang tinggal
    4. Buang lajur Open (tak boleh dipercayai)
    5. Simpan ke data/clean/

Usage:
    python clean_bursa.py
"""

import os
import glob
import numpy as np
import pandas as pd

RAWDIR = "data/raw"
CLEANDIR = "data/clean"

MAX_FROZEN_PCT = 3.0      # counter dengan lebih ni digugurkan
MAX_GAP_DAYS = 10         # jurang kalendar lebih ni = masalah serius
MIN_ROWS_AFTER = 1200


def detect(df):
    """Tandakan baris beku - keempat-empat harga sama."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    frozen = (
        np.isclose(o, c, rtol=1e-9, atol=0)
        & np.isclose(h, l, rtol=1e-9, atol=0)
        & np.isclose(o, h, rtol=1e-9, atol=0)
    )
    # High==Low sahaja pun mencurigakan walaupun Open berbeza
    flat_range = np.isclose(h, l, rtol=1e-9, atol=0)
    return frozen, flat_range


def clean_one(path):
    ticker = os.path.basename(path).replace(".csv", "")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.dropna(subset=["Close"]).sort_index()
    n0 = len(df)
    if n0 < 200:
        return ticker, None, {"status": "DROP", "why": "terlalu sedikit baris"}

    frozen, flat_range = detect(df)

    stat = {
        "rows_raw": n0,
        "frozen_pct": round(frozen.mean() * 100, 2),
        "flat_range_pct": round(flat_range.mean() * 100, 2),
        "zero_vol_pct": round((df["Volume"] == 0).mean() * 100, 2),
    }

    if stat["frozen_pct"] > MAX_FROZEN_PCT:
        stat.update(status="DROP", why=f"beku {stat['frozen_pct']}%")
        return ticker, None, stat

    # buang baris beku dan baris range-rata
    keep = ~(frozen | flat_range)
    out = df.loc[keep].copy()

    # Open tak boleh dipercayai - buang terus supaya tak tersilap guna
    out = out.drop(columns=[c for c in ["Open", "Adj Close"] if c in out.columns])

    # semak jurang kalendar selepas pembuangan
    gaps = out.index.to_series().diff().dt.days
    stat["max_gap_days"] = int(gaps.max()) if len(gaps.dropna()) else 0
    stat["rows_clean"] = len(out)
    stat["removed"] = n0 - len(out)

    if len(out) < MIN_ROWS_AFTER:
        stat.update(status="DROP", why="tak cukup baris selepas bersih")
        return ticker, None, stat
    if stat["max_gap_days"] > MAX_GAP_DAYS:
        stat.update(status="FLAG", why=f"jurang {stat['max_gap_days']} hari")
        return ticker, out, stat

    stat.update(status="KEEP", why="")
    return ticker, out, stat


def main():
    os.makedirs(CLEANDIR, exist_ok=True)

    files = sorted(glob.glob(f"{RAWDIR}/*.csv"))
    files = [f for f in files if not os.path.basename(f).startswith("_")]
    if not files:
        print(f"Tiada fail dalam {RAWDIR}/")
        return

    rows = []
    for f in files:
        ticker, out, stat = clean_one(f)
        stat["ticker"] = ticker
        rows.append(stat)
        if out is not None:
            out.to_csv(f"{CLEANDIR}/{ticker}.csv")

    rep = pd.DataFrame(rows)
    cols = ["ticker", "status", "why", "rows_raw", "rows_clean", "removed",
            "frozen_pct", "flat_range_pct", "zero_vol_pct", "max_gap_days"]
    rep = rep.reindex(columns=cols).sort_values("frozen_pct", ascending=False)
    rep.to_csv(f"{CLEANDIR}/_clean_report.csv", index=False)

    print(rep.to_string(index=False))
    print("\n" + "=" * 60)
    print(rep["status"].value_counts().to_string())

    # ---- semakan penting: adakah baris beku jatuh pada tarikh SAMA? ----
    print("\n" + "=" * 60)
    print("SEMAKAN: adakah baris beku berkongsi tarikh yang sama?")
    print("(kalau ya, ia hari feed rosak, bukan sifat counter)")
    print("=" * 60)

    date_hits = {}
    for f in files[:20]:
        try:
            d = pd.read_csv(f, index_col=0, parse_dates=True).dropna(subset=["Close"])
            fr, _ = detect(d)
            for dt in d.index[fr]:
                date_hits[dt] = date_hits.get(dt, 0) + 1
        except Exception:
            pass

    if date_hits:
        top = sorted(date_hits.items(), key=lambda x: -x[1])[:15]
        print(f"{'Tarikh':<14} {'Bilangan counter beku'}")
        for dt, n in top:
            bar = "#" * min(n, 20)
            print(f"{str(dt.date()):<14} {n:>3}  {bar}")
        print("\nKalau satu tarikh kena banyak counter serentak,")
        print("itu hari feed rosak. Buang tarikh tu untuk SEMUA counter.")
    else:
        print("Tiada baris beku dijumpai.")

    print(f"\nFail bersih: {CLEANDIR}/")


if __name__ == "__main__":
    main()
