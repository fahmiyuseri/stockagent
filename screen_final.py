"""
Screening tradability MUKTAMAD - dijalankan atas data BERSIH.

Kenapa tulis semula:
    Screening lama guna data/raw/ yang ada baris cuti dan Open sintetik.
    gap_median_pct dikira dari Open yang dikarang. Hasil tu tak sah.

Perubahan:
    - Input: data/clean/  (tiada Open, tiada baris cuti)
    - dead_days DIBUANG sebagai peraturan gugur (ia cuti umum, bukan sifat saham)
    - Metrik utama tukar ke pergerakan 3-HARI, padan dengan tempoh pegangan
    - Tambah ujian pulangan-bersih-selepas-kos
    - Output: data/final_universe.csv  (dikunci, jangan ubah lepas ni)

Usage:
    python screen_final.py --cost 0.5 --capital 10000
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

CLEANDIR = "data/clean"
HOLD_DAYS = 3

# ---- ambang: laraskan selepas tengok taburan ----
MIN_TURNOVER_RM = 3_000_000
MIN_ROWS = 1200
MAX_PARTICIPATION_PCT = 1.0    # saiz posisi vs turnover harian
MIN_EDGE_DAYS_PCT = 40.0       # % hari pergerakan 3h melebihi kos


def metrics(df, cost_pct, capital):
    c = df["Close"]
    v = df["Volume"]
    h, l = df["High"], df["Low"]

    recent = df.tail(500)
    rc, rv = recent["Close"], recent["Volume"]

    turnover = rc * rv
    turn_med = turnover.median()

    # pergerakan dalam tetingkap pegangan sebenar
    ret3 = (c.shift(-HOLD_DAYS) / c - 1).dropna()
    ret3_recent = ret3.tail(500)
    move3 = ret3_recent.abs() * 100

    # berapa kerap pergerakan cukup besar untuk tutup kos
    edge_days = (move3 > cost_pct).mean() * 100

    # jejak pasaran: posisi kau berbanding turnover harian
    participation = capital / turn_med * 100 if turn_med > 0 else 999

    return {
        "rows": len(df),
        "first": str(df.index[0].date()),
        "last": str(df.index[-1].date()),
        "price": round(rc.iloc[-1], 3),
        "turnover_med_rm": int(turn_med),
        "turnover_p10_rm": int(turnover.quantile(0.10)),
        "participation_pct": round(participation, 3),
        "range_1d_pct": round(((h - l) / c).tail(500).median() * 100, 2),
        "move_3d_med_pct": round(move3.median(), 2),
        "move_3d_p75_pct": round(move3.quantile(0.75), 2),
        "edge_days_pct": round(edge_days, 1),
        "vol_annual_pct": round(rc.pct_change().std() * np.sqrt(252) * 100, 1),
    }


def verdict(m):
    r = []
    if m["rows"] < MIN_ROWS:
        r.append("sejarah pendek")
    if m["turnover_med_rm"] < MIN_TURNOVER_RM:
        r.append("turnover rendah")
    if m["participation_pct"] > MAX_PARTICIPATION_PCT:
        r.append("posisi terlalu besar vs turnover")
    if m["edge_days_pct"] < MIN_EDGE_DAYS_PCT:
        r.append(f"hanya {m['edge_days_pct']}% hari lepas kos")
    return ("DROP", "; ".join(r)) if r else ("KEEP", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=0.5,
                    help="kos pusingan penuh dalam %% (brokerage+duti+clearing+slippage)")
    ap.add_argument("--capital", type=float, default=10000,
                    help="saiz posisi RM setiap trade")
    a = ap.parse_args()

    files = sorted(glob.glob(f"{CLEANDIR}/*.csv"))
    files = [f for f in files
             if not os.path.basename(f).startswith(("_", "IDX_"))]
    if not files:
        print(f"Tiada fail dalam {CLEANDIR}/. Run clean_bursa2.py dulu.")
        return

    print(f"Kos pusingan : {a.cost}%")
    print(f"Saiz posisi  : RM{a.capital:,.0f}")
    print(f"Tempoh pegang: {HOLD_DAYS} hari\n")

    rows = []
    for f in files:
        t = os.path.basename(f).replace(".csv", "")
        try:
            d = pd.read_csv(f, index_col=0, parse_dates=True).sort_index()
            if "Open" in d.columns:
                print(f"  AMARAN: {t} masih ada lajur Open - data belum bersih?")
            m = metrics(d, a.cost, a.capital)
            k, why = verdict(m)
            rows.append({"ticker": t, "keep": k, "why": why, **m})
        except Exception as e:
            rows.append({"ticker": t, "keep": "DROP",
                         "why": f"ralat: {str(e)[:30]}"})

    rep = pd.DataFrame(rows).sort_values("turnover_med_rm", ascending=False)
    rep.to_csv("data/screen_final.csv", index=False)

    # ---- taburan dahulu, keputusan kemudian ----
    print("TABURAN")
    print("-" * 70)
    for col, lab in [("turnover_med_rm", "Turnover harian RM"),
                     ("move_3d_med_pct", "Pergerakan 3-hari %"),
                     ("edge_days_pct", "Hari lepas kos %"),
                     ("participation_pct", "Penyertaan %")]:
        s = rep[col].dropna()
        if len(s):
            print(f"{lab:<22} p10={s.quantile(.10):>11,.2f}  "
                  f"median={s.median():>11,.2f}  p90={s.quantile(.90):>11,.2f}")

    ok = rep[rep["keep"] == "KEEP"]
    print("\n" + "=" * 70)
    print(f"KEEP: {len(ok)}   DROP: {len(rep)-len(ok)}")
    print("=" * 70)

    cols = ["ticker", "price", "turnover_med_rm", "range_1d_pct",
            "move_3d_med_pct", "edge_days_pct", "participation_pct", "rows"]
    if len(ok):
        print("\nUNIVERSE MUKTAMAD:")
        print(ok[cols].to_string(index=False))
        ok[["ticker", "price", "turnover_med_rm", "move_3d_med_pct",
            "edge_days_pct", "rows", "first", "last"]].to_csv(
            "data/final_universe.csv", index=False)
        print(f"\nDikunci ke: data/final_universe.csv  ({len(ok)} saham)")

    bad = rep[rep["keep"] == "DROP"]
    if len(bad):
        print("\nGUGUR:")
        print(bad[["ticker", "why"]].to_string(index=False))

    # ---- konteks: apa maksud nombor kos tu ----
    if len(ok):
        med_move = ok["move_3d_med_pct"].median()
        print("\n" + "=" * 70)
        print(f"Pergerakan 3-hari median universe : {med_move:.2f}%")
        print(f"Kos pusingan penuh                : {a.cost:.2f}%")
        print(f"Baki sebelum sebarang ketepatan   : {med_move - a.cost:.2f}%")
        print("\nItu jumlah kasar yang ada untuk direbut. Model kau kena")
        print("betul dengan margin yang cukup untuk merampas sebahagian")
        print("daripadanya secara konsisten - dan separuh trade akan salah arah.")


if __name__ == "__main__":
    main()
