"""
Screening berasaskan KADAR MENANG DIPERLUKAN, bukan saiz pergerakan.

Kenapa tulis semula (lagi):
    edge_days_pct meluluskan 42/44. Ia cuma sahkan harga bergerak.
    Soalannya bukan "adakah saham bergerak cukup" - tapi "berapa
    tepat model kena jadi supaya untung selepas kos".

Formula:
    E[trade] = p*W - (1-p)*L - C = 0
    p_breakeven = (L + C) / (W + L)

    W = purata pergerakan 3-hari positif
    L = purata magnitud pergerakan 3-hari negatif
    C = kos pusingan penuh (bergantung harga - lihat di bawah)

Model kos:
    Kos BUKAN peratusan rata. Ia berkadar songsang dengan harga,
    sebab tick Bursa bersaiz tetap dalam sen.
    Saham RM0.24: satu tick = 2.08% daripada harga.
    Saham RM94:   satu tick = 0.02% daripada harga.

Usage:
    python screen_winrate.py --brokerage-pct 0.10 --brokerage-min 8
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

CLEANDIR = "data/clean"
HOLD_DAYS = 3

# ---------------------------------------------------------------------
# SAHKAN NOMBOR NI DENGAN BROKER KAU. Ini andaian, bukan fakta.
# ---------------------------------------------------------------------
def tick_size(price):
    """Saiz tick Bursa (RM). SAHKAN dengan Bursa/broker."""
    if price < 1.00:
        return 0.005
    if price < 10.00:
        return 0.01
    if price < 100.00:
        return 0.02
    return 0.10


STAMP_DUTY_PCT = 0.10      # % - SAHKAN kadar & had semasa
CLEARING_PCT = 0.03        # % - SAHKAN kadar & had semasa

MAX_REQUIRED_WINRATE = 0.60   # atas ni = tak realistik


def round_trip_cost_pct(price, capital, brok_pct, brok_min):
    """Kos pusingan penuh sebagai % nilai posisi."""
    shares = capital / price
    value = capital

    # brokerage: kadar % atau minimum, mana lebih tinggi - dua hala
    brok_one = max(value * brok_pct / 100, brok_min)
    brok = brok_one * 2

    # duti setem + clearing - dua hala
    stamp = value * STAMP_DUTY_PCT / 100 * 2
    clear = value * CLEARING_PCT / 100 * 2

    # melintas spread: anggap spread = 1 tick, kau bayar sekali pusingan
    spread = shares * tick_size(price)

    return (brok + stamp + clear + spread) / value * 100


def analyse(df, capital, brok_pct, brok_min):
    c = df["Close"]
    v = df["Volume"]
    price = float(c.iloc[-1])

    ret3 = (c.shift(-HOLD_DAYS) / c - 1).dropna().tail(500) * 100
    pos = ret3[ret3 > 0]
    neg = ret3[ret3 < 0]
    if len(pos) < 30 or len(neg) < 30:
        return None

    W = pos.mean()
    L = abs(neg.mean())
    C = round_trip_cost_pct(price, capital, brok_pct, brok_min)

    p_be = (L + C) / (W + L)

    turnover = (c * v).tail(500)

    return {
        "price": round(price, 3),
        "tick_pct": round(tick_size(price) / price * 100, 3),
        "cost_pct": round(C, 3),
        "avg_win_pct": round(W, 2),
        "avg_loss_pct": round(L, 2),
        "req_winrate": round(p_be, 3),
        "turnover_med_rm": int(turnover.median()),
        "rows": len(df),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=10000)
    ap.add_argument("--brokerage-pct", type=float, default=0.10,
                    help="kadar brokerage %% sehala - SAHKAN dengan broker")
    ap.add_argument("--brokerage-min", type=float, default=8.0,
                    help="brokerage minimum RM setiap trade - SAHKAN")
    a = ap.parse_args()

    # amaran fail basi
    rpt = f"{CLEANDIR}/_clean_report2.csv"
    expected = None
    if os.path.exists(rpt):
        r = pd.read_csv(rpt)
        expected = set(r[r["status"].isin(["KEEP", "FLAG"])]["ticker"])

    files = sorted(glob.glob(f"{CLEANDIR}/*.csv"))
    files = [f for f in files
             if not os.path.basename(f).startswith(("_", "IDX_"))]

    if expected is not None:
        found = {os.path.basename(f).replace(".csv", "") for f in files}
        stale = found - expected
        if stale:
            print("!" * 66)
            print(f"FAIL BASI dikesan: {sorted(stale)}")
            print("Ini sisa run lama. Buat: rm -rf data/clean")
            print("kemudian run clean_bursa2.py semula.")
            print("!" * 66 + "\n")

    print(f"Modal per trade   : RM{a.capital:,.0f}")
    print(f"Brokerage         : {a.brokerage_pct}% / min RM{a.brokerage_min}")
    print(f"Tempoh pegang     : {HOLD_DAYS} hari\n")

    rows = []
    for f in files:
        t = os.path.basename(f).replace(".csv", "")
        try:
            d = pd.read_csv(f, index_col=0, parse_dates=True).sort_index()
            m = analyse(d, a.capital, a.brokerage_pct, a.brokerage_min)
            if m:
                rows.append({"ticker": t, **m})
        except Exception as e:
            print(f"  ralat {t}: {str(e)[:40]}")

    rep = pd.DataFrame(rows).sort_values("req_winrate")
    rep["verdict"] = np.where(rep["req_winrate"] <= MAX_REQUIRED_WINRATE,
                              "KEEP", "DROP")
    rep.to_csv("data/screen_winrate.csv", index=False)

    print("Disusun ikut kadar menang diperlukan (rendah = lebih mudah)\n")
    print(rep[["ticker", "price", "tick_pct", "cost_pct", "avg_win_pct",
               "avg_loss_pct", "req_winrate", "verdict"]].to_string(index=False))

    ok = rep[rep["verdict"] == "KEEP"]
    print("\n" + "=" * 70)
    print(f"KEEP: {len(ok)}   DROP: {len(rep)-len(ok)}")
    print(f"Ambang: perlu <= {MAX_REQUIRED_WINRATE:.0%} ketepatan")
    print("=" * 70)

    if len(ok):
        ok[["ticker", "price", "cost_pct", "req_winrate",
            "turnover_med_rm", "rows"]].to_csv(
            "data/final_universe.csv", index=False)
        print(f"\nDikunci: data/final_universe.csv  ({len(ok)} saham)")
        print(f"Kadar menang diperlukan terbaik : {ok['req_winrate'].min():.1%}")
        print(f"Median universe                 : {ok['req_winrate'].median():.1%}")

    print("\nRujukan: model ramalan arah harian yang baik dapat 53-56%.")
    print("Mana-mana saham yang perlu lebih 60% tidak boleh didagangkan")
    print("olehmu, tak kira seni bina model.")


if __name__ == "__main__":
    main()
