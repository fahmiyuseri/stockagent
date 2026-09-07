"""
Sapuan tempoh pegangan.

Soalan: tempoh pegangan mana yang memberi ruang cukup antara
        peluang dan kos?

Aritmetiknya:
    Kos C  = tetap setiap trade (tak kira pegang berapa lama)
    Gerak M ~ berkembang mengikut punca kuasa dua masa
    Kadar menang perlu = (L + C) / (W + L)

    Pegang lebih lama -> M naik, C tetap -> kadar menang perlu turun.
    Harganya: kurang trade setahun -> anggaran lebih bising.

Script ni juga kira KUASA STATISTIK - berapa tepat kau boleh
mengukur kadar menang dalam tetingkap test kau.

Usage:
    python sweep_horizon.py --brokerage-pct 0.10 --brokerage-min 8
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

CLEANDIR = "data/clean"
HORIZONS = [1, 2, 3, 5, 10, 20, 40]

# tetingkap test untuk kiraan kuasa statistik
TEST_YEARS = 2.7
TRADING_DAYS_YEAR = 250
CONCURRENT_POSITIONS = 3

STAMP_DUTY_PCT = 0.10      # SAHKAN
CLEARING_PCT = 0.03        # SAHKAN


def tick_size(p):
    if p < 1.00:
        return 0.005
    if p < 10.00:
        return 0.01
    if p < 100.00:
        return 0.02
    return 0.10


def cost_pct(price, capital, brok_pct, brok_min):
    shares = capital / price
    brok = max(capital * brok_pct / 100, brok_min) * 2
    stamp = capital * STAMP_DUTY_PCT / 100 * 2
    clear = capital * CLEARING_PCT / 100 * 2
    spread = shares * tick_size(price)
    return (brok + stamp + clear + spread) / capital * 100


def analyse(df, h, capital, brok_pct, brok_min):
    c = df["Close"]
    price = float(c.iloc[-1])

    ret = (c.shift(-h) / c - 1).dropna().tail(750) * 100
    pos, neg = ret[ret > 0], ret[ret < 0]
    if len(pos) < 30 or len(neg) < 30:
        return None

    W, L = pos.mean(), abs(neg.mean())
    C = cost_pct(price, capital, brok_pct, brok_min)
    p_be = (L + C) / (W + L)

    return {"price": price, "cost": C, "W": W, "L": L, "req": p_be}


def stat_power(h):
    """Ralat piawai anggaran kadar menang dalam tetingkap test."""
    n = TEST_YEARS * TRADING_DAYS_YEAR / h * CONCURRENT_POSITIONS
    se = np.sqrt(0.55 * 0.45 / n) if n > 0 else np.nan
    return int(n), se * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=10000)
    ap.add_argument("--brokerage-pct", type=float, default=0.10)
    ap.add_argument("--brokerage-min", type=float, default=8.0)
    ap.add_argument("--threshold", type=float, default=0.58,
                    help="kadar menang maksimum yang dianggap boleh dicapai")
    a = ap.parse_args()

    files = sorted(glob.glob(f"{CLEANDIR}/*.csv"))
    files = [f for f in files
             if not os.path.basename(f).startswith(("_", "IDX_"))]
    if not files:
        print(f"Tiada fail dalam {CLEANDIR}/")
        return

    data = {}
    for f in files:
        t = os.path.basename(f).replace(".csv", "")
        try:
            data[t] = pd.read_csv(f, index_col=0, parse_dates=True).sort_index()
        except Exception:
            pass

    print(f"Universe : {len(data)} saham")
    print(f"Modal    : RM{a.capital:,.0f}")
    print(f"Ambang   : perlu <= {a.threshold:.0%}\n")

    # ---- jadual ringkasan setiap horizon ----
    print("RINGKASAN MENGIKUT TEMPOH PEGANGAN")
    print("=" * 78)
    print(f"{'Hari':>5} {'Lulus':>6} {'Req terbaik':>12} {'Req median':>11} "
          f"{'Trade/test':>11} {'Ralat piawai':>13}")
    print("-" * 78)

    all_rows = []
    for h in HORIZONS:
        reqs = []
        for t, d in data.items():
            r = analyse(d, h, a.capital, a.brokerage_pct, a.brokerage_min)
            if r:
                reqs.append({"ticker": t, "horizon": h, **r})
        if not reqs:
            continue
        all_rows.extend(reqs)

        rr = pd.DataFrame(reqs)
        passed = rr[rr["req"] <= a.threshold]
        n, se = stat_power(h)

        print(f"{h:>5} {len(passed):>6} {rr['req'].min():>12.1%} "
              f"{rr['req'].median():>11.1%} {n:>11} {se:>12.2f}pp")

    full = pd.DataFrame(all_rows)
    full.to_csv("data/horizon_sweep.csv", index=False)

    # ---- pertimbangan utama ----
    print("\n" + "=" * 78)
    print("PERTIMBANGAN")
    print("=" * 78)
    print("Pegang lebih lama  -> kadar menang perlu TURUN (bagus)")
    print("Pegang lebih lama  -> kurang trade, ralat piawai NAIK (buruk)")
    print()
    print("Ralat piawai ialah ketidakpastian anggaran kadar menang kau.")
    print("Kalau backtest bagi 56% dengan ralat piawai 2pp, kadar sebenar")
    print("kau berada di antara 52% dan 60% - dari rugi ke untung.")
    print("Kau perlukan jurang antara kadar menang perlu dan kadar")
    print("dicapai yang LEBIH BESAR daripada ralat piawai untuk membuat")
    print("keputusan yang bermakna.")

    # ---- 10 terbaik setiap horizon utama ----
    for h in [3, 10, 20]:
        sub = full[full["horizon"] == h].nsmallest(10, "req")
        if len(sub) == 0:
            continue
        print("\n" + "-" * 78)
        print(f"10 TERBAIK - pegang {h} hari")
        print("-" * 78)
        s = sub.copy()
        s["req"] = (s["req"] * 100).round(1)
        s["cost"] = s["cost"].round(3)
        s["W"] = s["W"].round(2)
        s["L"] = s["L"].round(2)
        print(s[["ticker", "price", "cost", "W", "L", "req"]]
              .to_string(index=False))

    print("\nData penuh: data/horizon_sweep.csv")


if __name__ == "__main__":
    main()
