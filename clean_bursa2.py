"""
Pembersihan v2 - buang tarikh cuti pasaran secara GLOBAL dahulu.

Penemuan dari v1:
    Baris beku (Open==High==Low==Close, volume 0) jatuh pada tarikh yang
    SAMA merentas semua counter. Ia hari cuti umum di mana Bursa tutup.
    Yahoo masukkan baris pengisi.

    Jadi ia bukan sifat per-saham. Kena buang secara global.

Urutan:
    1. Kesan tarikh di mana majoriti counter beku -> senarai tarikh buruk
    2. Buang tarikh tu dari SEMUA counter
    3. Baru ukur baki baris beku per-counter
    4. Buang lajur Open (tak boleh dipercayai - isu berasingan)
    5. Lapor jurang kalendar yang tinggal

Usage:
    python clean_bursa2.py
"""

import os
import glob
import numpy as np
import pandas as pd

RAWDIR = "data/raw"
CLEANDIR = "data/clean"

BAD_DATE_FRAC = 0.5        # >50% counter beku pada tarikh tu = tarikh buruk
MAX_RESIDUAL_PCT = 2.0     # baki beku selepas buang tarikh global
MIN_ROWS_AFTER = 1200


def frozen_mask(df):
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    return (np.isclose(o, c) & np.isclose(h, l) & np.isclose(o, h))


def load_all(files):
    data = {}
    for f in files:
        t = os.path.basename(f).replace(".csv", "")
        try:
            d = pd.read_csv(f, index_col=0, parse_dates=True)
            d = d.dropna(subset=["Close"]).sort_index()
            if len(d) >= 200:
                data[t] = d
        except Exception as e:
            print(f"  gagal baca {t}: {str(e)[:40]}")
    return data


def find_bad_dates(data):
    """Tarikh di mana sebahagian besar counter beku."""
    froz = {}   # tarikh -> bil beku
    pres = {}   # tarikh -> bil counter yang ada data
    for t, d in data.items():
        m = frozen_mask(d)
        for dt in d.index:
            pres[dt] = pres.get(dt, 0) + 1
        for dt in d.index[m]:
            froz[dt] = froz.get(dt, 0) + 1

    bad = []
    for dt, n in froz.items():
        total = pres.get(dt, 1)
        if total >= 5 and n / total >= BAD_DATE_FRAC:
            bad.append({"date": dt, "frozen": n, "present": total,
                        "frac": round(n / total, 3)})
    return pd.DataFrame(bad).sort_values("date") if bad else pd.DataFrame()


def main():
    os.makedirs(CLEANDIR, exist_ok=True)

    files = sorted(glob.glob(f"{RAWDIR}/*.csv"))
    files = [f for f in files if not os.path.basename(f).startswith("_")]
    if not files:
        print(f"Tiada fail dalam {RAWDIR}/")
        return

    print(f"Memuat {len(files)} fail...")
    data = load_all(files)
    print(f"Berjaya: {len(data)}\n")

    # ---- LANGKAH 1: tarikh buruk global ----
    bad = find_bad_dates(data)
    if len(bad):
        bad.to_csv(f"{CLEANDIR}/_bad_dates.csv", index=False)
        print(f"TARIKH BURUK GLOBAL: {len(bad)}")
        print("(tarikh di mana majoriti counter beku = pasaran tutup)\n")
        print(bad.head(25).to_string(index=False))
        if len(bad) > 25:
            print(f"... dan {len(bad)-25} lagi")
        per_year = bad["date"].dt.year.value_counts().sort_index()
        print(f"\nPer tahun:\n{per_year.to_string()}")
        badset = set(bad["date"])
    else:
        print("Tiada tarikh buruk global dikesan.")
        badset = set()

    # ---- LANGKAH 2-4: bersih per counter ----
    print("\n" + "=" * 70)
    rows = []
    for t, d in data.items():
        n0 = len(d)
        d = d[~d.index.isin(badset)]
        after_global = len(d)

        m = frozen_mask(d)
        resid_pct = round(m.mean() * 100, 2)

        stat = {
            "ticker": t,
            "rows_raw": n0,
            "removed_global": n0 - after_global,
            "residual_frozen_pct": resid_pct,
        }

        if resid_pct > MAX_RESIDUAL_PCT:
            stat.update(status="DROP", why=f"baki beku {resid_pct}%",
                        rows_clean=0, max_gap_days=0)
            rows.append(stat)
            continue

        d = d[~m]
        d = d.drop(columns=[c for c in ["Open", "Adj Close"] if c in d.columns])

        gaps = d.index.to_series().diff().dt.days
        mx = int(gaps.max()) if len(gaps.dropna()) else 0

        stat["rows_clean"] = len(d)
        stat["max_gap_days"] = mx

        if len(d) < MIN_ROWS_AFTER:
            stat.update(status="DROP", why="tak cukup baris")
        elif mx > 10:
            stat.update(status="FLAG", why=f"jurang {mx}h")
            d.to_csv(f"{CLEANDIR}/{t}.csv")
        else:
            stat.update(status="KEEP", why="")
            d.to_csv(f"{CLEANDIR}/{t}.csv")

        rows.append(stat)

    rep = pd.DataFrame(rows)
    cols = ["ticker", "status", "why", "rows_raw", "removed_global",
            "residual_frozen_pct", "rows_clean", "max_gap_days"]
    rep = rep.reindex(columns=cols).sort_values("residual_frozen_pct",
                                                ascending=False)
    rep.to_csv(f"{CLEANDIR}/_clean_report2.csv", index=False)

    print(rep.to_string(index=False))
    print("\n" + "=" * 70)
    print(rep["status"].value_counts().to_string())

    # ---- di mana jurang jatuh? penting untuk pembahagian train/test ----
    flagged = rep[rep["status"] == "FLAG"]["ticker"].tolist()
    if flagged:
        print("\n" + "=" * 70)
        print("LOKASI JURANG (counter FLAG)")
        print("=" * 70)
        for t in flagged:
            d = pd.read_csv(f"{CLEANDIR}/{t}.csv", index_col=0,
                            parse_dates=True)
            g = d.index.to_series().diff().dt.days
            big = g[g > 10]
            for dt, n in big.items():
                print(f"  {t:<10} {str(dt.date())}  jurang {int(n)} hari")

    print(f"\nFail bersih: {CLEANDIR}/")


if __name__ == "__main__":
    main()
