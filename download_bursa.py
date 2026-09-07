"""
Download + verify Bursa Malaysia daily OHLCV data.

Usage:
    python download_bursa.py

Output:
    data/raw/<TICKER>.csv       - one file per stock
    data/raw/_manifest.csv      - what downloaded, what failed, what to verify
"""

import os
import time
import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
START = "2015-01-01"
END   = "2026-09-04"
OUTDIR = "data/raw"
SLEEP = 0.6          # jeda antara request, elak rate limit

# ----------------------------------------------------------------------
# VERIFIED: FBM KLCI 30 constituents (Bursa semi-annual review Dec 2024)
# Nama di sini adalah nama yang KAU JANGKA. Script akan bandingkan dengan
# nama sebenar dari Yahoo. Kalau tak padan -> flagged.
# ----------------------------------------------------------------------
KLCI30 = {
    "1155.KL": "Malayan Banking",
    # "1295.KL": "Public Bank",
    # "1023.KL": "CIMB Group",
    # "1066.KL": "RHB Bank",
    # "5819.KL": "Hong Leong Bank",
    # "1082.KL": "Hong Leong Financial",
    # "5347.KL": "Tenaga Nasional",
    # "6742.KL": "YTL Power",
    # "4677.KL": "YTL Corporation",
    # "6947.KL": "CelcomDigi",
    # "6012.KL": "Maxis",
    # "4863.KL": "Telekom Malaysia",
    # "6888.KL": "Axiata",
    # "5225.KL": "IHH Healthcare",
    # "1961.KL": "IOI Corporation",
    # "2445.KL": "Kuala Lumpur Kepong",
    # "5285.KL": "SD Guthrie",
    # "4065.KL": "PPB Group",
    # "4197.KL": "Sime Darby",
    # "5183.KL": "Petronas Chemicals",
    # "5681.KL": "Petronas Dagangan",
    # "6033.KL": "Petronas Gas",
    # "3816.KL": "MISC",
    # "8869.KL": "Press Metal",
    # "4707.KL": "Nestle Malaysia",
    # "7084.KL": "QL Resources",
    # "5296.KL": "Mr DIY",
    # "5326.KL": "99 Speed Mart",
    # "5398.KL": "Gamuda",
    # "5211.KL": "Sunway",
}

# ----------------------------------------------------------------------
# UNVERIFIED: mid-cap candidates. Kod ni aku TAK dapat sahkan.
# Script akan check nama - kalau tak padan, buang.
# ----------------------------------------------------------------------
MIDCAP_CANDIDATES = {
    "3182.KL": "Genting",
    "4715.KL": "Genting Malaysia",
    "1015.KL": "AMMB",
    "2488.KL": "Alliance Bank",
    "5258.KL": "Bank Islam",
    "5168.KL": "Hartalega",
    "7277.KL": "Dialog Group",
    "7293.KL": "Yinson",
    "5246.KL": "Westports",
    "5014.KL": "Malaysia Airports",
    "3336.KL": "IJM",
    "0138.KL": "MY EG Services",
    "0166.KL": "Inari Amertron",
    "0128.KL": "Frontken",
    "0208.KL": "Greatech",
    "7160.KL": "Pentamaster",
    "0097.KL": "ViTrox",
    "5031.KL": "Time dotCom",
    "5099.KL": "Capital A",
    "5176.KL": "Sunway REIT",
}

# Market index - untuk market-level features
INDEX = {"^KLSE": "FTSE Bursa Malaysia KLCI"}


def flatten(df):
    """yfinance kadang pulangkan MultiIndex columns. Ratakan."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


def name_matches(expected, actual):
    """Padanan longgar - abaikan Berhad/Bhd/casing/punctuation."""
    if not actual:
        return None                      # tak dapat nama, tak boleh judge
    def norm(s):
        s = s.lower()
        for junk in ["berhad", "bhd", "group", "holdings", "(m)", "corporation",
                     "corp", ".", ",", "-", "&"]:
            s = s.replace(junk, " ")
        return " ".join(s.split())
    e, a = norm(expected), norm(actual)
    first = e.split()[0] if e else ""
    return first in a or e in a or a in e


def fetch(ticker, expected_name):
    """Download satu ticker. Pulangkan dict hasil."""
    row = {
        "ticker": ticker,
        "expected": expected_name,
        "actual": "",
        "rows": 0,
        "start": "",
        "end": "",
        "status": "",
    }

    try:
        tk = yf.Ticker(ticker)

        # nama sebenar
        try:
            row["actual"] = tk.info.get("longName") or tk.info.get("shortName") or ""
        except Exception:
            row["actual"] = ""

        df = yf.download(
            ticker, start=START, end=END,
            progress=False, auto_adjust=True,
        )
        df = flatten(df)

        if df is None or len(df) == 0:
            row["status"] = "NO_DATA"
            return row, None

        df = df.dropna(subset=["Close"])
        row["rows"] = len(df)
        row["start"] = str(df.index[0].date())
        row["end"] = str(df.index[-1].date())

        match = name_matches(expected_name, row["actual"])
        if match is False:
            row["status"] = "NAME_MISMATCH"
        elif match is None:
            row["status"] = "OK_NAME_UNKNOWN"
        elif row["rows"] < 1000:
            row["status"] = "OK_SHORT_HISTORY"
        else:
            row["status"] = "OK"

        return row, df

    except Exception as e:
        row["status"] = f"ERROR: {str(e)[:40]}"
        return row, None


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    universe = {}
    universe.update(INDEX)
    universe.update(KLCI30)
    universe.update(MIDCAP_CANDIDATES)

    print(f"Universe: {len(universe)} ticker")
    print(f"Period  : {START} -> {END}")
    print("=" * 72)

    manifest = []

    for i, (ticker, expected) in enumerate(universe.items(), 1):
        print(f"[{i:>2}/{len(universe)}] {ticker:<10}", end=" ", flush=True)

        row, df = fetch(ticker, expected)
        manifest.append(row)

        if df is not None:
            safe = ticker.replace("^", "IDX_")
            df.to_csv(f"{OUTDIR}/{safe}.csv")
            print(f"{row['rows']:>5} rows  {row['status']:<18} {row['actual'][:34]}")
        else:
            print(f"{'':>5}       {row['status']}")

        time.sleep(SLEEP)

    # simpan manifest
    mf = pd.DataFrame(manifest)
    mf.to_csv(f"{OUTDIR}/_manifest.csv", index=False)

    print("=" * 72)
    print(mf["status"].str.split(":").str[0].value_counts().to_string())

    bad = mf[~mf["status"].str.startswith("OK")]
    if len(bad):
        print("\nPERLU SEMAK:")
        for _, r in bad.iterrows():
            print(f"  {r['ticker']:<10} jangka='{r['expected']}'  "
                  f"dapat='{r['actual']}'  [{r['status']}]")

    print(f"\nFail disimpan: {OUTDIR}/")
    print("Manifest     : {}/_manifest.csv".format(OUTDIR))


if __name__ == "__main__":
    main()
