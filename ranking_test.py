"""
Rangka ujian PENARAFAN - 20 hari, tak bertindih.

Soalan sebenar: bolehkah model memilih subset saham yang pulangan
                bersyaratnya cukup tinggi untuk menampung kos?

Perbezaan utama dari eksperimen lepas:
  1. PENARAFAN, bukan klasifikasi. Setiap tarikh, tarafkan semua saham,
     ambil N teratas.
  2. TAK BERTINDIH. Sampel setiap 20 hari bursa. n=140 sebenar,
     bukan n=6000 palsu. Nilai-p sebelum ni terlalu optimistik.
  3. TANDA ARAS = penarafan rawak, bukan sifar. Rawak dapat 56.5%
     pada 20 hari kerana hanyut. Model kena kalahkan ITU.
  4. Universe penuh (43 saham), bukan 14. Penyebaran lebih bermakna.

Penaraf yang diuji:
    R_random     - kawalan
    M_momentum   - pulangan 60 hari lepas
    V_lowvol     - volatiliti terendah
    S_rsi        - RSI terendah (mean reversion)
    L_linear     - regresi linear atas feature
    G_lgbm       - LightGBM (kalau dipasang)

Usage:
    python ranking_test.py --split train --top 5
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import stats

CLEANDIR = "data/clean"
UNADJDIR = "data/unadj"
HOLD = 20
STEP = 20                    # tak bertindih
SEED = 42

STAMP_DUTY_PCT = 0.10
CLEARING_PCT = 0.03
BROKERAGE_FLAT = 8.0

TRAIN_END = "2021-12-31"
VAL_END = "2023-12-31"


def tick_size(p):
    if p < 1.00: return 0.005
    if p < 10.00: return 0.01
    if p < 100.00: return 0.02
    return 0.10


def cost_pct(price, capital):
    shares = capital / price
    return ((BROKERAGE_FLAT * 2
             + capital * STAMP_DUTY_PCT / 100 * 2
             + capital * CLEARING_PCT / 100 * 2
             + shares * tick_size(price)) / capital * 100)


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build_panel(split):
    """Panel sejajar: baris = tarikh, lajur = saham."""
    files = [f for f in sorted(glob.glob(f"{CLEANDIR}/*.csv"))
             if not os.path.basename(f).startswith(("_", "IDX_"))]
    feats, closes, unadj = {}, {}, {}

    for f in files:
        t = os.path.basename(f).replace(".csv", "")
        d = pd.read_csv(f, index_col=0, parse_dates=True).sort_index()
        if len(d) < 400:
            continue
        c = d["Close"]
        feats[t] = pd.DataFrame({
            "ret_5":   c / c.shift(5) - 1,
            "ret_20":  c / c.shift(20) - 1,
            "ret_60":  c / c.shift(60) - 1,
            "rsi":     rsi(c),
            "vol_20":  c.pct_change().rolling(20).std(),
            "ma_dist": c / c.rolling(50).mean() - 1,
            "vol_chg": np.log(d["Volume"] /
                              d["Volume"].rolling(20).mean().replace(0, np.nan)),
        })
        closes[t] = c
        up = f"{UNADJDIR}/{t}.csv"
        if os.path.exists(up):
            unadj[t] = pd.read_csv(up, index_col=0, parse_dates=True)["Close"]

    px = pd.DataFrame(closes).sort_index()
    if split == "train":
        px = px[px.index <= TRAIN_END]
    elif split == "val":
        px = px[(px.index > TRAIN_END) & (px.index <= VAL_END)]
    elif split == "test":
        px = px[px.index > VAL_END]
    return px, feats, unadj


def forward_return(px, dt, h):
    """Pulangan h hari dagangan hadapan, masuk pada Close hari berikut."""
    i = px.index.get_loc(dt)
    if i + 1 + h >= len(px):
        return None
    e = px.iloc[i + 1]
    x = px.iloc[i + 1 + h]
    return ((x / e - 1) * 100), e


RANKERS = ["R_random", "M_momentum", "V_lowvol", "S_rsi",
           "L_linear", "G_lgbm"]


def rank_scores(name, dt, tickers, feats, rng, model=None):
    """Skor lebih tinggi = lebih menarik."""
    rows = {}
    for t in tickers:
        f = feats.get(t)
        if f is None or dt not in f.index:
            continue
        r = f.loc[dt]
        if r.isna().any():
            continue
        rows[t] = r
    if len(rows) < 8:
        return None
    X = pd.DataFrame(rows).T

    if name == "R_random":
        return pd.Series(rng.random(len(X)), index=X.index)
    if name == "M_momentum":
        return X["ret_60"]
    if name == "V_lowvol":
        return -X["vol_20"]
    if name == "S_rsi":
        return -X["rsi"]
    if name in ("L_linear", "G_lgbm") and model is not None:
        return pd.Series(model.predict(X.values), index=X.index)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train",
                    choices=["train", "val", "test"])
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--capital", type=float, default=25000)
    a = ap.parse_args()

    px, feats, unadj = build_panel(a.split)
    tickers = list(px.columns)
    print(f"Universe : {len(tickers)} saham")
    print(f"Split    : {a.split}   {px.index[0].date()} - {px.index[-1].date()}")
    print(f"Pegang   : {HOLD} hari, tak bertindih (setiap {STEP} hari)")
    print(f"Top-N    : {a.top}\n")

    dates = px.index[60::STEP]
    dates = [d for d in dates
             if px.index.get_loc(d) + 1 + HOLD < len(px)]
    print(f"Tetingkap bebas: {len(dates)}\n")

    # ---- latih model atas train sahaja bila split bukan train ----
    models = {"L_linear": None, "G_lgbm": None}
    tr_px, tr_feats, _ = build_panel("train")
    Xs, ys = [], []
    for dt in tr_px.index[60::STEP]:
        fr = forward_return(tr_px, dt, HOLD)
        if fr is None:
            continue
        ret, _ = fr
        for t in tr_px.columns:
            f = tr_feats.get(t)
            if f is None or dt not in f.index:
                continue
            r = f.loc[dt]
            if r.isna().any() or pd.isna(ret.get(t)):
                continue
            Xs.append(r.values)
            ys.append(ret[t])
    if Xs:
        X, y = np.array(Xs), np.array(ys)
        print(f"Sampel latihan: {len(X)}")
        from sklearn.linear_model import Ridge
        models["L_linear"] = Ridge(alpha=1.0).fit(X, y)
        try:
            import lightgbm as lgb
            models["G_lgbm"] = lgb.LGBMRegressor(
                n_estimators=120, max_depth=3, learning_rate=0.05,
                verbose=-1).fit(X, y)
        except ImportError:
            print("lightgbm tiada - langkau (pip install lightgbm)")

    # ---- jalankan penarafan ----
    rng = np.random.default_rng(SEED)
    results = {k: [] for k in RANKERS}

    for dt in dates:
        fr = forward_return(px, dt, HOLD)
        if fr is None:
            continue
        ret, entry_px = fr

        for name in RANKERS:
            if name in models and models[name] is None:
                continue
            sc = rank_scores(name, dt, tickers, feats, rng, models.get(name))
            if sc is None:
                continue
            top = sc.sort_values(ascending=False).head(a.top).index
            vals, costs = [], []
            for t in top:
                if pd.isna(ret.get(t)):
                    continue
                p = float(unadj.get(t, {}).get(
                    px.index[px.index.get_loc(dt) + 1], entry_px[t])
                    if t in unadj else entry_px[t])
                vals.append(ret[t])
                costs.append(cost_pct(p, a.capital))
            if vals:
                g = float(np.mean(vals))
                c = float(np.mean(costs))
                results[name].append({"date": dt, "gross": g,
                                      "cost": c, "net": g - c})

    rows = []
    for name, tr in results.items():
        if len(tr) < 10:
            continue
        d = pd.DataFrame(tr)
        n = len(d)
        t_stat, p_t = stats.ttest_1samp(d["net"], 0)
        rows.append({
            "ranker": name, "windows": n,
            "gross_mean": round(d["gross"].mean(), 2),
            "gross_med": round(d["gross"].median(), 2),
            "cost": round(d["cost"].mean(), 3),
            "net_mean": round(d["net"].mean(), 2),
            "net_med": round(d["net"].median(), 2),
            "win_pct": round((d["net"] > 0).mean() * 100, 1),
            "p_vs_zero": round(p_t, 3),
            "_net": d["net"].values,
        })

    res = pd.DataFrame(rows).sort_values("net_mean", ascending=False)
    show = [c for c in res.columns if not c.startswith("_")]
    res[show].to_csv(f"data/ranking_{a.split}_top{a.top}.csv", index=False)
    print("\n" + "=" * 84)
    print(res[show].to_string(index=False))

    # ---- vs penarafan rawak: ini tanda aras sebenar ----
    rnd = res[res["ranker"] == "R_random"]
    if len(rnd):
        base = rnd.iloc[0]["_net"]
        print("\n" + "=" * 84)
        print("BERBANDING PENARAFAN RAWAK  (tanda aras sebenar)")
        print("=" * 84)
        print(f"Rawak net purata: {base.mean():+.2f}%  "
              f"({len(base)} tetingkap)\n")
        for _, r in res.iterrows():
            if r["ranker"] == "R_random":
                continue
            v = r["_net"]
            m = min(len(v), len(base))
            t_stat, p = stats.ttest_rel(v[:m], base[:m])
            diff = v[:m].mean() - base[:m].mean()
            verdict = ("BAIK" if p < 0.05 and diff > 0 else
                       "TERUK" if p < 0.05 and diff < 0 else "sama")
            print(f"{r['ranker']:<14} {diff:+7.2f}pp  p={p:.3f}  {verdict}")

    print("\nCatatan: 'sama' bermakna penaraf itu tidak menyumbang")
    print("apa-apa di atas memegang saham rawak selama 20 hari.")


if __name__ == "__main__":
    main()
