#!/usr/bin/env python3
"""
📓 Paper Trading Journal

Forward testing, not backtesting. Every trade here is recorded before
the outcome is known, on data nobody has seen yet -- which is the only
honest test a trading idea gets.

The part that matters most: each entry records WHY you took it. Model
signal, your own read, news, or a mix. After thirty or forty trades the
journal can tell you something no backtest can -- whether your own
judgement beats the model, or the other way round.

Run:  streamlit run paper_trade.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date
import json
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="📓 Paper Trading",
    page_icon="📓",
    layout="wide",
    initial_sidebar_state="expanded"
)

JOURNAL_FILE = "paper_trades.json"

# Same cost model as the backtest, so paper results are comparable.
COST_CONFIG = {
    'brokerage_pct': 0.0042,
    'brokerage_min': 8.00,
    'stamp_duty_per_1000': 1.00,
    'clearing_pct': 0.0003,
}

REASONS = [
    "Signal model",
    "Analisis sendiri",
    "Berita / fundamental",
    "Model + analisis sendiri",
    "Lain-lain",
]


# ============================================================================
# STORAGE
# ============================================================================

REQUIRED_FIELDS = ('ticker', 'entry_price', 'shares', 'entry_date')


def load_journal():
    """
    Read the journal from disk.

    Entries missing required fields are dropped rather than allowed
    through -- a half-written record would otherwise produce an empty
    DataFrame downstream and fail with a confusing KeyError.
    """
    if not os.path.exists(JOURNAL_FILE):
        return []

    try:
        with open(JOURNAL_FILE, 'r') as f:
            raw = json.load(f)
    except Exception as e:
        st.error(f"Jurnal tak boleh dibaca: {e}")
        return []

    if not isinstance(raw, list):
        st.error(
            f"Format jurnal salah — dijangka senarai, dapat "
            f"{type(raw).__name__}. Padam {JOURNAL_FILE} untuk mula semula."
        )
        return []

    good, bad = [], 0
    for item in raw:
        if isinstance(item, dict) and all(k in item for k in REQUIRED_FIELDS):
            good.append(item)
        else:
            bad += 1

    if bad:
        st.warning(
            f"⚠️ {bad} rekod dilangkau kerana tidak lengkap. "
            f"{len(good)} rekod sah dimuatkan."
        )

    return good


def save_journal(trades):
    with open(JOURNAL_FILE, 'w') as f:
        json.dump(trades, f, indent=2, default=str)


def round_trip_cost(position_value):
    """Buy + sell cost in RM, matching the backtest's assumptions."""
    c = COST_CONFIG
    brokerage = max(position_value * c['brokerage_pct'], c['brokerage_min'])
    stamp = np.ceil(position_value / 1000) * c['stamp_duty_per_1000']
    clearing = position_value * c['clearing_pct']
    return (brokerage + stamp + clearing) * 2


# ============================================================================
# PRICES
# ============================================================================

@st.cache_resource
def load_yfinance():
    import yfinance as yf
    return yf


@st.cache_data(ttl=600, show_spinner=False)
def get_price(ticker):
    """Latest close. Cached ten minutes so the page stays responsive."""
    try:
        yf = load_yfinance()
        d = yf.Ticker(ticker).history(period='5d')
        if len(d) == 0:
            return None
        return float(d.iloc[-1]['Close'])
    except Exception:
        return None


# ============================================================================
# ANALYSIS
# ============================================================================

EMPTY_COLUMNS = [
    'id', 'ticker', 'entry_price', 'shares', 'entry_date', 'reason',
    'notes', 'target', 'stop', 'exit_price', 'exit_date', 'status',
    'current_price', 'position_value', 'cost_rm', 'gross_rm',
    'net_rm', 'gross_pct', 'net_pct',
]


def enrich(trades):
    """
    Attach live prices and P&L to every trade.

    Always returns a frame carrying the expected columns, even when
    every row fails -- callers filter on 'status', and an empty frame
    without columns would raise KeyError instead of showing nothing.
    """
    rows = []
    skipped = []

    for t in trades:
        try:
            r = dict(t)
            entry = float(r['entry_price'])
            shares = int(r['shares'])

            if entry <= 0 or shares <= 0:
                skipped.append(r.get('ticker', '?'))
                continue

            position = entry * shares
            cost = round_trip_cost(position)

            if r.get('exit_price'):
                exit_p = float(r['exit_price'])
                r['status'] = 'CLOSED'
            else:
                live = get_price(r['ticker'])
                exit_p = live if live else entry
                r['status'] = 'OPEN'

            gross = (exit_p - entry) * shares
            net = gross - cost

            r.update({
                'current_price': exit_p,
                'position_value': position,
                'cost_rm': cost,
                'gross_rm': gross,
                'net_rm': net,
                'gross_pct': (exit_p - entry) / entry * 100,
                'net_pct': net / position * 100,
            })
            rows.append(r)

        except (KeyError, TypeError, ValueError) as e:
            skipped.append(f"{t.get('ticker', '?')} ({e})")

    if skipped:
        st.warning(f"⚠️ Trade dilangkau: {', '.join(map(str, skipped))}")

    if not rows:
        return pd.DataFrame(columns=EMPTY_COLUMNS)

    return pd.DataFrame(rows)


def summarise(df, subset=None):
    """Headline numbers for a set of trades."""
    if df.empty:
        return None
    d = df if subset is None else df[subset]
    if d.empty:
        return None

    closed = d[d['status'] == 'CLOSED']

    return {
        'n': len(d),
        'n_closed': len(closed),
        'invested': d['position_value'].sum(),
        'gross_rm': d['gross_rm'].sum(),
        'cost_rm': d['cost_rm'].sum(),
        'net_rm': d['net_rm'].sum(),
        'net_pct': (
            d['net_rm'].sum() / d['position_value'].sum() * 100
            if d['position_value'].sum() else 0
        ),
        'win_rate': (
            (closed['net_rm'] > 0).mean() * 100 if len(closed) else 0
        ),
    }


# ============================================================================
# UI
# ============================================================================

st.title("📓 Paper Trading Journal")
st.markdown(
    "Ujian ke hadapan. Setiap trade direkod **sebelum** keputusannya "
    "diketahui — itu satu-satunya ujian jujur yang idea dagangan dapat."
)

trades = load_journal()

with st.sidebar:
    st.header("➕ Trade Baru")

    with st.form("new_trade", clear_on_submit=True):
        ticker = st.text_input("Ticker", value="4456.KL").strip().upper()

        col_a, col_b = st.columns(2)
        entry_price = col_a.number_input(
            "Harga masuk (RM)", min_value=0.001, value=0.470,
            step=0.001, format="%.4f"
        )
        shares = col_b.number_input(
            "Unit", min_value=100, value=10000, step=100
        )

        entry_date = st.date_input("Tarikh masuk", value=date.today())

        reason = st.selectbox("Kenapa trade ini?", REASONS)

        st.caption(
            "Rekod sebab sekarang, sebelum kau tahu hasilnya. "
            "Selepas 30 trade, jurnal boleh beritahu sama ada firasat "
            "kau atau model yang lebih baik."
        )

        notes = st.text_area(
            "Nota (pilihan)",
            placeholder="Apa yang kau nampak? Apa yang boleh buat kau salah?",
            height=80
        )

        target = st.number_input(
            "Sasaran keluar (RM, 0 = tiada)",
            min_value=0.0, value=0.0, step=0.001, format="%.4f"
        )
        stop = st.number_input(
            "Stop loss (RM, 0 = tiada)",
            min_value=0.0, value=0.0, step=0.001, format="%.4f"
        )

        if st.form_submit_button("Rekod trade", type="primary"):
            if not ticker:
                st.error("Ticker kosong")
            else:
                trades.append({
                    'id': int(datetime.now().timestamp()),
                    'ticker': ticker,
                    'entry_price': entry_price,
                    'shares': shares,
                    'entry_date': str(entry_date),
                    'reason': reason,
                    'notes': notes,
                    'target': target if target > 0 else None,
                    'stop': stop if stop > 0 else None,
                    'exit_price': None,
                    'exit_date': None,
                })
                save_journal(trades)
                st.success(f"Direkod: {ticker}")
                st.rerun()

if not trades:
    st.info(
        "Jurnal kosong. Rekod trade pertama di sidebar.\n\n"
        "Cadangan: paper trade sekurang-kurangnya 30 kali sebelum "
        "pertimbangkan duit sebenar. Tiga puluh sampel adalah minimum "
        "kasar sebelum win rate bermakna apa-apa."
    )
    st.stop()

with st.spinner("Mengambil harga semasa..."):
    df = enrich(trades)

if df.empty:
    st.error(
        "Tiada trade sah untuk dipaparkan. Semua rekod dalam jurnal "
        "gagal diproses — lihat amaran di atas."
    )
    with st.expander("Lihat kandungan jurnal mentah"):
        st.json(trades)
    st.stop()

# --- Headline ---
overall = summarise(df)
open_df = df[df['status'] == 'OPEN']
closed_df = df[df['status'] == 'CLOSED']

c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah trade", overall['n'],
          f"{len(open_df)} terbuka · {len(closed_df)} tutup")
c2.metric("Modal digunakan", f"RM {overall['invested']:,.0f}")
c3.metric("Untung/rugi bersih", f"RM {overall['net_rm']:+,.2f}",
          f"{overall['net_pct']:+.2f}%")
c4.metric("Kos ditelan", f"RM {overall['cost_rm']:,.2f}",
          f"{overall['cost_rm'] / overall['invested'] * 100:.2f}% modal")

if overall['n_closed'] > 0:
    st.caption(
        f"Win rate pada {overall['n_closed']} trade tertutup: "
        f"{overall['win_rate']:.1f}%"
    )
    if overall['n_closed'] < 30:
        st.caption(
            f"⚠️ {overall['n_closed']} trade sahaja. Di bawah 30, win "
            f"rate lebih banyak nasib daripada kemahiran."
        )

st.markdown("---")

# --- Open positions ---
if not open_df.empty:
    st.subheader("📂 Posisi Terbuka")

    for _, r in open_df.iterrows():
        dp = 4 if r['entry_price'] < 5 else 2
        with st.container():
            k1, k2, k3, k4, k5 = st.columns([2, 1, 1, 1, 1])

            k1.markdown(
                f"**{r['ticker']}** · {r['shares']:,} unit  \n"
                f"<small>{r['reason']} · {r['entry_date']}</small>",
                unsafe_allow_html=True
            )
            k2.metric("Masuk", f"{r['entry_price']:.{dp}f}")
            k3.metric("Semasa", f"{r['current_price']:.{dp}f}",
                      f"{r['gross_pct']:+.2f}%")
            k4.metric("Bersih", f"RM {r['net_rm']:+,.2f}",
                      f"{r['net_pct']:+.2f}%")

            with k5:
                st.write("")
                if st.button("Tutup", key=f"close_{r['id']}"):
                    st.session_state[f'closing_{r["id"]}'] = True

            # Flag target/stop breaches -- the plan you set at entry
            if r.get('target') and r['current_price'] >= r['target']:
                st.success(
                    f"🎯 {r['ticker']} cecah sasaran RM {r['target']:.{dp}f}"
                )
            if r.get('stop') and r['current_price'] <= r['stop']:
                st.error(
                    f"🛑 {r['ticker']} tembus stop RM {r['stop']:.{dp}f}"
                )

            if st.session_state.get(f'closing_{r["id"]}'):
                with st.form(f"close_form_{r['id']}"):
                    fc1, fc2 = st.columns(2)
                    px = fc1.number_input(
                        "Harga keluar", value=float(r['current_price']),
                        step=0.001, format="%.4f", key=f"px_{r['id']}"
                    )
                    dt = fc2.date_input(
                        "Tarikh keluar", value=date.today(),
                        key=f"dt_{r['id']}"
                    )
                    if st.form_submit_button("Sahkan tutup"):
                        for t in trades:
                            if t['id'] == r['id']:
                                t['exit_price'] = px
                                t['exit_date'] = str(dt)
                        save_journal(trades)
                        st.session_state.pop(f'closing_{r["id"]}', None)
                        st.rerun()

            if r.get('notes'):
                st.caption(f"📝 {r['notes']}")

            st.markdown("---")

# --- Which decisions worked? ---
if len(closed_df) >= 3:
    st.subheader("🔍 Sumber Keputusan")
    st.caption(
        "Ini yang backtest tak boleh beritahu: bila kau ikut model dan "
        "bila kau ikut kepala sendiri, mana yang jadi?"
    )

    breakdown = []
    for reason in REASONS:
        sub = closed_df[closed_df['reason'] == reason]
        if sub.empty:
            continue
        breakdown.append({
            'Sumber': reason,
            'Trade': len(sub),
            'Win rate': f"{(sub['net_rm'] > 0).mean() * 100:.0f}%",
            'Purata bersih': f"{sub['net_pct'].mean():+.2f}%",
            'Jumlah': f"RM {sub['net_rm'].sum():+,.2f}",
        })

    if breakdown:
        st.dataframe(pd.DataFrame(breakdown),
                     use_container_width=True, hide_index=True)
        st.caption(
            "Sampel kecil, jadi jangan buat kesimpulan besar terlalu "
            "awal. Tapi kalau satu sumber terus-menerus rugi selepas "
            "20-30 trade, itu isyarat yang patut dipercayai."
        )

    st.markdown("---")

# --- Equity curve ---
if len(closed_df) >= 2:
    st.subheader("📈 Prestasi Terkumpul")

    cd = closed_df.copy()
    cd['exit_date'] = pd.to_datetime(cd['exit_date'])
    cd = cd.sort_values('exit_date')
    cd['cum_net'] = cd['net_rm'].cumsum()
    cd['cum_gross'] = cd['gross_rm'].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cd['exit_date'], y=cd['cum_gross'],
        mode='lines+markers', name='Sebelum kos',
        line=dict(color='#9aa0a6', width=1.6, dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=cd['exit_date'], y=cd['cum_net'],
        mode='lines+markers', name='Selepas kos',
        line=dict(color='#2ed573', width=2)
    ))
    fig.add_hline(y=0, line=dict(color='#ced4da', width=1))
    fig.update_layout(
        height=340, margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title='Untung/rugi terkumpul (RM)',
        legend=dict(orientation='h', y=1.14, x=0),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    gap = cd['cum_gross'].iloc[-1] - cd['cum_net'].iloc[-1]
    st.caption(
        f"Jurang antara dua garisan ialah kos: RM {gap:,.2f} setakat ini. "
        f"Kalau garisan hijau di bawah sifar sedangkan yang kelabu di "
        f"atas, strategi kau menang di atas kertas dan rugi dalam realiti."
    )

    st.markdown("---")

# --- Closed trades table ---
if not closed_df.empty:
    with st.expander(f"📋 Semua {len(closed_df)} trade tertutup"):
        show = closed_df[[
            'ticker', 'entry_date', 'entry_price', 'exit_date',
            'exit_price', 'shares', 'reason', 'gross_pct',
            'cost_rm', 'net_rm'
        ]].copy()
        show.columns = [
            'Ticker', 'Masuk', 'Harga masuk', 'Keluar', 'Harga keluar',
            'Unit', 'Sumber', 'Kasar %', 'Kos RM', 'Bersih RM'
        ]
        show['Kasar %'] = show['Kasar %'].map(lambda v: f"{v:+.2f}%")
        show['Kos RM'] = show['Kos RM'].map(lambda v: f"{v:,.2f}")
        show['Bersih RM'] = show['Bersih RM'].map(lambda v: f"{v:+,.2f}")
        st.dataframe(show, use_container_width=True, hide_index=True)

# --- Housekeeping ---
with st.expander("⚙️ Urus jurnal"):
    st.caption(f"Disimpan di `{JOURNAL_FILE}` ({len(trades)} rekod)")

    st.download_button(
        "Muat turun jurnal (JSON)",
        data=json.dumps(trades, indent=2, default=str),
        file_name=f"paper_trades_{date.today()}.json",
        mime="application/json"
    )

    if st.checkbox("Saya nak padam semua rekod"):
        if st.button("Padam jurnal", type="secondary"):
            save_journal([])
            st.rerun()

st.markdown("---")
st.caption(
    "Paper trading tiada risiko kewangan tetapi juga tiada tekanan "
    "emosi sebenar. Kebanyakan orang berdagang lebih tenang di atas "
    "kertas berbanding dengan duit sendiri — ambil kira itu bila "
    "menilai keputusan kau."
)