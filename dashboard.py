"""
INSIDELINE | TERMINAL v2.0
Run: streamlit run dashboard.py
"""

from __future__ import annotations

import os
import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Credentials ──────────────────────────────────────────────────────────────
CREDS_PATH = os.environ.get("GOOGLE_SHEETS_CREDS_PATH", "")
SHEET_ID   = os.environ.get("GOOGLE_SHEETS_SPREADSHEET_ID", "")
SCOPES     = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# ─── Design tokens ────────────────────────────────────────────────────────────
BG          = "#06080a"
BG_PANEL    = "#0d1117"
BG_SURFACE  = "#111827"
BORDER      = "#1e293b"
BORDER_GRID = "#1e293b"
TEXT        = "#f8fafc"
TEXT_MUTED  = "#64748b"
ACCENT      = "#00ff88"       # primary green
BLUE        = "#2563eb"       # sportsbook blue
GREEN       = "#22c55e"       # success green
RED         = "#ef4444"       # miss / loss red
YELLOW      = "#eab308"       # B-tier
BREAK_EVEN  = 52.4

GRADE_COLOUR = {
    "[A]":     GREEN,
    "[A-dog]": "#86efac",
    "[B+]":    "#a3a800",   # yellow-green, closer to B yellow than A green
    "[B]":     YELLOW,
    "[C]":     RED,
}
GRADE_LABEL = {"[A]": "A", "[A-dog]": "A-dog", "[B+]": "B+", "[B]": "B", "[C]": "C"}

CHART_LAYOUT = dict(
    plot_bgcolor=BG_PANEL,
    paper_bgcolor=BG,
    font=dict(family="'IBM Plex Mono', 'Roboto Mono', monospace", color=TEXT, size=12),
    xaxis=dict(
        gridcolor=BORDER_GRID, gridwidth=1,
        linecolor=BORDER, tickcolor=BORDER,
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor=BORDER_GRID, gridwidth=1,
        linecolor=BORDER, tickcolor=BORDER,
        zeroline=False,
    ),
    legend=dict(
        bgcolor=BG_PANEL, bordercolor=BORDER, borderwidth=1,
        font=dict(size=11),
    ),
    margin=dict(t=40, b=32, l=16, r=16),
    hoverlabel=dict(bgcolor=BG_PANEL, bordercolor=BORDER, font_color=TEXT),
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
TERMINAL_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=Inter:wght@300;400;500;600&display=swap');

  /* ── Global reset ── */
  html, body, .stApp, [data-testid="stAppViewContainer"] {{
      background-color: {BG} !important;
      color: {TEXT};
      font-family: 'Inter', sans-serif;
  }}
  [data-testid="stHeader"] {{ background: {BG} !important; border-bottom: 1px solid {BORDER}; }}
  [data-testid="stSidebar"] {{ background: {BG_PANEL} !important; }}
  section[data-testid="stMain"] {{ background: {BG} !important; }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: {BG}; }}
  ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 2px; }}

  /* ── Top-level nav tabs (MLB / NBA) ── */
  .stTabs [data-baseweb="tab-list"] {{
      gap: 0;
      border-bottom: 1px solid {BORDER};
      background: {BG};
      padding: 0;
  }}
  .stTabs [data-baseweb="tab"] {{
      background: transparent;
      border-radius: 0;
      border-bottom: 2px solid transparent;
      padding: 10px 28px;
      color: {TEXT_MUTED};
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      transition: color 0.15s, border-color 0.15s;
  }}
  .stTabs [aria-selected="true"] {{
      background: transparent !important;
      border-bottom: 2px solid {ACCENT} !important;
      color: {ACCENT} !important;
  }}
  .stTabs [data-baseweb="tab"]:hover {{
      color: {TEXT} !important;
      background: {BG_SURFACE} !important;
  }}
  /* ── Selectbox (market / sport filter) ── */
  [data-testid="stSelectbox"] label {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.65rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: {TEXT_MUTED};
  }}
  [data-testid="stSelectbox"] > div > div {{
      background: {BG_PANEL} !important;
      border: 1px solid {BORDER} !important;
      border-radius: 0 !important;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.82rem;
      color: {TEXT};
  }}

  /* ── Metrics panel ── */
  [data-testid="stMetric"] {{
      background: {BG_PANEL};
      border: 1px solid {BORDER};
      border-top: 2px solid {ACCENT};
      border-radius: 0;
      padding: 14px 18px;
  }}
  [data-testid="stMetricLabel"] p {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.65rem !important;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: {TEXT_MUTED} !important;
  }}
  [data-testid="stMetricValue"] {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 1.35rem !important;
      font-weight: 600;
      color: {TEXT} !important;
  }}

  /* ── Section headers ── */
  h3 {{
      font-family: 'IBM Plex Mono', monospace !important;
      font-size: 0.7rem !important;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: {TEXT_MUTED} !important;
      border-bottom: 1px solid {BORDER};
      padding-bottom: 6px;
      margin-top: 28px !important;
      margin-bottom: 12px !important;
  }}

  /* ── Info / warning boxes ── */
  [data-testid="stAlert"] {{
      background: {BG_PANEL};
      border: 1px solid {BORDER};
      border-radius: 0;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.8rem;
  }}

  /* ── Dataframe / table ── */
  [data-testid="stDataFrame"] {{
      border: 1px solid {BORDER};
  }}
  iframe[title="st_dataframe"] {{
      background: {BG_PANEL};
  }}

  /* ── Radio (time-filter) ── */
  [data-testid="stRadio"] label {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.72rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
  }}

  /* ── Divider ── */
  hr {{ border-color: {BORDER}; opacity: 1; margin: 16px 0; }}

  /* ── Spinner ── */
  [data-testid="stSpinner"] {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }}

  /* ── Caption ── */
  [data-testid="stCaptionContainer"] p {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
      color: {TEXT_MUTED};
  }}

  /* ── Expander — keep header muted on hover ── */
  [data-testid="stExpander"] summary {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.75rem;
      letter-spacing: 0.06em;
      color: {TEXT_MUTED} !important;
      background: {BG_PANEL} !important;
  }}
  [data-testid="stExpander"] summary:hover {{
      color: {TEXT_MUTED} !important;
      background: {BG_PANEL} !important;
  }}
  [data-testid="stExpander"] summary p {{
      color: {TEXT_MUTED} !important;
  }}
  [data-testid="stExpander"] {{
      border: 1px solid {BORDER} !important;
      border-radius: 0 !important;
      background: {BG_PANEL} !important;
  }}
</style>
"""


# ─── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_sheet_data() -> pd.DataFrame:
    if not CREDS_PATH or not SHEET_ID:
        return pd.DataFrame()
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        creds = Credentials.from_service_account_file(CREDS_PATH, scopes=SCOPES)
        gc    = gspread.authorize(creds)
        sheet = gc.open_by_key(SHEET_ID).sheet1
        rows  = sheet.get_all_values()
    except Exception as exc:
        st.error(f"Sheet connection failed: {exc}")
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    plays = []
    for row in rows[1:]:
        if not row:
            continue
        first = (row[0] or "").strip().upper()
        if first.startswith("CURRENT") or first.startswith("COMPLETED") or first.startswith("FUTURE"):
            continue
        plays.append((row + [""] * 13)[:13])

    if not plays:
        return pd.DataFrame()

    df = pd.DataFrame(plays, columns=[
        "date", "grade", "league", "matchup", "time",
        "market", "side", "line", "proj", "margin", "ev", "wager", "result",
    ])

    for col in ("league", "market"):
        df[col] = df[col].str.strip().str.upper()
    for col in ("grade", "result"):
        df[col] = df[col].str.strip()

    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()  # floor to midnight → clean day-level x-axis
    df["ev_num"]      = pd.to_numeric(
        df["ev"].str.replace("%", "", regex=False)
                 .str.replace("$", "", regex=False)
                 .str.replace("+", "", regex=False),
        errors="coerce",
    )
    df["wager_num"]   = pd.to_numeric(df["wager"].str.replace("$", "", regex=False), errors="coerce")
    df["hit"]         = df["result"].map(
        lambda r: True if r.upper() == "HIT" else (False if r.upper() == "MISS" else None)
    )
    return df


def _today_iso() -> str:
    return dt.date.today().isoformat()


# ─── Shared chart theme helper ────────────────────────────────────────────────
def _apply_chart_theme(fig: go.Figure, **overrides) -> go.Figure:
    layout = {**CHART_LAYOUT, **overrides}
    fig.update_layout(**layout)
    return fig


# ─── Play cards (HTML list-item UI) ───────────────────────────────────────────
def _plays_table(sub: pd.DataFrame) -> None:
    """Render each play as a styled HTML card — no dataframe grid lines."""
    import html as _html

    if sub.empty:
        st.info("No plays found for today.")
        return

    cards: list[str] = []

    for _, row in sub.iterrows():
        grade      = row.get("grade", "")
        matchup    = _html.escape(str(row.get("matchup", "")))
        side       = _html.escape(str(row.get("side", "")))
        market     = _html.escape(str(row.get("market", "")))
        book_line  = _html.escape(str(row.get("line", "—")))
        model_proj = _html.escape(str(row.get("proj", "—")))
        margin     = _html.escape(str(row.get("margin", "—")))
        ev         = _html.escape(str(row.get("ev", "—")))
        wager      = _html.escape(str(row.get("wager", "—")))
        game_time  = _html.escape(str(row.get("time", "")))
        league     = _html.escape(str(row.get("league", "")))

        grade_colour = GRADE_COLOUR.get(grade, TEXT_MUTED)
        grade_label  = GRADE_LABEL.get(grade, grade)

        # Sub-label under the matchup: sport · time (if available)
        meta_parts = [p for p in [league, game_time] if p and p != "nan"]
        meta_line  = " · ".join(meta_parts) if meta_parts else market

        card = (
            f'<div style="background:#111827;border:1px solid {BORDER};border-left:4px solid {grade_colour};'
            f'display:flex;align-items:center;padding:16px 22px;margin-bottom:8px;'
            f'font-family:\'IBM Plex Mono\',\'Roboto Mono\',monospace;">'

            f'<div style="flex:2.2;min-width:0;padding-right:20px;">'
            f'<div style="display:inline-block;color:{grade_colour};font-size:0.6rem;font-weight:700;'
            f'letter-spacing:0.14em;text-transform:uppercase;border:1px solid {grade_colour}55;'
            f'padding:1px 6px;margin-bottom:6px;">{grade_label}</div>'
            f'<div style="color:{TEXT};font-size:0.92rem;font-weight:600;white-space:nowrap;'
            f'overflow:hidden;text-overflow:ellipsis;">{matchup}</div>'
            f'<div style="color:{TEXT_MUTED};font-size:0.72rem;margin-top:3px;">'
            f'{side} &nbsp;&middot;&nbsp; {meta_line}</div>'
            f'</div>'

            f'<div style="flex:2;text-align:center;border-left:1px solid {BORDER};'
            f'border-right:1px solid {BORDER};padding:0 24px;">'
            f'<div style="color:{TEXT_MUTED};font-size:0.58rem;letter-spacing:0.12em;'
            f'text-transform:uppercase;margin-bottom:8px;">Book &nbsp;&rarr;&nbsp; Model</div>'
            f'<div style="display:flex;align-items:center;justify-content:center;gap:14px;">'
            f'<span style="color:{BLUE};font-size:1.05rem;font-weight:600;">{book_line}</span>'
            f'<span style="color:{TEXT_MUTED};font-size:0.9rem;">&rarr;</span>'
            f'<span style="color:{ACCENT};font-size:1.05rem;font-weight:600;">{model_proj}</span>'
            f'</div>'
            f'<div style="color:{TEXT_MUTED};font-size:0.68rem;margin-top:6px;">'
            f'{market} &nbsp;&middot;&nbsp; Edge: {margin}</div>'
            f'</div>'

            f'<div style="flex:1;text-align:right;padding-left:24px;min-width:100px;">'
            f'<div style="color:{TEXT_MUTED};font-size:0.58rem;letter-spacing:0.12em;text-transform:uppercase;">EV</div>'
            f'<div style="color:{ACCENT};font-size:1.15rem;font-weight:700;line-height:1.2;">{ev}</div>'
            f'<div style="color:{TEXT_MUTED};font-size:0.58rem;letter-spacing:0.12em;text-transform:uppercase;margin-top:8px;">Wager</div>'
            f'<div style="color:{TEXT};font-size:0.9rem;font-weight:600;">{wager}</div>'
            f'</div>'

            f'</div>'
        )
        cards.append(card)

    st.markdown("\n".join(cards), unsafe_allow_html=True)


# ─── Chart: Book vs Model ─────────────────────────────────────────────────────
def _chart_book_vs_model(sub: pd.DataFrame, market: str, key: str = "") -> None:
    if sub.empty:
        return

    def _to_num(s: str):
        try:
            return float(s.replace("+", "").replace(" runs", "").replace(" pts", "").strip())
        except Exception:
            return None

    labels     = sub.apply(lambda r: f"{r['matchup']} ({r['side']})", axis=1)
    book_vals  = sub["line"].apply(_to_num)
    model_vals = sub["proj"].apply(_to_num)

    if book_vals.isna().all() and model_vals.isna().all():
        st.caption("No numeric line data available.")
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Book Line",
        x=labels, y=book_vals,
        marker_color=BLUE,
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Book: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Model",
        x=labels, y=model_vals,
        marker_color=ACCENT,
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Model: %{y}<extra></extra>",
    ))
    _apply_chart_theme(
        fig,
        barmode="group",
        title=dict(text=f"{market} — BOOK vs MODEL", font=dict(size=11, family="'IBM Plex Mono', monospace"), x=0),
        xaxis_title="Game",
        yaxis_title="Line",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"bvm_{key}")


# ─── Chart: Edge & EV scatter ─────────────────────────────────────────────────
def _chart_edge_scatter(sub: pd.DataFrame, key: str = "") -> None:
    if sub.empty:
        return

    def _margin_num(s: str):
        try:
            return float(
                s.replace("+", "").replace("$", "")
                 .replace(" runs", "").replace(" pts", "")
                 .replace("%", "").strip()
            )
        except Exception:
            return None

    plot_df = sub.copy()
    plot_df["margin_num"] = plot_df["margin"].apply(_margin_num)
    plot_df = plot_df.dropna(subset=["margin_num", "ev_num"])

    if plot_df.empty:
        st.caption("No numeric edge / EV data to display.")
        return

    max_wager = plot_df["wager_num"].fillna(10).clip(lower=5).max()

    fig = go.Figure()
    for grade, grp in plot_df.groupby("grade"):
        sizes = (grp["wager_num"].fillna(10).clip(lower=5) / max_wager * 36 + 8).tolist()
        fig.add_trace(go.Scatter(
            x=grp["margin_num"], y=grp["ev_num"],
            mode="markers",
            name=GRADE_LABEL.get(grade, grade),
            marker=dict(
                size=sizes,
                color=GRADE_COLOUR.get(grade, TEXT_MUTED),
                line=dict(width=1, color=BG),
                opacity=0.88,
            ),
            text=grp["matchup"] + " — " + grp["side"],
            hovertemplate="<b>%{text}</b><br>Edge: %{x}<br>EV: %{y:.1f}%<extra></extra>",
        ))

    _apply_chart_theme(
        fig,
        title=dict(text="EDGE vs EV  (size = wager)", font=dict(size=11, family="'IBM Plex Mono', monospace"), x=0),
        xaxis_title="Edge",
        yaxis_title="EV %",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"scatter_{key}")

    with st.expander("How to read this chart"):
        st.markdown(f"""
<span style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;color:{TEXT_MUTED};line-height:1.9;">

**X-Axis — Edge**
The raw difference between the model's projection and the book's line.
For totals/spreads this is in runs or points (e.g. `-2.5 runs` means the model projects the total 2.5 runs lower than the book). For moneylines it is the win-probability gap after removing the book's vig.
A negative edge means the book line is set *above* our projection — we are betting the market is inflated.

**Y-Axis — EV %**
Expected Value as a percentage of the wager. A play at `+25%` EV means for every $100 wagered, the model expects a $25 profit long-run. Anything above `0%` is theoretically profitable; the model's grading thresholds are higher to account for model error and variance.

**Dot Size — Wager**
Larger dots = larger Kelly-sized wager. The model scales bet size to conviction: a big green dot in the top-right corner is the ideal play — high edge, high EV, high confidence.

**Dot Colour — Grade**<br>
<span style="color:{GRADE_COLOUR['[A]']}">●&nbsp;A</span> &nbsp;&nbsp; Highest conviction. Both edge and EV clear the model's strictest thresholds.<br>
<span style="color:{GRADE_COLOUR['[B+]']}">●&nbsp;B+</span> &nbsp; Strong play, narrowly below A-tier on one dimension.<br>
<span style="color:{GRADE_COLOUR['[B]']}">●&nbsp;B</span> &nbsp;&nbsp; Solid value, lower confidence or smaller edge.<br>
<span style="color:{GRADE_COLOUR['[C]']}">●&nbsp;C</span> &nbsp;&nbsp; Marginal edge — smallest wager, informational only.

**What to look for**
Plays in the upper-right quadrant (positive edge, high EV, large dot) are the strongest bets. Plays with negative edge but high EV are typically moneylines where the model disagrees with the book's implied probability after devigging.

</span>
""", unsafe_allow_html=True)


# ─── Metric row helper ────────────────────────────────────────────────────────
def _metric_row(data: list[tuple[str, str]]) -> None:
    cols = st.columns(len(data))
    for col, (label, value) in zip(cols, data):
        col.metric(label, value)


# ─── Bet-type panel ───────────────────────────────────────────────────────────
def _render_bet_tab(df: pd.DataFrame, league: str, market: str) -> None:
    """
    league : "MLB" | "NBA" | "ALL"
    market : "MONEYLINE" | "SPREAD" | "TOTAL" | "ALL"
    """
    today = _today_iso()

    league_mask = (df["league"] == league) if league != "ALL" else pd.Series(True, index=df.index)
    market_mask = (df["market"] == market) if market != "ALL" else pd.Series(True, index=df.index)

    today_ts   = pd.Timestamp(today)
    sub_today  = df[league_mask & market_mask & (df["date"] == today)].copy()
    sub_future = df[
        league_mask & market_mask &
        df["date_parsed"].notna() &
        (df["date_parsed"] > today_ts)
    ].copy()

    # ── Metrics (today only) ───────────────────────────────────────────────────
    total_plays = len(sub_today)
    a_tier      = len(sub_today[sub_today["grade"].isin(["[A]", "[A-dog]"])])
    avg_ev      = sub_today["ev_num"].mean() if not sub_today.empty else float("nan")
    total_wager = sub_today["wager_num"].sum() if not sub_today.empty else 0.0

    with st.container():
        _metric_row([
            ("Total Plays",   str(total_plays)),
            ("A-Tier Plays",  str(a_tier)),
            ("Avg EV",        f"{avg_ev:.1f}%" if pd.notna(avg_ev) else "—"),
            ("Total Wagered", f"${total_wager:.0f}" if total_wager else "—"),
        ])

    st.divider()

    # ── Today's plays ──────────────────────────────────────────────────────────
    if sub_today.empty:
        market_label = market.title() if market != "ALL" else "Any"
        sport_label  = league if league != "ALL" else "any sport"
        st.info(f"No {market_label} plays for {sport_label} today ({today}).")
    else:
        with st.container():
            st.subheader("Today's Plays")
            _plays_table(sub_today)

        chart_key = f"{league}_{market}"

        # Book vs Model chart only makes sense for a single market type
        if market != "ALL":
            with st.container():
                st.subheader("Book Line vs Model Projection")
                _chart_book_vs_model(sub_today, market, key=chart_key)

        with st.container():
            st.subheader("Edge & EV Distribution")
            _chart_edge_scatter(sub_today, key=chart_key)

    # ── Upcoming plays (future dates) ──────────────────────────────────────────
    if not sub_future.empty:
        GRADE_ORDER = {"[A]": 0, "[A-dog]": 1, "[B+]": 2, "[B]": 3, "[C]": 4}
        sub_future["_grade_rank"] = sub_future["grade"].map(lambda g: GRADE_ORDER.get(g, 99))

        st.divider()
        for date_str in sorted(sub_future["date"].unique()):
            day_sub   = (
                sub_future[sub_future["date"] == date_str]
                .sort_values("_grade_rank")
                .drop(columns=["_grade_rank"])
                .copy()
            )
            # Windows-safe: remove leading zero from day number
            day_label = pd.Timestamp(date_str).strftime("%A, %B %d").replace(" 0", " ").upper()
            st.subheader(f"UPCOMING — {day_label}")
            _plays_table(day_sub)


# ─── History tab ──────────────────────────────────────────────────────────────
def _render_history_tab(df: pd.DataFrame, league: str) -> None:
    """league : "MLB" | "NBA" | "ALL" """
    league_mask = (df["league"] == league) if league != "ALL" else pd.Series(True, index=df.index)
    sub = df[league_mask & df["hit"].notna()].sort_values("date_parsed").copy()

    if sub.empty:
        st.info("No settled bets on record. Enter results in the 'Bet Result' column of the Google Sheet.")
        return

    # ── Summary metrics ────────────────────────────────────────────────────────
    total_s  = len(sub)
    hits_s   = int(sub["hit"].sum())
    misses_s = total_s - hits_s
    overall  = hits_s / total_s * 100 if total_s else 0.0

    def _tier_rate(grades: list[str]) -> str:
        t = sub[sub["grade"].isin(grades)]
        if len(t) == 0:
            return "—"
        return f"{t['hit'].sum() / len(t) * 100:.1f}%"

    with st.container():
        _metric_row([
            ("Overall Hit Rate", f"{overall:.1f}%"),
            ("A-Tier Hit Rate",  _tier_rate(["[A]", "[A-dog]"])),
            ("B-Tier Hit Rate",  _tier_rate(["[B+]", "[B]"])),
            ("C-Tier Hit Rate",  _tier_rate(["[C]"])),
            ("Total Plays",      str(total_s)),
            ("Hits / Misses",    f"{hits_s} / {misses_s}"),
        ])

    st.divider()

    # ── Time filter ────────────────────────────────────────────────────────────
    today_dt = pd.Timestamp(dt.date.today())
    tf = st.radio(
        "Range", ["APR 16", "30D", "6M", "ALL"],
        index=3, horizontal=True,
        key=f"tf_{league}",
        label_visibility="collapsed",
    )
    cutoff = {
        "APR 16": pd.Timestamp("2026-04-16"),   # model breakthrough — no training-data leakage
        "30D":    today_dt - pd.Timedelta(days=30),
        "6M":     today_dt - pd.Timedelta(days=182),
        "ALL":    pd.Timestamp("2000-01-01"),
    }[tf]

    view = sub[sub["date_parsed"] >= cutoff].copy()

    # ── Chart 1: Rolling accuracy ──────────────────────────────────────────────
    with st.container():
        st.subheader("Accuracy Over Time")
        if len(view) >= 2:
            daily = (
                view.groupby("date_parsed")["hit"]
                .agg(["sum", "count"])
                .rename(columns={"sum": "hits", "count": "plays"})
                .reset_index()
            )
            daily["rate"] = (
                daily["hits"].rolling(7, min_periods=1).sum() /
                daily["plays"].rolling(7, min_periods=1).sum() * 100
            )
            daily["daily_rate"] = daily["hits"] / daily["plays"] * 100
            x_dates = daily["date_parsed"].tolist()
            y_rates = daily["rate"].tolist()

            fig = go.Figure()

            # Daily win rate — prominent solid line
            fig.add_trace(go.Scatter(
                x=daily["date_parsed"],
                y=daily["daily_rate"],
                mode="lines+markers",
                name="Daily",
                line=dict(color=ACCENT, width=2.5),
                marker=dict(size=6, color=ACCENT),
                customdata=daily[["hits", "plays"]].values,
                hovertemplate="%{x|%b %d}<br>Daily: %{y:.1f}%  (%{customdata[0]:.0f}/%{customdata[1]:.0f})<extra></extra>",
            ))

            # Green fill above break-even
            fig.add_trace(go.Scatter(
                x=x_dates + x_dates[::-1],
                y=[max(v, BREAK_EVEN) for v in y_rates] + [BREAK_EVEN] * len(x_dates),
                fill="toself", fillcolor="rgba(0,255,136,0.07)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False,
            ))
            # Red fill below break-even
            fig.add_trace(go.Scatter(
                x=x_dates + x_dates[::-1],
                y=[min(v, BREAK_EVEN) for v in y_rates] + [BREAK_EVEN] * len(x_dates),
                fill="toself", fillcolor="rgba(239,68,68,0.07)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False,
            ))
            # Break-even line
            fig.add_hline(
                y=BREAK_EVEN, line_dash="dash", line_color=TEXT_MUTED, line_width=1,
                annotation_text=f"Break-even {BREAK_EVEN}%",
                annotation_font=dict(color=TEXT_MUTED, size=10, family="'IBM Plex Mono', monospace"),
                annotation_position="bottom right",
            )
            # 7-day rolling — visible but secondary (blue, thinner, open markers)
            fig.add_trace(go.Scatter(
                x=daily["date_parsed"], y=daily["rate"],
                mode="lines+markers",
                name="7-day avg",
                line=dict(color=BLUE, width=1.5),
                marker=dict(size=4, color=BLUE, symbol="circle-open", line=dict(width=1.5, color=BLUE)),
                hovertemplate="%{x|%b %d}<br>7-day avg: %{y:.1f}%<extra></extra>",
            ))
            _apply_chart_theme(
                fig,
                title=dict(
                    text="ACCURACY OVER TIME"
                          "<br><sup style='color:#64748b;font-size:0.7em'>"
                          "<span style='color:#00ff88'>&#9644;</span> DAILY WIN RATE  &nbsp;·&nbsp;  <span style='color:#2563eb'>&#9644;</span> 7-DAY ROLLING AVG</sup>",
                    font=dict(family="'IBM Plex Mono', monospace", size=13, color=TEXT),
                    x=0.01, xanchor="left", y=0.97, yanchor="top",
                ),
                yaxis=dict(ticksuffix="%", range=[0, 100], gridcolor=BORDER_GRID),
                xaxis_title=None, yaxis_title="Win Rate",
                showlegend=False,
                margin=dict(t=70, b=32, l=16, r=16),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("At least 2 settled bets required to render this chart.")

    # ── Chart 2: Grade hit rate per day ────────────────────────────────────────
    with st.container():
        st.subheader("Grade Hit Rate Per Day")
        if not view.empty:
            grp = (
                view.groupby(["date_parsed", "grade"])["hit"]
                .agg(["sum", "count"])
                .reset_index()
            )
            grp["rate"] = grp["sum"] / grp["count"] * 100

            fig2 = go.Figure()
            for grade in ["[A]", "[A-dog]", "[B+]", "[B]", "[C]"]:
                g = grp[grp["grade"] == grade]
                if g.empty:
                    continue
                fig2.add_trace(go.Bar(
                    x=g["date_parsed"], y=g["rate"],
                    name=GRADE_LABEL.get(grade, grade),
                    marker_color=GRADE_COLOUR.get(grade, TEXT_MUTED),
                    opacity=0.85,
                    hovertemplate="%{x|%b %d}<br>%{y:.1f}%<extra></extra>",
                ))
            fig2.add_hline(y=BREAK_EVEN, line_dash="dash", line_color=TEXT_MUTED, line_width=1)
            _apply_chart_theme(
                fig2,
                barmode="group",
                yaxis=dict(ticksuffix="%", range=[0, 110], gridcolor=BORDER_GRID),
                xaxis_title=None, yaxis_title="Hit %",
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3: Bet type breakdown ────────────────────────────────────────────
    with st.container():
        st.subheader("Win Rate by Market Type")
        if not view.empty:
            mkt = (
                view.groupby("market")["hit"]
                .agg(["sum", "count"])
                .reset_index()
            )
            mkt["rate"] = mkt["sum"] / mkt["count"] * 100
            mkt = mkt.sort_values("rate", ascending=True)

            bar_colors = [GREEN if r >= BREAK_EVEN else RED for r in mkt["rate"]]
            fig3 = go.Figure(go.Bar(
                x=mkt["rate"], y=mkt["market"],
                orientation="h",
                marker_color=bar_colors,
                opacity=0.85,
                text=mkt.apply(
                    lambda r: f"{r['rate']:.1f}%  ({int(r['sum'])}/{int(r['count'])})", axis=1
                ),
                textposition="outside",
                textfont=dict(family="'IBM Plex Mono', monospace", size=11, color=TEXT),
                hovertemplate="%{y}<br>Win rate: %{x:.1f}%<extra></extra>",
            ))
            fig3.add_vline(x=BREAK_EVEN, line_dash="dash", line_color=TEXT_MUTED, line_width=1)
            _apply_chart_theme(
                fig3,
                xaxis=dict(ticksuffix="%", range=[0, 115], gridcolor=BORDER_GRID),
                yaxis_title=None, xaxis_title="Win Rate",
                margin=dict(t=20, b=32, l=16, r=80),
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ── Settled bets cards ─────────────────────────────────────────────────────
    with st.container():
        st.subheader("Settled Bets")
        import html as _html

        settled_sorted = view.sort_values("date_parsed", ascending=False)
        cards: list[str] = []

        for _, row in settled_sorted.iterrows():
            import html as _html
            grade        = row.get("grade", "")
            result_raw   = (row.get("result", "") or "").strip().upper()
            matchup      = _html.escape(str(row.get("matchup", "")))
            side         = _html.escape(str(row.get("side", "")))
            market       = _html.escape(str(row.get("market", "")))
            book_line    = _html.escape(str(row.get("line", "—")))
            model_proj   = _html.escape(str(row.get("proj", "—")))
            margin       = _html.escape(str(row.get("margin", "—")))
            ev           = _html.escape(str(row.get("ev", "—")))
            wager        = _html.escape(str(row.get("wager", "—")))
            date_str     = _html.escape(str(row.get("date", "")))

            grade_colour  = GRADE_COLOUR.get(grade, TEXT_MUTED)
            grade_label   = GRADE_LABEL.get(grade, grade)
            result_colour = GREEN if result_raw == "HIT" else (RED if result_raw == "MISS" else TEXT_MUTED)
            result_label  = result_raw if result_raw in ("HIT", "MISS") else "PENDING"
            left_border   = result_colour  # border signals outcome, not grade

            card = (
                f'<div style="background:#111827;border:1px solid {BORDER};border-left:4px solid {left_border};'
                f'display:flex;align-items:center;padding:14px 22px;margin-bottom:6px;'
                f'font-family:\'IBM Plex Mono\',\'Roboto Mono\',monospace;">'

                f'<div style="flex:2.2;min-width:0;padding-right:20px;">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                f'<span style="color:{result_colour};font-size:0.6rem;font-weight:700;'
                f'letter-spacing:0.14em;text-transform:uppercase;border:1px solid {result_colour}55;'
                f'padding:1px 6px;">{result_label}</span>'
                f'<span style="color:{grade_colour};font-size:0.6rem;font-weight:600;'
                f'letter-spacing:0.1em;text-transform:uppercase;">{grade_label}</span>'
                f'</div>'
                f'<div style="color:{TEXT};font-size:0.9rem;font-weight:600;white-space:nowrap;'
                f'overflow:hidden;text-overflow:ellipsis;">{matchup}</div>'
                f'<div style="color:{TEXT_MUTED};font-size:0.7rem;margin-top:3px;">'
                f'{side} &nbsp;&middot;&nbsp; {market} &nbsp;&middot;&nbsp; {date_str}</div>'
                f'</div>'

                f'<div style="flex:2;text-align:center;border-left:1px solid {BORDER};'
                f'border-right:1px solid {BORDER};padding:0 22px;">'
                f'<div style="color:{TEXT_MUTED};font-size:0.58rem;letter-spacing:0.12em;'
                f'text-transform:uppercase;margin-bottom:7px;">Book &nbsp;&rarr;&nbsp; Model</div>'
                f'<div style="display:flex;align-items:center;justify-content:center;gap:12px;">'
                f'<span style="color:{BLUE};font-size:1rem;font-weight:600;">{book_line}</span>'
                f'<span style="color:{TEXT_MUTED};">&rarr;</span>'
                f'<span style="color:{ACCENT};font-size:1rem;font-weight:600;">{model_proj}</span>'
                f'</div>'
                f'<div style="color:{TEXT_MUTED};font-size:0.68rem;margin-top:5px;">Edge: {margin}</div>'
                f'</div>'

                f'<div style="flex:1;text-align:right;padding-left:22px;min-width:90px;">'
                f'<div style="color:{TEXT_MUTED};font-size:0.58rem;letter-spacing:0.12em;text-transform:uppercase;">EV</div>'
                f'<div style="color:{ACCENT};font-size:1.1rem;font-weight:700;">{ev}</div>'
                f'<div style="color:{TEXT_MUTED};font-size:0.58rem;letter-spacing:0.12em;text-transform:uppercase;margin-top:7px;">Wager</div>'
                f'<div style="color:{TEXT};font-size:0.88rem;font-weight:600;">{wager}</div>'
                f'</div>'

                f'</div>'
            )
            cards.append(card)

        if cards:
            st.markdown("\n".join(cards), unsafe_allow_html=True)
        else:
            st.info("No settled bets in the selected range.")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="INSIDELINE | TERMINAL",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

    # ── Header bar ─────────────────────────────────────────────────────────────
    today_label = dt.date.today().strftime("%Y-%m-%d")
    st.markdown(f"""
    <div style="
        display: flex; align-items: baseline; gap: 16px;
        border-bottom: 1px solid {BORDER};
        padding-bottom: 14px; margin-bottom: 4px;
    ">
      <span style="
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.05rem; font-weight: 600;
        color: {ACCENT}; letter-spacing: 0.06em;
      ">INSIDELINE</span>
      <span style="
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.05rem; font-weight: 300;
        color: {TEXT_MUTED}; letter-spacing: 0.06em;
      ">| TERMINAL v2.0</span>
      <span style="
        margin-left: auto;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem; color: {TEXT_MUTED};
        letter-spacing: 0.08em;
      ">{today_label}</span>
    </div>
    """, unsafe_allow_html=True)

    if not CREDS_PATH or not SHEET_ID:
        st.error(
            "GOOGLE_SHEETS_CREDS_PATH or GOOGLE_SHEETS_SPREADSHEET_ID not set in .env"
        )
        return

    with st.spinner("Fetching sheet data…"):
        df = load_sheet_data()

    if df.empty:
        st.warning("No data returned. Verify credentials and sheet content.")
        return

    # ── Top-level sport tabs ───────────────────────────────────────────────────
    tab_all, tab_mlb, tab_nba, tab_hist = st.tabs(["ALL", "MLB", "NBA", "History"])

    # ── ALL / MLB / NBA tabs: market dropdown + content ────────────────────────
    for sport_tab, league in [(tab_all, "ALL"), (tab_mlb, "MLB"), (tab_nba, "NBA")]:
        with sport_tab:
            dd_col, _ = st.columns([1, 3])
            with dd_col:
                market_choice = st.selectbox(
                    "Market",
                    ["All", "Moneyline", "Spread", "Total"],
                    key=f"mkt_{league}",
                )
            market_arg = market_choice.upper() if market_choice != "All" else "ALL"
            _render_bet_tab(df, league, market_arg)

    # ── History tab: sport dropdown + history content ──────────────────────────
    with tab_hist:
        dd_col, _ = st.columns([1, 3])
        with dd_col:
            sport_choice = st.selectbox(
                "Sport",
                ["All", "MLB", "NBA"],
                key="hist_sport",
            )
        league_arg = sport_choice.upper() if sport_choice != "All" else "ALL"
        _render_history_tab(df, league_arg)


if __name__ == "__main__":
    main()
