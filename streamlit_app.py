#!/usr/bin/env python3
"""
Bitcoin DCA Dashboard - 6-Agent Ensemble
Template-compliant Streamlit interface
Uses fear_greed_value_feargreed_imputed column
"""

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from capstone import (
    load_data,
    compute_weights,
    compute_allocation_multipliers,
    compute_weights_hp,
    compute_weights_rap,
    compute_weights_emp,
    compute_weights_mvrv,
    compute_weights_fear_greed,
    compute_weights_spd_momentum,
    BACKTEST_START,
    BACKTEST_END,
    PRICE_COL,
    INVESTMENT_WINDOW,
    AGENT_WEIGHTS,
)

st.set_page_config(
    page_title="Bitcoin DCA – 6-Agent Ensemble",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=True)
def _load_price_and_weights():
    """Load price data and compute weights."""
    price_df = load_data()

    if BACKTEST_START is not None:
        price_df = price_df.loc[pd.to_datetime(BACKTEST_START):]
    if BACKTEST_END is not None:
        price_df = price_df.loc[:pd.to_datetime(BACKTEST_END)]

    price_df = price_df.sort_index()

    # Get RAW allocation multipliers (0.7-1.5 range)
    alloc_base = compute_allocation_multipliers(price_df, AGENT_WEIGHTS)
    alloc_base = alloc_base.reindex(price_df.index).ffill().fillna(1.0)
    
    # Get normalized weights for reference
    weights_series = compute_weights(price_df)
    weights_series = weights_series.reindex(price_df.index).ffill().fillna(0.0)

    return price_df, weights_series, alloc_base


def _apply_aggressiveness(alloc_base: pd.Series, aggressiveness: float, cap: float) -> pd.Series:
    """Scale allocation multipliers by aggressiveness factor."""
    scaled = 1.0 + aggressiveness * (alloc_base - 1.0)
    return scaled.clip(lower=0.0, upper=cap)


def _nearest_index(ts_index: pd.DatetimeIndex, target_date: dt.date) -> pd.Timestamp:
    """Find the nearest date in the index at or before target_date."""
    target_ts = pd.Timestamp(target_date)
    if target_ts in ts_index:
        return target_ts
    prev_dates = ts_index[ts_index <= target_ts]
    if len(prev_dates) == 0:
        return ts_index[0]
    return prev_dates[-1]


def _compute_backtest(price_df: pd.DataFrame, alloc_mult: pd.Series, base_daily=1.0):
    """Compute backtest results comparing strategy vs uniform DCA."""
    price = price_df[PRICE_COL].astype(float)

    m = alloc_mult.reindex(price.index).ffill().fillna(1.0).shift(1).fillna(1.0)
    m = m.clip(lower=0.0)

    contrib_bench = pd.Series(base_daily, index=price.index)
    contrib_strat = base_daily * m

    btc_bench = (contrib_bench / price).cumsum()
    btc_strat = (contrib_strat / price).cumsum()

    equity_bench = btc_bench * price
    equity_strat = btc_strat * price

    bench_curve = equity_bench / equity_bench.iloc[0]
    strat_curve = equity_strat / equity_strat.iloc[0]

    cum_contrib_bench = contrib_bench.cumsum()
    cum_contrib_strat = contrib_strat.cumsum()

    eff_bench = equity_bench / cum_contrib_bench
    eff_strat = equity_strat / cum_contrib_strat

    eff_bench_ret = eff_bench.pct_change()
    eff_strat_ret = eff_strat.pct_change()

    spd_bench = (btc_bench * 1e8) / cum_contrib_bench
    spd_strat = (btc_strat * 1e8) / cum_contrib_strat

    spd_bench_ret = spd_bench.pct_change()
    spd_strat_ret = spd_strat.pct_change()

    for s in (eff_bench_ret, eff_strat_ret, spd_bench_ret, spd_strat_ret):
        if len(s) > 0:
            s.iloc[0] = np.nan

    out = pd.DataFrame({
        "equity_benchmark": equity_bench,
        "equity_strategy": equity_strat,
        "benchmark_curve": bench_curve,
        "strategy_curve": strat_curve,
        "benchmark_ret": bench_curve.pct_change().fillna(0.0),
        "strategy_ret": strat_curve.pct_change().fillna(0.0),
        "alloc": m,
        "cum_contrib_benchmark": cum_contrib_bench,
        "cum_contrib_strategy": cum_contrib_strat,
        "btc_holdings_benchmark": btc_bench,
        "btc_holdings_strategy": btc_strat,
        "spd_benchmark": spd_bench,
        "spd_strategy": spd_strat,
        "spd_ret_benchmark": spd_bench_ret,
        "spd_ret_strategy": spd_strat_ret,
        "eff_benchmark": eff_bench,
        "eff_strategy": eff_strat,
        "eff_ret_benchmark": eff_bench_ret,
        "eff_ret_strategy": eff_strat_ret,
    }, index=price.index)

    return out


def _performance_stats_dca(equity: pd.Series, cum_contrib: pd.Series, daily_ret: pd.Series):
    """Calculate performance statistics for DCA strategy."""
    equity = equity.dropna()
    cum_contrib = cum_contrib.reindex(equity.index).ffill()
    daily_ret = daily_ret.reindex(equity.index).fillna(0.0)

    if len(equity) <= 2:
        return {"final_value": np.nan, "contributed": np.nan, "roi": np.nan, "vol": np.nan, "max_dd": np.nan}

    final_value = float(equity.iloc[-1])
    contributed = float(cum_contrib.iloc[-1])
    roi = (final_value / contributed - 1.0) if contributed > 0 else np.nan

    vol = float(daily_ret.std() * np.sqrt(252.0))

    running_max = equity.cummax()
    drawdowns = (equity / running_max) - 1.0
    max_dd = float(drawdowns.min())

    return {
        "final_value": final_value,
        "contributed": contributed,
        "roi": roi,
        "vol": vol,
        "max_dd": max_dd,
    }


def compute_agent_components(price_df: pd.DataFrame, as_of_date) -> dict:
    """Return 6 agent raw contributions at a given date."""
    price_df = price_df.sort_index()
    as_of_ts = pd.to_datetime(as_of_date)
    
    sub = price_df.loc[:as_of_ts].copy()
    if sub.empty:
        return {"RAP": 1.0, "SPD": 1.0, "F&G": 1.0, "MVRV": 1.0, "HP": 1.0, "EMP": 1.0}
    
    agents = {
        'RAP': compute_weights_rap,
        'SPD': compute_weights_spd_momentum,
        'F&G': compute_weights_fear_greed,
        'MVRV': compute_weights_mvrv,
        'HP': compute_weights_hp,
        'EMP': compute_weights_emp,
    }
    
    contributions = {}
    for name, func in agents.items():
        try:
            weight_series = func(sub)
            if len(weight_series) > 0:
                contributions[name] = float(weight_series.iloc[-1])
            else:
                contributions[name] = 1.0
        except Exception:
            contributions[name] = 1.0
    
    return contributions


SENTIMENT_COLORS = {
    "People (BitcoinTalk)": "#1f77b4",
    "World (News)": "#ff7f0e",
    "Market (F&G scaled)": "#2ca02c",
    "BTC price": "#c7c7c7",
}


@st.cache_data(show_spinner=False)
def load_sentiment_cached(freq):
    """Load sentiment data with caching."""
    return _load_sentiment_layers(freq=freq)


@st.cache_data(show_spinner=True)
def _load_sentiment_layers(freq: str = "W"):
    """
    Load 3 sentiment sources and BTC price, align on common time index.
    Uses fear_greed_value_feargreed_imputed column.
    """
    def _read_sent_file(path: str, value_col: str) -> pd.DataFrame:
        df = pd.read_csv(path)

        if "post_date" in df.columns:
            dcol = "post_date"
        elif "date" in df.columns:
            dcol = "date"
        else:
            raise ValueError(f"No 'post_date' or 'date' column in {path}")

        df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.normalize()
        df = df.dropna(subset=[dcol]).set_index(dcol).sort_index()

        out = df[[value_col]].copy()
        out = out.groupby(out.index).mean()
        return out

    base = os.path.dirname(__file__)

    people_path = os.path.join(base, "btc_talk_ensemble_data.csv")
    world_path  = os.path.join(base, "btc_news_sentiment_final.csv")
    fg_path     = os.path.join(base, "f&g_imputed_2.csv")

    people = _read_sent_file(people_path, "sent_ensemble").rename(columns={"sent_ensemble": "people_sent"})
    world  = _read_sent_file(world_path, "ensemble").rename(columns={"ensemble": "world_sent"})
    fg_raw = pd.read_csv(fg_path)

    if "time" in fg_raw.columns:
        dcol = "time"
    elif "date" in fg_raw.columns:
        dcol = "date"
    elif "post_date" in fg_raw.columns:
        dcol = "post_date"
    else:
        raise ValueError(f"No date column in {fg_path}")

    fg_raw[dcol] = pd.to_datetime(fg_raw[dcol], errors="coerce").dt.normalize()
    fg_raw = fg_raw.dropna(subset=[dcol]).set_index(dcol).sort_index()

    # Use fear_greed_value_feargreed_imputed column
    if "fear_greed_value_feargreed_imputed" not in fg_raw.columns:
        raise ValueError("fear_greed_value_feargreed_imputed column not found")

    fg = fg_raw[["fear_greed_value_feargreed_imputed"]].rename(columns={"fear_greed_value_feargreed_imputed": "fg"}).groupby(lambda x: x).mean()

    price_col_candidates = ["PriceUSD_coinmetrics", "btc_price", "price", "PriceUSD"]
    price_col = next((c for c in price_col_candidates if c in fg_raw.columns), None)

    if price_col is not None:
        btc = fg_raw[[price_col]].rename(columns={price_col: "btc_price"}).groupby(lambda x: x).mean()
    else:
        btc = load_data()[[PRICE_COL]].rename(columns={PRICE_COL: "btc_price"})

    people = people.resample(freq).mean()
    world  = world.resample(freq).mean()
    fg     = fg.resample(freq).mean()
    btc    = btc.resample(freq).last()

    # Convert to timezone-naive
    if people.index.tz is not None:
        people.index = people.index.tz_localize(None)
    if world.index.tz is not None:
        world.index = world.index.tz_localize(None)
    if fg.index.tz is not None:
        fg.index = fg.index.tz_localize(None)
    if btc.index.tz is not None:
        btc.index = btc.index.tz_localize(None)

    df = pd.concat([people, world, fg, btc], axis=1)
    df["market_sent_scaled"] = (df["fg"] - 50) / 50

    return df


def _compute_price_metrics(df_sent: pd.DataFrame, vol_window: int = 26, ma_window: int = 52) -> pd.DataFrame:
    """Compute market metrics from BTC price."""
    out = df_sent.copy()
    px = out["btc_price"].astype(float)
    out["ret"] = px.pct_change()
    out["realized_vol"] = out["ret"].rolling(vol_window).std()

    peak = px.cummax()
    out["drawdown"] = (px / peak) - 1.0

    ma = px.rolling(ma_window).mean()
    out["trend_dist"] = (px / ma) - 1.0

    return out


def _plot_quadrant_matplotlib(x: pd.Series, y: pd.Series, x0: float, y0: float, title: str, xlabel: str, ylabel: str):
    """Create quadrant plot with matplotlib."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.scatter(x, y, s=14, alpha=0.65)
    ax.axvline(x0, linewidth=1.2, alpha=0.6)
    ax.axhline(y0, linewidth=1.2, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def _corr_heatmap_matplotlib(corr: pd.DataFrame, title: str = "Correlation heatmap"):
    """Create correlation heatmap with matplotlib."""
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(corr.values, aspect="auto", cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(corr.shape[1]))
    ax.set_yticks(range(corr.shape[0]))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", size=8)
    ax.set_yticklabels(corr.index, size=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

try:
    price_df, weights_series, alloc_base = _load_price_and_weights()
    st.sidebar.markdown("### Strategy Controls")
    aggr = st.sidebar.slider("Aggressiveness", min_value=0.0, max_value=5.0, value=1.0, step=0.05)
    alloc_cap = st.sidebar.slider("Max allocation cap (×)", min_value=1.0, max_value=5.0, value=5.0, step=0.5)
    alloc_series = _apply_aggressiveness(alloc_base, aggr, alloc_cap)
    bt_df = _compute_backtest(price_df, alloc_series, base_daily=1.0)
    data_loaded_ok = True
except Exception as e:
    data_loaded_ok = False
    load_error = e


st.sidebar.header("Controls")

if data_loaded_ok:
    all_dates = price_df.index.date
    min_date = all_dates.min()
    max_date = all_dates.max()
    default_today = max_date
    
    selected_date = st.sidebar.date_input(
        "As-of date for recommendation",
        value=default_today,
        min_value=min_date,
        max_value=max_date,
    )
    
    k_features = st.sidebar.slider(
        "Top-N features to display (Agent transparency)",
        min_value=3,
        max_value=50,
        value=30,
        step=1,
    )
else:
    selected_date = None
    k_features = 30

st.sidebar.markdown("---")
st.sidebar.caption("6-Agent ensemble Bitcoin DCA strategy (Template Compliant)")


st.title("Bitcoin DCA Agent – 6-Agent Ensemble")

if not data_loaded_ok:
    st.error(
        "Could not load data from capstone.py.\n\n"
        f"Error: {repr(load_error)}"
    )
    st.stop()

tab_today, tab_transparency, tab_history, tab_sentiment = st.tabs(
    ["Today's recommendation", "Agent transparency", "Historical effectiveness", "Sentiment overview"]
)

# ============================================================================
# TAB 1 – TODAY'S RECOMMENDATION
# ============================================================================
with tab_today:
    st.subheader("Today's DCA recommendation")
    
    idx = _nearest_index(price_df.index, selected_date)
    date_str = idx.strftime("%Y-%m-%d")
    
    today_alloc = float(alloc_series.loc[idx])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("As-of date", date_str)
    
    with col2:
        st.metric("Recommended BTC allocation multiplier", f"{today_alloc:.2f}×")
    
    with col3:
        today_price = float(price_df.loc[idx, PRICE_COL])
        st.metric("BTC price", f"${today_price:,.0f}")

    st.markdown("---")
    
    st.write("**Allocation gauge (vs regular DCA)**")
    gauge_cols = st.columns(2)
    
    gauge_min = alloc_series.quantile(0.01)
    gauge_max = alloc_series.quantile(0.99)
    gauge_val = (today_alloc - gauge_min) / (gauge_max - gauge_min)
    gauge_val = float(np.clip(gauge_val, 0.0, 1.0))
    
    with gauge_cols[0]:
        st.progress(gauge_val)
    
    with gauge_cols[1]:
        median_alloc = alloc_series.median()
        if today_alloc < median_alloc * 0.5:
            msg = "🔵 Very cautious – significantly underweight today's DCA"
        elif today_alloc < median_alloc * 0.8:
            msg = "🔵 Cautious – underweight today's DCA"
        elif today_alloc < median_alloc * 1.2:
            msg = "⚪ Neutral – around typical allocation"
        elif today_alloc < median_alloc * 1.5:
            msg = "🟢 Confident – overweight today's DCA"
        else:
            msg = "🟢 High conviction – significantly overweight today's DCA"
        
        st.info(msg)
    
    st.caption(
        f"Median allocation: {median_alloc:.2f}×. "
        "Gauge shows today's allocation relative to historical distribution."
    )


# ============================================================================
# TAB 2 – AGENT TRANSPARENCY
# ============================================================================
with tab_transparency:
    st.subheader("What is the agent seeing today?")
    
    idx = _nearest_index(price_df.index, selected_date)
    today_alloc = float(alloc_series.loc[idx])
    
    st.markdown(
        f"**As-of date:** `{idx.date()}` | "
        f"**Allocation multiplier:** `{today_alloc:.2f}×`"
    )
    
    st.markdown("### Agent contribution breakdown")
    
    contrib_raw = compute_agent_components(price_df, idx)
    contrib = pd.Series(contrib_raw, name="raw")
    
    if contrib.sum() != 0:
        contrib_scaled = contrib / contrib.sum() * today_alloc
    else:
        contrib_scaled = contrib
    
    contrib_df = contrib_scaled.to_frame("Allocation")
    
    st.bar_chart(contrib_df)
    st.caption(
        "Shows how each agent contributes to today's allocation multiplier. "
        "Agents: RAP (regime), SPD (momentum), F&G (sentiment), MVRV (value), HP (volatility), EMP (protection)."
    )
    
    st.markdown("---")
    st.markdown("### Top features for today")
    st.caption("All available features ranked by |z-score|. Higher |z-score| = more unusual regime.")

    sub = price_df.loc[:idx].copy()

    past_price = sub[PRICE_COL].astype(float).shift(1)

    # RAP agent features
    gap_ma_90 = (past_price / past_price.rolling(90, min_periods=1).mean() - 1) * 100
    gap_ma_30 = (past_price / past_price.rolling(30, min_periods=1).mean() - 1) * 100
    gap_ma_200 = (past_price / past_price.rolling(200, min_periods=1).mean() - 1) * 100
    drawdown = (past_price / past_price.rolling(365, min_periods=30).max() - 1) * 100
    drawdown_90d = (past_price / past_price.rolling(90, min_periods=30).max() - 1) * 100
    delta = past_price.diff()
    rsi = 100 - (100 / (1 + delta.where(delta > 0, 0).rolling(14).mean() / -delta.where(delta < 0, 0).rolling(14).mean()))

    # HP agent features
    ret_log = np.log(past_price).diff()
    vol30 = ret_log.rolling(30, min_periods=10).std(ddof=0)
    vol90 = ret_log.rolling(90, min_periods=30).std(ddof=0)
    mu_vol = vol30.rolling(365, min_periods=60).mean().shift(1)
    sd_vol = vol30.rolling(365, min_periods=60).std(ddof=0).shift(1)
    zvol = (vol30 - mu_vol) / (sd_vol.replace(0.0, np.nan) + 1e-12)

    # SPD agent features
    spd = 1 / past_price * 1e8
    spd_roc_7 = spd.pct_change(7, fill_method=None) * 100
    spd_roc_30 = spd.pct_change(30, fill_method=None) * 100
    spd_roc_90 = spd.pct_change(90, fill_method=None) * 100
    spd_accel = (spd.rolling(7, min_periods=3).mean() - spd.rolling(30, min_periods=10).mean()) / (spd.rolling(30, min_periods=10).mean() + 1e-6) * 100

    # MVRV agent features
    mvrv_ratio = past_price / past_price.rolling(365, min_periods=100).mean()
    mvrv_ratio_180 = past_price / past_price.rolling(180, min_periods=60).mean()
    mvrv_z = (mvrv_ratio - mvrv_ratio.rolling(365, min_periods=100).mean()) / (mvrv_ratio.rolling(365, min_periods=100).std().shift(1) + 1e-6)

    # Fear & Greed
    fg = sub["fear_greed_value_feargreed_imputed"]

    # Price momentum features
    ret_7d = past_price.pct_change(7) * 100
    ret_30d = past_price.pct_change(30) * 100
    ret_90d = past_price.pct_change(90) * 100
    ret_365d = past_price.pct_change(365) * 100

    # Trend features
    ma_50 = past_price.rolling(50, min_periods=20).mean()
    ma_200 = past_price.rolling(200, min_periods=100).mean()
    price_vs_ma50 = (past_price / ma_50 - 1) * 100
    price_vs_ma200 = (past_price / ma_200 - 1) * 100

    # Volatility features
    vol_7d = ret_log.rolling(7, min_periods=3).std() * np.sqrt(365) * 100
    vol_180d = ret_log.rolling(180, min_periods=60).std() * np.sqrt(365) * 100
    vol_ratio = vol30 / (vol90 + 1e-9)

    # Additional on-chain features (if available)
    feature_snapshot = pd.Series({
        "fear_greed": float(fg.iloc[-1]),
        "rap_gap_ma_90": float(gap_ma_90.iloc[-1]),
        "rap_gap_ma_30": float(gap_ma_30.iloc[-1]),
        "rap_gap_ma_200": float(gap_ma_200.iloc[-1]),
        "rap_drawdown_365": float(drawdown.iloc[-1]),
        "rap_drawdown_90": float(drawdown_90d.iloc[-1]),
        "rap_rsi_14": float(rsi.iloc[-1]),
        "hp_zvol": float(zvol.iloc[-1]),
        "hp_vol_30d": float(vol30.iloc[-1] * np.sqrt(365) * 100),
        "hp_vol_90d": float(vol90.iloc[-1] * np.sqrt(365) * 100),
        "spd_roc_7": float(spd_roc_7.iloc[-1]),
        "spd_roc_30": float(spd_roc_30.iloc[-1]),
        "spd_roc_90": float(spd_roc_90.iloc[-1]),
        "spd_accel": float(spd_accel.iloc[-1]),
        "mvrv_ratio": float(mvrv_ratio.iloc[-1]),
        "mvrv_ratio_180": float(mvrv_ratio_180.iloc[-1]),
        "mvrv_z": float(mvrv_z.iloc[-1]),
        "ret_7d": float(ret_7d.iloc[-1]),
        "ret_30d": float(ret_30d.iloc[-1]),
        "ret_90d": float(ret_90d.iloc[-1]),
        "ret_365d": float(ret_365d.iloc[-1]),
        "price_vs_ma50": float(price_vs_ma50.iloc[-1]),
        "price_vs_ma200": float(price_vs_ma200.iloc[-1]),
        "vol_7d": float(vol_7d.iloc[-1]),
        "vol_180d": float(vol_180d.iloc[-1]),
        "vol_ratio_30_90": float(vol_ratio.iloc[-1]),
    }).replace([np.inf, -np.inf], np.nan).dropna()

    # Add on-chain features if available
    onchain_features = [
        'CapMVRVCur_coinmetrics',
        'NVTAdj_coinmetrics',
        'SplyAct1yr_coinmetrics',
        'VelCur1yr_coinmetrics',
        'AdrActCnt_coinmetrics',
        'TxCnt_coinmetrics',
        'FlowInExNtv_coinmetrics',
        'FlowOutExNtv_coinmetrics',
    ]
    
    for feat in onchain_features:
        if feat in sub.columns:
            try:
                val = float(sub[feat].iloc[-1])
                if not np.isnan(val) and not np.isinf(val):
                    feature_snapshot[feat] = val
            except:
                pass

    # Compute z-scores
    if feature_snapshot.std() == 0:
        z = feature_snapshot - feature_snapshot.mean()
    else:
        z = (feature_snapshot - feature_snapshot.mean()) / (feature_snapshot.std() + 1e-9)

    top_feats = (
        pd.DataFrame({
            "feature": feature_snapshot.index,
            "value": feature_snapshot.values,
            "z_score": z.values,
            "abs_z": np.abs(z.values),
        })
        .sort_values("abs_z", ascending=False)
        .head(k_features)
    )

    st.dataframe(
        top_feats[["feature", "value", "z_score"]]
        .set_index("feature")
        .style.format({"value": "{:.2f}", "z_score": "{:.2f}"}),
        use_container_width=True,
    )


# ============================================================================
# TAB 3 – HISTORICAL EFFECTIVENESS
# ============================================================================
with tab_history:
    st.subheader("Agent vs regular DCA – historical effectiveness")

    hist_min_date = price_df.index.date.min()
    hist_max_date = price_df.index.date.max()

    hist_range = st.slider(
        "Backtest window",
        min_value=hist_min_date,
        max_value=hist_max_date,
        value=(hist_min_date, hist_max_date),
    )

    start_ts = pd.Timestamp(hist_range[0])
    end_ts   = pd.Timestamp(hist_range[1])

    price_win = price_df.loc[start_ts:end_ts].copy()
    alloc_win = alloc_series.loc[start_ts:end_ts].copy()

    backtest_df = _compute_backtest(price_win, alloc_win, base_daily=1.0)

    strat_stats = _performance_stats_dca(
        backtest_df["equity_strategy"],
        backtest_df["cum_contrib_strategy"],
        backtest_df["strategy_ret"],
    )

    bench_stats = _performance_stats_dca(
        backtest_df["equity_benchmark"],
        backtest_df["cum_contrib_benchmark"],
        backtest_df["benchmark_ret"],
    )

    valid_mask = backtest_df[["spd_ret_strategy", "spd_ret_benchmark"]].notna().all(axis=1)
    comp = backtest_df.loc[valid_mask]
    win_rate_daily = (comp["spd_ret_strategy"] > comp["spd_ret_benchmark"]).mean() if len(comp) else np.nan

    spd_strat = float(backtest_df["spd_strategy"].iloc[-1])
    spd_bench = float(backtest_df["spd_benchmark"].iloc[-1])

    excess_spd = (spd_strat / spd_bench - 1.0) if spd_bench > 0 else np.nan

    spd_vol_strat = float(comp["spd_ret_strategy"].std() * np.sqrt(365)) if len(comp) else np.nan
    spd_vol_bench = float(comp["spd_ret_benchmark"].std() * np.sqrt(365)) if len(comp) else np.nan

    spd_dd_strat = float((backtest_df["spd_strategy"] / backtest_df["spd_strategy"].cummax() - 1.0).min())
    spd_dd_bench = float((backtest_df["spd_benchmark"] / backtest_df["spd_benchmark"].cummax() - 1.0).min())

    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("##### 6-Agent Strategy")
        st.metric("Total contributed", f"${strat_stats['contributed']:,.0f}")
        st.metric("Final value", f"${strat_stats['final_value']:,.0f}")
        st.metric("Final SPD (sats/$)", f"{spd_strat:,.1f}")
        st.metric("SPD volatility", f"{spd_vol_strat*100:.2f}%" if np.isfinite(spd_vol_strat) else "n/a")
        st.metric("SPD max drawdown", f"{spd_dd_strat*100:.2f}%")

    with col_b:
        st.markdown("##### Regular DCA")
        st.metric("Total contributed", f"${bench_stats['contributed']:,.0f}")
        st.metric("Final value", f"${bench_stats['final_value']:,.0f}")
        st.metric("Final SPD (sats/$)", f"{spd_bench:,.1f}")
        st.metric("SPD volatility", f"{spd_vol_bench*100:.2f}%" if np.isfinite(spd_vol_bench) else "n/a")
        st.metric("SPD max drawdown", f"{spd_dd_bench*100:.2f}%")

    with col_c:
        st.markdown("##### Difference")
        st.metric("Daily SPD win rate", f"{win_rate_daily*100:.1f}%" if np.isfinite(win_rate_daily) else "n/a")
        st.metric("Excess SPD vs DCA", f"{excess_spd*100:.2f}%" if np.isfinite(excess_spd) else "n/a")
        delta_spd = spd_strat - spd_bench
        st.metric("Extra sats per $", f"{delta_spd:,.1f}" if np.isfinite(delta_spd) else "n/a")
    
    st.markdown("---")
    st.markdown("#### Equity curves")
    
    equity_plot_data = backtest_df[["equity_benchmark", "equity_strategy"]].copy()
    equity_plot_data.columns = ["Regular DCA", "6-Agent Strategy"]
    st.line_chart(equity_plot_data, use_container_width=True)
    
    st.markdown("#### Historical allocation multiplier")
    st.line_chart(backtest_df["alloc"], use_container_width=True)
    st.caption("Regular DCA = 1.0× daily. Agent scales based on 6-agent ensemble.")

    st.markdown("#### Rolling Excess SPD vs DCA")
    roll_win = st.slider("Rolling window (days)", 5, 365, 180, 5)
    excess_spd_series = backtest_df["spd_strategy"] / backtest_df["spd_benchmark"] - 1.0
    excess_spd_roll = excess_spd_series.rolling(roll_win, min_periods=roll_win//3).mean()
    st.line_chart(excess_spd_roll.rename("Excess SPD (agent/DCA - 1)"), use_container_width=True)
    st.caption("Above 0 = agent accumulates more sats per dollar than DCA.")


# ============================================================================
# TAB 4 – SENTIMENT OVERVIEW
# ============================================================================
with tab_sentiment:
    st.subheader("Sentiment overview (3 sources)")
    st.info("Using Fear & Greed imputation: fear_greed_value_feargreed_imputed")

    sent_freq = st.selectbox(
        "Aggregation frequency",
        options=["D", "W", "M"],
        index=1,
        format_func=lambda x: {"D": "Daily", "W": "Weekly", "M": "Monthly"}[x],
    )

    sent_tab_timeline, sent_tab_structure, sent_tab_diag = st.tabs(
        ["Timeline", "Structure", "Diagnostics"]
    )

    df_sent = load_sentiment_cached(sent_freq).copy()

    dmin = df_sent.index.min().date()
    dmax = df_sent.index.max().date()

    start_date, end_date = st.slider(
        "Date range",
        min_value=dmin,
        max_value=dmax,
        value=(dmin, dmax),
    )

    df_sent = df_sent.loc[pd.to_datetime(start_date): pd.to_datetime(end_date)].copy()

    if not isinstance(df_sent.index, pd.DatetimeIndex):
        df_sent.index = pd.to_datetime(df_sent.index)

    df_sent = df_sent.dropna(subset=["btc_price"])

    with sent_tab_timeline:
        st.markdown("#### BTC price + sentiment layers")

        c1, c2 = st.columns(2)
        with c1:
            show_people = st.checkbox("People (BitcoinTalk)", value=True)
            show_world  = st.checkbox("World (News)", value=True)
        with c2:
            show_market = st.checkbox("Market (F&G scaled)", value=True)
            show_price  = st.checkbox("BTC price overlay", value=True)

        try:
            fig = go.Figure()

            if show_people and "people_sent" in df_sent.columns:
                fig.add_trace(go.Scatter(
                    x=df_sent.index, y=df_sent["people_sent"],
                    mode="lines", name="People (BitcoinTalk)",
                    line=dict(color=SENTIMENT_COLORS["People (BitcoinTalk)"])
                ))

            if show_world and "world_sent" in df_sent.columns:
                fig.add_trace(go.Scatter(
                    x=df_sent.index, y=df_sent["world_sent"],
                    mode="lines", name="World (News)",
                    line=dict(color=SENTIMENT_COLORS["World (News)"])
                ))

            if show_market and "market_sent_scaled" in df_sent.columns:
                fig.add_trace(go.Scatter(
                    x=df_sent.index, y=df_sent["market_sent_scaled"],
                    mode="lines", name="Market (F&G scaled)",
                    line=dict(color=SENTIMENT_COLORS["Market (F&G scaled)"])
                ))

            if show_price and "btc_price" in df_sent.columns:
                fig.add_trace(go.Scatter(
                    x=df_sent.index, y=df_sent["btc_price"],
                    mode="lines", name="BTC price",
                    line=dict(color=SENTIMENT_COLORS["BTC price"]),
                    yaxis="y2"
                ))

            fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                yaxis=dict(title="Sentiment"),
                yaxis2=dict(title="BTC price", overlaying="y", side="right", type="log")
            )

            st.plotly_chart(fig, use_container_width=True)

        except:
            cols = []
            if show_people: cols.append("people_sent")
            if show_world: cols.append("world_sent")
            if show_market: cols.append("market_sent_scaled")
            if show_price: cols.append("btc_price")
            if cols:
                st.line_chart(df_sent[cols], use_container_width=True)

        st.caption("F&G scaled to [-1, 1] for comparison. Native scale is 0-100.")

    with sent_tab_structure:
        st.markdown("#### Sentiment state occupancy")

        eps = st.slider("Neutral zone threshold", 0.00, 0.20, 0.05, 0.01)

        use = pd.DataFrame(index=df_sent.index)
        if "people_sent" in df_sent.columns:
            use["people_sent"] = pd.to_numeric(df_sent["people_sent"], errors="coerce")
        if "world_sent" in df_sent.columns:
            use["world_sent"] = pd.to_numeric(df_sent["world_sent"], errors="coerce")
        if "market_sent_scaled" in df_sent.columns:
            use["market_sent_scaled"] = pd.to_numeric(df_sent["market_sent_scaled"], errors="coerce")

        def to_state(x):
            if pd.isna(x): return np.nan
            if x > eps: return "Positive"
            if x < -eps: return "Negative"
            return "Neutral"

        state_df = use.applymap(to_state)

        order = ["Negative", "Neutral", "Positive"]
        share = pd.DataFrame(index=state_df.columns, columns=order, dtype=float)
        for c in state_df.columns:
            counts = state_df[c].value_counts(normalize=True, dropna=True)
            for s_ in order:
                share.loc[c, s_] = counts.get(s_, 0.0)
        share = share.fillna(0)

        fig, ax = plt.subplots(figsize=(12, 3.6))
        left = np.zeros(len(share))
        y = np.arange(len(share.index))

        for s_ in order:
            ax.barh(y, share[s_].values, left=left, label=s_, alpha=0.9)
            left += share[s_].values

        ax.set_yticks(y)
        ax.set_yticklabels(share.index)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of time")
        ax.set_title("Sentiment State Occupancy")
        ax.grid(axis="x", alpha=0.2)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.tight_layout()

        st.pyplot(fig, clear_figure=True)

        st.markdown("#### Sentiment turning points (sign flips)")

        eps = st.slider(
            "Neutral zone for flip detection (eps)",
            min_value=0.00,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Values in [-eps, eps] are treated as Neutral and ignored for flip counting",
            key="flip_eps",
        )

        sent_cols = {
            "people_sent": "People (BitcoinTalk)",
            "world_sent": "World (News)",
            "market_sent_scaled": "Market (F&G scaled)",
        }

        def to_state(x, eps):
            if pd.isna(x):
                return np.nan
            if x > eps:
                return 1
            if x < -eps:
                return -1
            return 0

        state_df = pd.DataFrame(index=df_sent.index)
        for col in sent_cols:
            if col in df_sent.columns:
                state_df[col] = df_sent[col].apply(lambda x: to_state(x, eps))

        def count_flips(x):
            x = x.dropna()
            x = x[x != 0]
            return (x != x.shift(1)).sum()

        flips = (
            state_df
            .groupby(state_df.index.year)
            .apply(lambda g: pd.Series({
                sent_cols[c]: count_flips(g[c])
                for c in state_df.columns
            }))
        )

        fig, ax = plt.subplots(figsize=(12, 4))

        for label in flips.columns:
            ax.plot(flips.index, flips[label], marker="o", linewidth=2, label=label)

        ax.set_title("Turning Points: Positive ↔ Negative Flips per Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of flips")
        ax.grid(alpha=0.25)
        ax.legend()

        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        st.caption(
            "Counts only meaningful sentiment reversals (Positive ↔ Negative). "
            "Neutral periods are ignored to reduce noise."
        )

    with sent_tab_diag:
        st.markdown("#### Diagnostics (click through charts)")

        chart = st.selectbox(
            "Choose a diagnostic chart",
            [
                "Quadrants: Sentiment vs Realized Volatility (price-based)",
                "Quadrants: Sentiment vs Drawdown (price-based)",
                "Quadrants: Sentiment vs Trend Distance (price-based)",
                "Correlation heatmap: Sentiment + BTC metrics",
            ],
            key="sent_diag_chart",
        )

        vol_window = 26
        ma_window = 52

        if chart.startswith("Quadrants"):
            sent_source = st.radio(
                "Sentiment source (X-axis)",
                options=["people_sent", "world_sent", "fg"],
                format_func=lambda x: (
                    "People (BitcoinTalk)" if x == "people_sent"
                    else "World (News)" if x == "world_sent"
                    else "Fear & Greed (0–100)"
                ),
                horizontal=True,
                key="sent_diag_source",
            )

        if ("Realized Volatility" in chart) or ("Correlation heatmap" in chart):
            vol_window = st.slider("Vol window (periods)", 1, 104, 26, key="sent_vol_window_diag")

        if ("Trend Distance" in chart) or ("Correlation heatmap" in chart):
            ma_window = st.slider("MA window (periods)", 1, 156, 52, key="sent_ma_window_diag")

        df_m = _compute_price_metrics(df_sent, vol_window=vol_window, ma_window=ma_window)

        pointwise_vol = (vol_window == 1)
        pointwise_ma  = (ma_window == 1)

        if pointwise_vol:
            df_m["realized_vol"] = df_m["ret"].abs()

        if pointwise_ma:
            df_m["trend_dist"] = df_m["ret"]

        base_cols = ["people_sent", "world_sent", "fg", "btc_price", "ret", "realized_vol", "drawdown", "trend_dist"]
        available = [c for c in base_cols if c in df_m.columns]
        tmp = df_m[available].copy()

        # ---------------------------
        # BRANCH: Quadrants vs Heatmap
        # ---------------------------
        if chart.startswith("Quadrants"):

            # Choose Y metric
            if "Realized Volatility" in chart:
                ycol = "realized_vol"
                yname = ("Shock magnitude |return|" if pointwise_vol
                        else f"Realized volatility (rolling {vol_window})")
            elif "Drawdown" in chart:
                ycol = "drawdown"
                yname = "Drawdown (from peak)"
            else:
                ycol = "trend_dist"
                yname = ("Return (1-period)" if pointwise_ma
                        else f"Trend distance (price vs MA {ma_window})")

            # ---- Quadrant plot (THIS USED TO BE INDENTED WRONG) ----
            sub = tmp[[sent_source, ycol]].dropna()
            x = sub[sent_source].astype(float)
            y = sub[ycol].astype(float)

            if sent_source == "fg":
                x0 = 50.0
                xlabel = "Fear & Greed (0–100)"
            else:
                x0 = 0.0
                xlabel = "Sentiment (scaled)"

            y0 = float(y.median())

            title_map = {
                "people_sent": "People",
                "world_sent": "World",
                "fg": "Fear & Greed",
            }
            title = f"{title_map.get(sent_source, sent_source)} × {yname}"

            metric_explain = {
                "realized_vol": (
                    "Pointwise stress proxy = |return| (absolute percent change). Higher = bigger single-period move."
                    if pointwise_vol else
                    f"Realized volatility = rolling std of returns (window={vol_window}). Higher = more turbulent."
                ),
                "drawdown": "Drawdown = % drop from prior peak. Closer to 0 = near peak; more negative = deeper drawdown.",
                "trend_dist": (
                    "Pointwise proxy = return (percent change for the period)."
                    if pointwise_ma else
                    f"Trend distance = distance from moving average (MA={ma_window}). Positive = above trend; negative = below trend."
                )
            }

            sent_explain = {
                "people_sent": "People sentiment comes from BitcoinTalk (scaled around 0).",
                "world_sent": "World sentiment comes from news sources (scaled around 0).",
                "fg": "Fear & Greed is a 0–100 index. 50 is the neutral midpoint."
            }

            st.markdown("**How to read this chart**")
            st.write(
                f"- **Each dot** is one period ({sent_freq}).\n"
                f"- **X-axis** is the selected sentiment source. {sent_explain.get(sent_source, '')}\n"
                f"- **Y-axis** is a BTC price-derived metric. {metric_explain.get(ycol, '')}\n"
            )

            fig = _plot_quadrant_matplotlib(
                x=x, y=y,
                x0=x0, y0=y0,
                title=title,
                xlabel=xlabel,
                ylabel=yname,
            )
            st.pyplot(fig)

            st.caption(
                "Quadrants use BTC-price-derived market metrics. "
                "Vertical line is sentiment=0 or F&G=50; horizontal line is the median of the market metric."
            )

        else:
            # ---- Heatmap ----
            heat_cols = {
                "People (BitcoinTalk)": "people_sent",
                "World (News)": "world_sent",
                "Fear & Greed (0–100)": "fg",
                "BTC price": "btc_price",
                "Returns": "ret",
                f"Realized vol ({vol_window})": "realized_vol",
                "Drawdown": "drawdown",
                f"Trend dist (MA {ma_window})": "trend_dist",
            }
            use = [v for v in heat_cols.values() if v in tmp.columns]
            corr = tmp[use].corr()

            corr.index = [k for k, v in heat_cols.items() if v in use]
            corr.columns = [k for k, v in heat_cols.items() if v in use]

            fig = _corr_heatmap_matplotlib(corr, title="Correlation matrix (Sentiment + BTC metrics)")
            st.pyplot(fig)

            st.caption(
                "Descriptive correlation only. Fear & Greed is included as native 0–100; "
                "volatility is computed from BTC returns (price-based)."
            )
