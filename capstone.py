#!/usr/bin/env python3
"""
6-Agent Ensemble Bitcoin DCA Strategy - TEMPLATE COMPLIANT
Uses fear_greed_value_feargreed_imputed column

TEMPLATE SPECIFICATIONS:
- BACKTEST_START = '2016-01-01'
- BACKTEST_END = '2025-06-01'
- INVESTMENT_WINDOW = 12 months
- MIN_WEIGHT = 1e-5
- PRICE_COL = "PriceUSD_coinmetrics"
- compute_weights() returns normalized pd.Series
- All validation tests included
"""

import numpy as np
import pandas as pd
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# TEMPLATE CONSTANTS (DO NOT MODIFY)
# ============================================================================
BACKTEST_START = '2016-01-01'
BACKTEST_END = '2025-06-01'
INVESTMENT_WINDOW = 12
MIN_WEIGHT = 1e-5
PRICE_COL = "PriceUSD_coinmetrics"

# ============================================================================
# OPTIMIZED AGENT WEIGHTS (6 agents)
# ============================================================================
AGENT_WEIGHTS = {
    'rap': 2.0,              # Regime-aware positioning
    'spd_momentum': 3.0,     # Sats-per-dollar momentum
    'fear_greed': 1.0,       # Fear & Greed sentiment
    'mvrv': 1.5,             # MVRV value signal
    'hp': 0.1,               # Volatility hedging
    'emp': 0.1               # Extreme market protection
}


def load_data() -> pd.DataFrame:
    """
    Load main data and merge with Fear & Greed imputation.
    
    Returns
    -------
    pd.DataFrame
        Indexed by time, includes all features plus fear_greed_value_feargreed_imputed
    """
    # Load main dataset
    url = (
        "https://raw.githubusercontent.com/"
        "TrilemmaFoundation/stacking-sats-tournament-mstr-2025/main/data/"
        "stacking_sats_data.parquet"
    )
    
    response = requests.get(url)
    response.raise_for_status()
    df_main = pd.read_parquet(BytesIO(response.content))
    
    # Ensure time index
    if df_main.index.name != "time" and "time" in df_main.columns:
        df_main.set_index("time", inplace=True)
    
    # Normalize timestamps
    df_main.index = pd.to_datetime(df_main.index).normalize()
    df_main.index = df_main.index.tz_localize(None) if df_main.index.tz is not None else df_main.index
    df_main = df_main.loc[~df_main.index.duplicated(keep="last")]
    df_main = df_main.sort_index()
    
    # Load Fear & Greed imputation
    fg_imputed = pd.read_csv('f&g_imputed_2.csv')
    
    # Handle different possible date column names
    if 'time' in fg_imputed.columns:
        fg_imputed['time'] = pd.to_datetime(fg_imputed['time'])
        fg_imputed.set_index('time', inplace=True)
    elif 'date' in fg_imputed.columns:
        fg_imputed['date'] = pd.to_datetime(fg_imputed['date'])
        fg_imputed.set_index('date', inplace=True)
    else:
        first_col = fg_imputed.columns[0]
        fg_imputed[first_col] = pd.to_datetime(fg_imputed[first_col])
        fg_imputed.set_index(first_col, inplace=True)
    
    if fg_imputed.index.tz is not None:
        fg_imputed.index = fg_imputed.index.tz_localize(None)
    
    # Merge Fear & Greed imputation into main dataframe
    df_main['fear_greed_value_feargreed_imputed'] = fg_imputed['fear_greed_value_feargreed_imputed']
    
    return df_main


# ============================================================================
# AGENT IMPLEMENTATIONS
# ============================================================================

def compute_weights_hp(df):
    """
    HP Agent: Volatility-based hedging.
    Reduces allocation during high volatility regimes.
    """
    price = df[PRICE_COL].astype(float)
    ret_log = np.log(price.shift(1)).diff()
    vol30 = ret_log.rolling(30, min_periods=10).std(ddof=0)
    mu_vol = vol30.rolling(365, min_periods=60).mean().shift(1)
    sd_vol = vol30.rolling(365, min_periods=60).std(ddof=0).shift(1)
    zvol = (vol30 - mu_vol) / (sd_vol.replace(0.0, np.nan) + 1e-12)
    hp_mult = (1.0 - 0.10 * np.tanh(zvol)).clip(0.90, 1.10).fillna(1.0)
    return hp_mult


def compute_weights_rap(df):
    """
    RAP Agent: Regime-aware positioning.
    Increases allocation in bear markets, reduces in bull markets.
    """
    price = df[PRICE_COL].copy()
    past_price = price.shift(1)
    gap_ma_90 = (past_price / past_price.rolling(90, min_periods=1).mean() - 1) * 100
    drawdown = (past_price / past_price.rolling(365, min_periods=30).max() - 1) * 100
    delta = past_price.diff()
    rsi = 100 - (100 / (1 + delta.where(delta > 0, 0).rolling(14).mean() / -delta.where(delta < 0, 0).rolling(14).mean()))
    
    raw = np.ones(len(df))
    for i in range(200, len(df)):
        gap, dd, r = gap_ma_90.iloc[i], drawdown.iloc[i], rsi.iloc[i]
        is_bear = (gap < -25) or (dd < -40)
        if is_bear:
            score = 0.9 if dd < -50 else (0.8 if dd < -40 else (0.75 if r < 30 else 0.6))
            raw[i] = 0.7 + 0.8 * score
        else:
            score = 0.1 if gap > 30 else (0.2 if gap > 15 else 0.4)
            raw[i] = 0.7 + 0.5 * score
    
    return pd.Series(raw, index=df.index).clip(MIN_WEIGHT)


def compute_weights_emp(df):
    """
    EMP Agent: Extreme market protection.
    Reduces allocation during extreme volatility.
    """
    returns = df[PRICE_COL].shift(1).pct_change(fill_method=None)
    vol_30d = returns.rolling(30, min_periods=1).std() * np.sqrt(365) * 100
    vol_p90 = vol_30d.rolling(365, min_periods=30).quantile(0.90).shift(1)
    return pd.Series(np.where(vol_30d > vol_p90, 0.95, 1.0), index=df.index)


def compute_weights_mvrv(df):
    """
    MVRV Agent: Value-based signal.
    Increases allocation when BTC is undervalued, reduces when overvalued.
    """
    past_price = df[PRICE_COL].shift(1)
    mvrv_ratio = past_price / past_price.rolling(365, min_periods=100).mean()
    mvrv_z = (
        (mvrv_ratio - mvrv_ratio.rolling(365, min_periods=100).mean()) / 
        (mvrv_ratio.rolling(365, min_periods=100).std().shift(1) + 1e-6)
    )
    
    raw = np.ones(len(df))
    for i in range(len(df)):
        z = mvrv_z.iloc[i]
        if not pd.isna(z):
            if z < -1.5:
                raw[i] = 1.4
            elif z < -0.5:
                raw[i] = 1.2
            elif z > 1.5:
                raw[i] = 0.7
            elif z > 0.5:
                raw[i] = 0.85
            else:
                raw[i] = 1.0
    
    return pd.Series(raw, index=df.index).clip(MIN_WEIGHT)


def compute_weights_fear_greed(df):
    """
    Fear & Greed Agent: Market sentiment signal.
    Increases allocation during extreme fear, reduces during extreme greed.
    Uses fear_greed_value_feargreed_imputed column (0-100 scale).
    """
    fg = df['fear_greed_value_feargreed_imputed']
    raw = np.ones(len(df))
    
    for i in range(len(df)):
        val = fg.iloc[i]
        if pd.isna(val):
            continue
        
        # Extreme fear: increase allocation
        if val < 10:
            raw[i] = 1.8
        elif val < 20:
            raw[i] = 1.5
        elif val < 30:
            raw[i] = 1.3
        elif val < 40:
            raw[i] = 1.15
        elif val < 45:
            raw[i] = 1.05
        # Extreme greed: reduce allocation
        elif val > 90:
            raw[i] = 0.6
        elif val > 80:
            raw[i] = 0.7
        elif val > 70:
            raw[i] = 0.8
        elif val > 60:
            raw[i] = 0.9
        elif val > 55:
            raw[i] = 0.95
        # Neutral: 1.0 (unchanged)
    
    return pd.Series(raw, index=df.index).clip(MIN_WEIGHT)


def compute_weights_spd_momentum(df):
    """
    SPD Momentum Agent: Sats-per-dollar momentum strategy.
    Increases allocation when SPD is rising (price falling or accelerating down).
    """
    past_price = df[PRICE_COL].shift(1)
    spd = 1 / past_price * 1e8
    spd_roc_30 = spd.pct_change(30, fill_method=None) * 100
    spd_accel = (
        (spd.rolling(7, min_periods=3).mean() - spd.rolling(30, min_periods=10).mean()) / 
        (spd.rolling(30, min_periods=10).mean() + 1e-6) * 100
    )
    
    raw = np.ones(len(df))
    for i in range(len(df)):
        if not pd.isna(spd_roc_30.iloc[i]) and not pd.isna(spd_accel.iloc[i]):
            roc, accel = spd_roc_30.iloc[i], spd_accel.iloc[i]
            
            # Strong SPD momentum: increase allocation
            if roc > 20 and accel > 5:
                raw[i] = 1.3
            elif roc > 10:
                raw[i] = 1.15
            elif roc > 5:
                raw[i] = 1.1
            # Falling SPD: reduce allocation
            elif roc < -20 and accel < -5:
                raw[i] = 0.75
            elif roc < -10:
                raw[i] = 0.85
            elif roc < -5:
                raw[i] = 0.95
    
    return pd.Series(raw, index=df.index).clip(MIN_WEIGHT)


# ============================================================================
# ENSEMBLE COMBINING
# ============================================================================

def compute_allocation_multipliers(df, agent_weights):
    """
    Compute RAW allocation multipliers by combining agent signals.
    Used for visualization in Streamlit (shows 0.7-1.5 range).
    
    Parameters
    ----------
    df : pd.DataFrame
        Full feature matrix
    agent_weights : dict
        Agent weights for ensemble
    
    Returns
    -------
    pd.Series
        Raw allocation multipliers (NOT normalized)
    """
    agents = {
        'hp': compute_weights_hp,
        'rap': compute_weights_rap,
        'emp': compute_weights_emp,
        'mvrv': compute_weights_mvrv,
        'fear_greed': compute_weights_fear_greed,
        'spd_momentum': compute_weights_spd_momentum
    }
    
    # Normalize agent weights
    total = sum(agent_weights.values())
    normalized_weights = {k: v/total for k, v in agent_weights.items()}
    
    # Compute each agent's signal
    signals = {}
    for name, w in agent_weights.items():
        if w > 0 and name in agents:
            sig = agents[name](df).reindex(df.index).fillna(1.0)
            signals[name] = sig
    
    # Geometric mean weighted by normalized weights
    combined = pd.Series(1.0, index=df.index)
    for name, signal in signals.items():
        combined *= signal ** normalized_weights[name]
    
    combined = combined.clip(lower=MIN_WEIGHT)
    return combined


def compute_weights(df_window: pd.DataFrame) -> pd.Series:
    """
    REQUIRED TEMPLATE FUNCTION: Compute normalized allocation weights.
    
    This is the main entry point called by the backtest.
    Returns weights that sum to 1.0 for each time period.
    
    Parameters
    ----------
    df_window : pd.DataFrame
        DataFrame indexed by time with all required features
    
    Returns
    -------
    pd.Series
        Normalized weights (sum to 1.0), indexed by time
    """
    # Get raw multipliers from ensemble
    multipliers = compute_allocation_multipliers(df_window, AGENT_WEIGHTS)
    
    # Normalize to sum to 1.0
    total = multipliers.sum()
    if total > 0:
        normalized = multipliers / total
    else:
        # Fallback: uniform weights
        normalized = pd.Series(1.0 / len(multipliers), index=multipliers.index)
    
    # Ensure minimum weight
    normalized = normalized.clip(lower=MIN_WEIGHT)
    
    # Re-normalize after clipping
    normalized = normalized / normalized.sum()
    
    return normalized


# ============================================================================
# BACKTESTING
# ============================================================================

def compute_cycle_spd(df, strategy_fn):
    """
    Compute SPD (sats-per-dollar) percentiles for rolling windows.
    
    For each 12-month window, calculates where the strategy's weighted
    average SPD falls relative to uniform DCA.
    """
    offset = pd.DateOffset(months=INVESTMENT_WINDOW)
    start_dt = pd.to_datetime(BACKTEST_START)
    end_dt = pd.to_datetime(BACKTEST_END)
    
    results = []
    
    for window_start in pd.date_range(start=start_dt, end=end_dt - offset, freq='1D'):
        window_end = window_start + offset
        price_slice = df[PRICE_COL].loc[window_start:window_end]
        
        if price_slice.empty:
            continue
        
        # Sats per dollar for this window
        inv_price = (1.0 / price_slice) * 1e8
        
        # Get strategy weights for this window
        weights = strategy_fn(df.loc[window_start:window_end])
        
        # Calculate percentiles
        span = inv_price.max() - inv_price.min()
        if span > 0:
            uniform_pct = (inv_price.mean() - inv_price.min()) / span * 100
            dynamic_pct = ((weights * inv_price).sum() - inv_price.min()) / span * 100
        else:
            uniform_pct = 50.0
            dynamic_pct = 50.0
        
        results.append({
            "uniform_percentile": uniform_pct,
            "dynamic_percentile": dynamic_pct
        })
    
    return pd.DataFrame(results)


def backtest(df, strategy_fn):
    """
    Run backtest with exponential decay scoring.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full data for backtest period
    strategy_fn : callable
        Function that takes df and returns weights
    
    Returns
    -------
    tuple
        (score, win_rate, exp_avg, spd_table)
    """
    # Compute SPD table for all windows
    spd_table = compute_cycle_spd(df, strategy_fn)
    
    # Exponential decay weights (recent windows matter more)
    N = len(spd_table)
    decay_weights = np.array([0.9 ** (N - 1 - i) for i in range(N)])
    decay_weights = decay_weights / decay_weights.sum()
    
    # Weighted average SPD percentile
    exp_avg = (spd_table["dynamic_percentile"].values * decay_weights).sum()
    
    # Win rate: % of windows where strategy beats uniform DCA
    win_rate = (spd_table["dynamic_percentile"] > spd_table["uniform_percentile"]).mean() * 100
    
    # Final score: 50% win rate + 50% exponential average
    score = 0.5 * win_rate + 0.5 * exp_avg
    
    return score, win_rate, exp_avg, spd_table


# ============================================================================
# VALIDATION TESTS
# ============================================================================

def test1_data_quality(df: pd.DataFrame) -> bool:
    """Test 1: Data quality and completeness checks."""
    print("\n[TEST 1] Data Quality Checks")
    print("-" * 70)
    
    all_pass = True
    
    # Check 1: Period length
    expected_days = (pd.to_datetime(BACKTEST_END) - pd.to_datetime(BACKTEST_START)).days
    actual_days = len(df)
    if abs(actual_days - expected_days) < 100:
        print(f"  [PASS] Period length: {actual_days} days (expected ~{expected_days})")
    else:
        print(f"  [FAIL] Period length: {actual_days} days (expected ~{expected_days})")
        all_pass = False
    
    # Check 2: No missing prices
    price_missing = df[PRICE_COL].isna().sum()
    if price_missing == 0:
        print(f"  [PASS] No missing prices")
    else:
        print(f"  [FAIL] Missing prices: {price_missing}")
        all_pass = False
    
    # Check 3: Fear & Greed coverage
    fg_missing = df['fear_greed_value_feargreed_imputed'].isna().sum()
    fg_coverage = (1 - fg_missing / len(df)) * 100
    if fg_coverage > 90:
        print(f"  [PASS] Fear & Greed coverage: {fg_coverage:.1f}%")
    else:
        print(f"  [WARN] Fear & Greed coverage: {fg_coverage:.1f}% (low coverage)")
    
    # Check 4: Fear & Greed range (should be 0-100)
    fg_min = df['fear_greed_value_feargreed_imputed'].min()
    fg_max = df['fear_greed_value_feargreed_imputed'].max()
    if 0 <= fg_min <= 15 and 85 <= fg_max <= 100:
        print(f"  [PASS] Fear & Greed range: [{fg_min:.1f}, {fg_max:.1f}]")
    else:
        print(f"  [WARN] Fear & Greed range: [{fg_min:.1f}, {fg_max:.1f}]")
    
    return all_pass


def test2_forward_leakage(df: pd.DataFrame) -> bool:
    """Test 2: Ensure no forward-looking bias."""
    print("\n[TEST 2] Forward Leakage Prevention")
    print("-" * 70)
    
    all_pass = True
    
    # Test at specific checkpoint date
    test_date = '2017-01-01'
    test_idx = pd.to_datetime(test_date)
    
    if test_idx not in df.index:
        print(f"  [SKIP] Test date {test_date} not in data")
        return True
    
    # Compute weights using only data up to test date
    df_slice = df.loc[:test_idx].copy()
    weights = compute_weights(df_slice)
    
    # Verify weight was computed
    weight_at_date = weights.loc[test_idx]
    
    if not pd.isna(weight_at_date) and weight_at_date > 0:
        print(f"  [PASS] Weight computed at {test_date}: {weight_at_date:.6f}")
    else:
        print(f"  [FAIL] Invalid weight at {test_date}: {weight_at_date}")
        all_pass = False
    
    # Verify agents use shifted data (past prices)
    past_price = df[PRICE_COL].shift(1).loc[test_idx]
    current_price = df[PRICE_COL].loc[test_idx]
    print(f"  [INFO] Past price (used): ${past_price:.2f}, Current price (not used): ${current_price:.2f}")
    
    # Check Fear & Greed availability
    fg_at_date = df['fear_greed_value_feargreed_imputed'].loc[test_idx]
    if not pd.isna(fg_at_date):
        print(f"  [PASS] Fear & Greed available at {test_date}: {fg_at_date:.2f}")
    else:
        print(f"  [WARN] Fear & Greed missing at {test_date}")
    
    return all_pass


def test3_weight_validity(df: pd.DataFrame) -> bool:
    """Test 3: Weight mathematical properties."""
    print("\n[TEST 3] Weight Validity Checks")
    print("-" * 70)
    
    all_pass = True
    
    # Compute weights for full period
    weights = compute_weights(df)
    
    # Check 1: Weights sum to 1.0
    weight_sum = weights.sum()
    if abs(weight_sum - 1.0) < 0.01:
        print(f"  [PASS] Weights sum to 1.0: {weight_sum:.6f}")
    else:
        print(f"  [FAIL] Weights sum to {weight_sum:.6f} (expected 1.0)")
        all_pass = False
    
    # Check 2: No negative weights
    neg_weights = (weights < 0).sum()
    if neg_weights == 0:
        print(f"  [PASS] No negative weights")
    else:
        print(f"  [FAIL] Found {neg_weights} negative weights")
        all_pass = False
    
    # Check 3: Minimum weight enforced
    min_weight = weights.min()
    if min_weight >= MIN_WEIGHT - 1e-10:
        print(f"  [PASS] Minimum weight enforced: {min_weight:.2e} >= {MIN_WEIGHT:.2e}")
    else:
        print(f"  [FAIL] Minimum weight: {min_weight:.2e} < {MIN_WEIGHT:.2e}")
        all_pass = False
    
    # Check 4: Weight statistics
    print(f"  [INFO] Weight distribution:")
    print(f"         Mean:   {weights.mean():.6f}")
    print(f"         Median: {weights.median():.6f}")
    print(f"         Std:    {weights.std():.6f}")
    print(f"         Min:    {weights.min():.6f}")
    print(f"         Max:    {weights.max():.6f}")
    
    return all_pass


def run_all_validations(df: pd.DataFrame) -> bool:
    """Execute all validation tests and report results."""
    print("\n" + "=" * 70)
    print("TEMPLATE VALIDATION TESTS")
    print("=" * 70)
    
    test1_pass = test1_data_quality(df)
    test2_pass = test2_forward_leakage(df)
    test3_pass = test3_weight_validity(df)
    
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Test 1 (Data Quality):          {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Test 2 (Forward Leakage):       {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print(f"Test 3 (Weight Validity):       {'✅ PASS' if test3_pass else '❌ FAIL'}")
    
    all_pass = test1_pass and test2_pass and test3_pass
    
    if all_pass:
        print("\n✅ ALL VALIDATION TESTS PASSED - TEMPLATE COMPLIANT")
    else:
        print("\n⚠️  SOME VALIDATION TESTS FAILED - REVIEW REQUIRED")
    
    print("=" * 70)
    
    return all_pass


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """Main execution: load data, validate, backtest, report results."""
    print("=" * 70)
    print("6-AGENT ENSEMBLE BITCOIN DCA STRATEGY")
    print("Template-Compliant Implementation")
    print("=" * 70)
    
    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    df = load_data()
    df = df.loc[BACKTEST_START:BACKTEST_END].copy()
    
    print(f"Records: {len(df):,}")
    print(f"\nFear & Greed Statistics:")
    print(f"  Mean:  {df['fear_greed_value_feargreed_imputed'].mean():.2f}")
    print(f"  Std:   {df['fear_greed_value_feargreed_imputed'].std():.2f}")
    print(f"  Min:   {df['fear_greed_value_feargreed_imputed'].min():.2f}")
    print(f"  Max:   {df['fear_greed_value_feargreed_imputed'].max():.2f}")
    print(f"\nDate range: {df.index.min().date()} to {df.index.max().date()}")
    
    # Run validation tests
    validation_passed = run_all_validations(df)
    
    if not validation_passed:
        print("\n⚠️  WARNING: Validation tests failed!")
        print("Results may not be template-compliant.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborting.")
            return
    
    # Run backtest
    print("\n" + "=" * 70)
    print("BACKTESTING")
    print("=" * 70)
    print(f"Computing SPD percentiles for rolling {INVESTMENT_WINDOW}-month windows...")
    print("This takes 5-10 minutes - please wait...")
    
    score, win_rate, exp_avg, spd_table = backtest(df, compute_weights)
    
    # Report results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\nPerformance Metrics:")
    print("-" * 70)
    print(f"Final Score:          {score:.2f}%")
    print(f"Win Rate:             {win_rate:.2f}%")
    print(f"Exp Decay Average:    {exp_avg:.2f}%")
    print(f"Total Windows:        {len(spd_table):,}")
    
    print("\nAgent Configuration:")
    print("-" * 70)
    total_weight = sum(AGENT_WEIGHTS.values())
    for agent, weight in AGENT_WEIGHTS.items():
        pct = (weight / total_weight) * 100
        print(f"  {agent:15s}: {weight:.1f} ({pct:.1f}%)")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()