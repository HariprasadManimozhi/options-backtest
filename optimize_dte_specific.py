"""
NIFTY 0-DTE Specific Parameter Optimizer
========================================
Runs a grid search and ranks strategies INDEPENDENTLY for each DTE type.
Outputs top 50 strategies for 0DTE, 1DTE, 2DTE, etc.

Usage:
    python optimize_dte_specific.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta, date as datetime_date
from pathlib import Path
import glob
import itertools
import multiprocessing
from functools import partial
import time as time_module
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

FOLDER_PATH = "./options_weekly_weekwise/FY2025" 

# Parameter Grid (Same as previous run)
PARAM_GRID = {
    'pe_offset': [0,1,2,3,4,5],
    'ce_offset': [0,1,2,3,4,5],
    'stoploss_pct': [75, 80, 85, 90, 95, 100],
    'profit_target_pct': [45, 50, 55, 60, 65, 70, 75],
    'index_movement': [100, 150, 200, 250],
    'entry_time': [time(9, 20)],
    'exit_time': [time(15, 20)],
    'lot_size': [1] 
}

# ============================================================================
# TRADE LOGIC & WORKER
# ============================================================================

def simulate_trade(day_df, date, dte, params):
    """Multi-Entry Logic"""
    if day_df.empty: return None
    
    total_day_pnl = 0
    trade_count = 0
    
    current_entry_time = params['entry_time']
    final_exit_time = params['exit_time']
    
    def add_minutes(t, mins):
        dummy_date = datetime_date.min
        dt = datetime.combine(dummy_date, t)
        return (dt + timedelta(minutes=mins)).time()

    while current_entry_time < final_exit_time:
        entry_mask = day_df['time_only'] == current_entry_time
        entry_rows = day_df[entry_mask]
        if entry_rows.empty: break
            
        index_at_entry = entry_rows.iloc[0]['index_close']
        if pd.isna(index_at_entry): break
        
        atm_strike = round(index_at_entry / 50) * 50
        pe_strike = int(atm_strike - (params['pe_offset'] * 50))
        ce_strike = int(atm_strike + (params['ce_offset'] * 50))
        
        pe_entry = entry_rows[(entry_rows['strike_price'] == pe_strike) & (entry_rows['option_type'] == 'PE')]
        ce_entry = entry_rows[(entry_rows['strike_price'] == ce_strike) & (entry_rows['option_type'] == 'CE')]
        if pe_entry.empty or ce_entry.empty: break
            
        pe_entry_price = pe_entry.iloc[0]['close']
        ce_entry_price = ce_entry.iloc[0]['close']
        total_entry_premium = pe_entry_price + ce_entry_price
        
        stoploss_price = total_entry_premium * (1 + params['stoploss_pct'] / 100)
        profit_target_price = total_entry_premium * (1 - params['profit_target_pct'] / 100)
        index_upper = index_at_entry + params['index_movement']
        index_lower = index_at_entry - params['index_movement']
        
        monitoring_mask = (day_df['time_only'] >= current_entry_time) & (day_df['time_only'] <= final_exit_time)
        monitoring_data = day_df[monitoring_mask]
        if monitoring_data.empty: break

        pe_data = monitoring_data[(monitoring_data['strike_price'] == pe_strike) & (monitoring_data['option_type'] == 'PE')]
        ce_data = monitoring_data[(monitoring_data['strike_price'] == ce_strike) & (monitoring_data['option_type'] == 'CE')]
        if pe_data.empty or ce_data.empty: break
        
        merged = pd.merge(
            pe_data[['timestamp', 'time_only', 'close', 'index_close']], 
            ce_data[['timestamp', 'close']], 
            on='timestamp', suffixes=('_pe', '_ce'), how='inner'
        )
        merged['total_premium'] = merged['close_pe'] + merged['close_ce']
        merged.sort_values('timestamp', inplace=True)
        
        merged = merged[merged['time_only'] > current_entry_time]
        if merged.empty: break
        
        sl_hit = merged['total_premium'] >= stoploss_price
        pt_hit = merged['total_premium'] <= profit_target_price
        idx_breach = (merged['index_close'] >= index_upper) | (merged['index_close'] <= index_lower)
        time_exit = merged['time_only'] >= final_exit_time
        
        exit_mask = sl_hit | pt_hit | idx_breach | time_exit
        if exit_mask.any():
            exit_row = merged.loc[exit_mask.idxmax()]
            trade_exit_time = exit_row['time_only']
        else:
            exit_row = merged.iloc[-1]
            trade_exit_time = exit_row['time_only']
            
        exit_premium = exit_row['total_premium']
        pnl = (total_entry_premium - exit_premium) * params['lot_size'] * 50
        
        total_day_pnl += pnl
        trade_count += 1
        current_entry_time = add_minutes(trade_exit_time, 5)

    if trade_count == 0: return None
    return total_day_pnl

def evaluate_batch(params, daily_datasets):
    """
    Run simulation, but aggregate PnL SEPARATELY by DTE.
    """
    # Dict to hold results per DTE
    # Structure: {'0DTE': [pnl1, pnl2], '1DTE': [...]}
    dte_results = {} 
    
    for day_data in daily_datasets:
        pnl = simulate_trade(day_data['df'], day_data['date'], day_data['dte'], params)
        if pnl is not None:
            dte_key = day_data['dte']
            if dte_key not in dte_results:
                dte_results[dte_key] = []
            dte_results[dte_key].append(pnl)
    
    # Summarize per DTE
    summary = {}
    for dte, pnls in dte_results.items():
        count = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        total_pnl = sum(pnls)
        
        # Simple Max Drawdown calculation for ranking
        cum_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdown = running_max - cum_pnl
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
        
        summary[dte] = {
            'total_pnl': total_pnl,
            'trades': count,
            'win_rate': (wins/count*100) if count > 0 else 0,
            'max_dd': max_dd
        }
        
    return {'params': params, 'dte_metrics': summary}

# ============================================================================
# MAIN
# ============================================================================

def load_all_data(folder_path):
    print("📂 Loading data...")
    csv_files = sorted(glob.glob(str(Path(folder_path) / "*.csv")))
    daily_datasets = []
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['time_only'] = df['timestamp'].dt.time
            
            dates = df['date'].unique()
            for d in dates:
                day_subset = df[df['date'] == d].copy()
                dtes = day_subset['DTE'].unique()
                for dte in dtes:
                    final_day_df = day_subset[day_subset['DTE'] == dte].copy()
                    daily_datasets.append({
                        'df': final_day_df,
                        'date': d,
                        'dte': dte
                    })
        except Exception: pass
            
    # Verify uniqueness of DTEs found
    all_dtes = set(d['dte'] for d in daily_datasets)
    print(f"✅ Loaded {len(daily_datasets)} sets. DTEs found: {sorted(list(all_dtes))}")
    return daily_datasets

def main():
    # 1. Load Data
    daily_data = load_all_data(FOLDER_PATH)
    if not daily_data: return

    # 2. Generate Params
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    param_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"📊 Optimizing {len(param_list)} combos across {multiprocessing.cpu_count()} cores.")
    
    # 3. Run Optimization
    start_time = time_module.time()
    worker_func = partial(evaluate_batch, daily_datasets=daily_data)
    
    results = []
    completed = 0
    total = len(param_list)
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for res in pool.imap_unordered(worker_func, param_list):
            results.append(res)
            completed += 1
            if completed % 100 == 0:
                elapsed = time_module.time() - start_time
                avg = elapsed / completed
                rem = (total - completed) * avg / 60
                print(f"  [{completed}/{total}] {completed/total*100:.1f}% done. Est remaining: {rem:.1f} min", end='\r')
    print()
    
    # 4. Aggregation & Selection
    print("✅ Logic complete. Aggregating results per DTE...")
    
    # We need to pivot the data: Group by DTE, then Sort by PnL
    # Collect all metrics for each DTE
    # Structure: {'0DTE': [ {'params':..., 'pnl':...}, ...], '1DTE': ...}
    
    dte_collections = {}
    
    for res in results:
        params = res['params']
        metrics_map = res['dte_metrics'] # {'0DTE': {...}, '1DTE': {...}}
        
        for dte_key, metrics in metrics_map.items():
            if dte_key not in dte_collections:
                dte_collections[dte_key] = []
            
            # Fuse params and metrics into one dict row
            row = params.copy()
            row.update(metrics)
            dte_collections[dte_key].append(row)
            
    # 5. Save Top 50 for each DTE
    output_dir = Path("./backtest_results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for dte_key, rows in dte_collections.items():
        if not rows: continue
        
        # Convert to DF
        df = pd.DataFrame(rows)
        # Sort by PnL Descending
        df = df.sort_values('total_pnl', ascending=False)
        
        # Take Top 50
        top_50 = df.head(50)
        
        filename = f"best_strategies_{dte_key}_{timestamp}.csv"
        path = output_dir / filename
        top_50.to_csv(path, index=False)
        
        print(f"💾 {dte_key}: Saved top 50 to {filename} (Best PnL: ₹{top_50.iloc[0]['total_pnl']:,.0f})")

if __name__ == "__main__":
    main()
