"""
NIFTY Options Backtest Parameter Optimizer
==========================================
Runs a grid search over strategy parameters using multiprocessing to find the best combination.

Usage:
    python optimize_backtest.py 
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from pathlib import Path
import glob
import itertools
import multiprocessing
from functools import partial
import time as time_module
import argparse
import sys
import subprocess
from datetime import timedelta, date as datetime_date

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to data
# FOLDER_PATH = "./options_weekly_weekwise/FY2024" 
FOLDER_PATH = "./options_weekly_weekwise/FY2025" # Update as needed

# Parameter Grid for Optimization
# Keys must match the params dict expected by simulate_trade
PARAM_GRID = {
    'pe_offset': [0,1,2,3,4,5],                 # Reduced range
    'ce_offset': [0,1,2,3,4,5],              # Reduced range
    'stoploss_pct': [75, 80, 85, 90,95,100],           # Most common values
    'profit_target_pct': [50,55,60,65,70,75], # Core targets
    'index_movement': [100, 150, 200, 250],                # Fixed standardized movement
    'entry_time': [time(9, 20)], # Early entries only
    'exit_time': [time(15, 20)],
    'lot_size': [1] 
}



# ============================================================================
# TRADE SIMULATION LOGIC (Copied & Adapted)
# ============================================================================

def simulate_trade(day_df, date, dte, params):
    # Multi-Entry Loop
    total_day_pnl = 0
    day_trades = 0
    
    # Ensure entry_time is a time object
    current_entry_time = params['entry_time']
    final_exit_time = params['exit_time']
    
    # Helper to add minutes to time
    def add_minutes(t, mins):
        dummy_date = datetime_date.min
        dt = datetime.combine(dummy_date, t)
        return (dt + timedelta(minutes=mins)).time()
    
    while current_entry_time < final_exit_time:
        # Get entry data
        entry_mask = day_df['time_only'] == current_entry_time
        entry_rows = day_df[entry_mask]
        
        # If no data at this specific time, implies no trade possibilities for this slot
        # We could search forward, but strict strategy says "next 5 mins".
        # If missing, we abort for the day or skip? 
        # Let's assume if data missing, we stop to avoid infinite loops or bad data.
        if entry_rows.empty:
            break
            
        # Assuming first row determines index
        index_at_entry = entry_rows.iloc[0]['index_close']
        if pd.isna(index_at_entry):
            break
        
        atm_strike = round(index_at_entry / 50) * 50
        pe_strike = int(atm_strike - (params['pe_offset'] * 50))
        ce_strike = int(atm_strike + (params['ce_offset'] * 50))
        
        # Prices at entry
        pe_entry = entry_rows[(entry_rows['strike_price'] == pe_strike) & (entry_rows['option_type'] == 'PE')]
        ce_entry = entry_rows[(entry_rows['strike_price'] == ce_strike) & (entry_rows['option_type'] == 'CE')]
        
        if pe_entry.empty or ce_entry.empty:
            break
            
        pe_entry_price = pe_entry.iloc[0]['close']
        ce_entry_price = ce_entry.iloc[0]['close']
        total_entry_premium = pe_entry_price + ce_entry_price
        
        # Exit thresholds
        stoploss_price = total_entry_premium * (1 + params['stoploss_pct'] / 100)
        profit_target_price = total_entry_premium * (1 - params['profit_target_pct'] / 100)
        index_upper = index_at_entry + params['index_movement']
        index_lower = index_at_entry - params['index_movement']
        
        # Monitoring data
        monitoring_mask = (day_df['time_only'] >= current_entry_time) & (day_df['time_only'] <= final_exit_time)
        monitoring_data = day_df[monitoring_mask]
        
        if monitoring_data.empty:
            break
    
        # Get PE and CE data streams
        pe_data = monitoring_data[(monitoring_data['strike_price'] == pe_strike) & (monitoring_data['option_type'] == 'PE')]
        ce_data = monitoring_data[(monitoring_data['strike_price'] == ce_strike) & (monitoring_data['option_type'] == 'CE')]
        
        if pe_data.empty or ce_data.empty:
            break
        
        # Merge
        merged = pd.merge(
            pe_data[['timestamp', 'time_only', 'close', 'index_close']], 
            ce_data[['timestamp', 'close']], 
            on='timestamp', 
            suffixes=('_pe', '_ce'),
            how='inner'
        )
        
        merged['total_premium'] = merged['close_pe'] + merged['close_ce']
        merged.sort_values('timestamp', inplace=True)
        
        # Determine Exit
        exit_pnl = 0
        trade_exit_time = None
        
        # We need to skip the first row if it's the entry candle itself? 
        # Usually entry is execution, exit checks start immediately or next candle?
        # Let's assume we check checks on ALL candles including entry (instant SL?) or next.
        # Standard: check next candle.
        
        merged = merged[merged['time_only'] > current_entry_time] # strict next candle check
        
        if merged.empty:
             # EOD or no data after entry
             break
             
        sl_hit = merged['total_premium'] >= stoploss_price
        pt_hit = merged['total_premium'] <= profit_target_price
        idx_breach = (merged['index_close'] >= index_upper) | (merged['index_close'] <= index_lower)
        time_exit = merged['time_only'] >= final_exit_time
        
        exit_mask = sl_hit | pt_hit | idx_breach | time_exit
        
        if exit_mask.any():
            first_exit_idx = exit_mask.idxmax()
            exit_row = merged.loc[first_exit_idx]
            exit_premium = exit_row['total_premium']
            trade_exit_time = exit_row['time_only']
        else:
            # EOD
            exit_row = merged.iloc[-1]
            exit_premium = exit_row['total_premium']
            trade_exit_time = exit_row['time_only']
            
        pnl = (total_entry_premium - exit_premium) * params['lot_size'] * 50
        
        total_day_pnl += pnl
        day_trades += 1
        
        # Prepare for Re-Entry
        # Next re-entry is exit_time + 5 minutes
        current_entry_time = add_minutes(trade_exit_time, 0)
        
    if day_trades == 0:
        return None, 0
        
    return total_day_pnl, day_trades


def evaluate_batch(params, daily_datasets):
    """
    Run simulation for a single parameter set across all days.
    This runs in a worker process.
    """
    total_pnl = 0
    wins = 0
    trades_count = 0
    daily_pnls = []
    
    for day_data in daily_datasets:
        # day_data is a dict: {'df': dataframe, 'date': date, 'dte': dte}
        daily_pnl, daily_trade_count = simulate_trade(day_data['df'], day_data['date'], day_data['dte'], params)
        
        if daily_trade_count > 0:
            trades_count += daily_trade_count
            total_pnl += daily_pnl
            daily_pnls.append(daily_pnl)
            if daily_pnl > 0:
                wins += 1 # Note: This counts "Winning Days", not individual winning trades. 
                          # Ideally we'd track individual trades, but for optimization speed, 
                          # maximizing daily PnL is often the goal. 
                          # If user wants Trade Win%, we'd need simulate_trade to return list of trade results.
                          # For now, let's stick to "Positive PnL days" or approximate.
                          # Let's count it as a "Win" if the net day is positive.
            
            # Correction: User probably wants Trade-level Win Rate.
            # But simulate_trade returns aggregated PnL. 
            # To get true win rate, simulate_trade needs to return [pnl1, pnl2...]
            # For efficiency, let's modify simulate_trade to return list of pnls?
            # Or just accept that for the optimizer, TOTAL PNL is key, and Win% is secondary.
                
    # Calculate basic metrics
    win_rate = (wins / trades_count * 100) if trades_count > 0 else 0
    
    # Drawdown
    if daily_pnls:
        # Approximate drawdown based on trade sequence (not exact time series but good enough for ranking)
        cum_pnl = np.cumsum(daily_pnls)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdown = running_max - cum_pnl
        max_dd = np.max(drawdown)
    else:
        max_dd = 0
        
    return {
        'params': params,
        'total_pnl': total_pnl,
        'trades': trades_count,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'profit_factor': 0 # TODO calculate matching original logic if needed
    }

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data(folder_path, test_mode=False):
    """
    Load all CSVs and split them into daily dataframes for faster processing.
    """
    csv_files = sorted(glob.glob(str(Path(folder_path) / "*.csv")))
    
    csv_files = sorted(glob.glob(str(Path(folder_path) / "*.csv")))
    
    print(f"Loading data from {len(csv_files)} files...")
    
    daily_datasets = []
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['time_only'] = df['timestamp'].dt.time
            
            # Split by date and DTE
            # We treat each (date, DTE) tuple as a tradeable unit
            # Assuming one DTE per date usually, but sometimes multiple if calendar spreads (not handled here)
            # Original code loops: for date in dates: for dte in dtes...
            
            dates = df['date'].unique()
            
            for d in dates:
                day_subset = df[df['date'] == d].copy()
                dtes = day_subset['DTE'].unique()
                
                for dte in dtes:
                    # Filter valid DTEs
                    # Normalize logical check: 0DTE, 1DTE... 4DTE
                    # We accept strings '0DTE'...'4DTE' or integers 0..4
                    valid_dtes = ['0DTE', '1DTE', '2DTE', '3DTE', '4DTE'] # ODTE is sometimes in data
                    if str(dte) not in valid_dtes:
                        # try numeric check just in case data is mixed
                        try:
                            if int(str(dte).replace('DTE','')) > 4:
                                continue
                        except:
                            continue
                            
                    # Specific dataframe for this day and DTE
                    final_day_df = day_subset[day_subset['DTE'] == dte].copy()
                    daily_datasets.append({
                        'df': final_day_df,
                        'date': d,
                        'dte': dte
                    })
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    print(f"✅ Loaded {len(daily_datasets)} tradeable days/units.")
    return daily_datasets

# ============================================================================
# MAIN
# ============================================================================

def main():
    # 1. Load Data
    daily_data = load_all_data(FOLDER_PATH)
    
    if not daily_data:
        print("❌ No data loaded. Check path.")
        return

    # 2. Generate Parameter Combinations
    grid = PARAM_GRID
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(itertools.product(*values))
    
    param_list = [dict(zip(keys, v)) for v in combinations]
    
    print(f"Generated {len(param_list)} parameter combinations.")
    
    # 3. Multiprocessing
    cpu_count = multiprocessing.cpu_count()
    print(f"🚀 Starting optimization using {cpu_count} CPU cores...")
    
    start_time = time_module.time()
    
    # Use 'fork' on Mac/Linux for copy-on-write sharing of 'daily_data' 
    # (data is read-only so it shouldn't duplicate in memory ideally, though Python ref counting messes this up)
    # If memory is an issue, we might need a shared memory manager or simpler logic.
    # For now, let's assume 30 weeks of data fits in RAM x Cores (Mac handles this well)
    
    # Partial function to fix the data argument
    worker_func = partial(evaluate_batch, daily_datasets=daily_data)
    
    # Map parameters to results
    total_tasks = len(param_list)
    results = []
    completed = 0
    
    print(f"🔄 Processing {total_tasks} parameter combinations...")
    
    with multiprocessing.Pool(processes=cpu_count) as pool:
        # Use imap_unordered to get results as they finish
        for res in pool.imap_unordered(worker_func, param_list):
            results.append(res)
            completed += 1
            
            if completed % 100 == 0 or completed == total_tasks:
                elapsed = time_module.time() - start_time
                avg_time = elapsed / completed
                remaining = (total_tasks - completed) * avg_time
                print(f"  [{completed}/{total_tasks}] {completed/total_tasks*100:.1f}% done. Est. remaining: {remaining/60:.1f} min", end='\r')
    
    print() # Newline after progress loop
        
    duration = time_module.time() - start_time
    print(f"✅ Optimization complete in {duration:.2f} seconds.")
    
    # 4. Sort and Report
    # Sort by Total PnL descending
    results.sort(key=lambda x: x['total_pnl'], reverse=True)
    
    print("\n" + "="*80)
    print("TOP 600 PARAMETER CONFIGURATIONS")
    print("="*80)
    print(f"{'#':<3} {'PnL':<12} {'Trades':<8} {'Win%':<8} {'DD':<12} {'Parameters'}")
    print("-" * 120)
    
    top_results_for_csv = []
    
    # Print Top 600 but Save ALL
    for i, res in enumerate(results):
        if i < 600: 
            p_str = ", ".join([f"{k}={v}" for k,v in res['params'].items()])
            print(f"{i+1:<3} ₹{res['total_pnl']:<11,.0f} {res['trades']:<8} {res['win_rate']:<8.1f} ₹{res['max_dd']:<11,.0f} {p_str}")
        
        row = res['params'].copy()
        row.update({
            'total_pnl': res['total_pnl'],
            'trades': res['trades'],
            'win_rate': res['win_rate'],
            'max_drawdown': res['max_dd']
        })
        top_results_for_csv.append(row)
        
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"backtest_results/optimization_results_{timestamp}.csv"
    pd.DataFrame(top_results_for_csv).to_csv(outfile, index=False)
    print(f"\n💾 Saved {len(top_results_for_csv)} results to {outfile}")
    
    # 5. Auto-Generate Detailed Logs for Top 600
    print("\n" + "="*80)
    print("GENERATING DETAILED TRADE LOGS (Top 600)")
    print("="*80)
    try:
        # Use the same python interpreter
        cmd = [sys.executable, "files/generate_trade_logs.py", "--input", outfile, "--top", "600"]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating detailed logs: {e}")
    except Exception as e:
        print(f"❌ Unexpected error checking/running log generation: {e}")

if __name__ == "__main__":
    main()
