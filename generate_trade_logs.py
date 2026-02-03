"""
Generate Detailed Trade Logs (Multi-Entry & Multiprocessing)
============================================================
Reads the optimization results and generates detailed trade-by-trade logs 
for the top performing parameter sets.

Features:
- Multi-Entry Logic: Re-enters market after exit (SL/Target/Index).
- Multiprocessing: Distributes work across cores for speed.

Usage:
    python generate_trade_logs.py [--top N] [--input CSV] [--cores N]
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta, date as datetime_date
from pathlib import Path
import glob
import argparse
import multiprocessing
import functools

# ============================================================================
# CONFIGURATION
# ============================================================================

FOLDER_PATH = "./options_weekly_weekwise/FY2025" 

# ============================================================================
# TRADE SIMULATION
# ============================================================================

def simulate_trade(df, date, dte, params):
    """
    Simulate trades for a specific day using Multi-Entry Logic.
    Returns a LIST of trade dictionaries (one per trade in the sequence).
    """
    day_data = df[(df['date'] == date) & (df['DTE'] == dte)].copy()
    
    if day_data.empty:
        return []
    
    trade_logs_list = []
    trade_number = 1
    
    # Parse times
    def to_time(t):
        if isinstance(t, str):
            # Handle HH:MM:SS or HH:MM
            try: return datetime.strptime(t, "%H:%M:%S").time()
            except: return datetime.strptime(t, "%H:%M").time()
        return t

    current_entry_time = to_time(params['entry_time'])
    final_exit_time = to_time(params['exit_time'])
    
    def add_minutes(t, mins):
        dummy_date = datetime_date.min
        dt = datetime.combine(dummy_date, t)
        return (dt + timedelta(minutes=mins)).time()
        
    # Validation
    if current_entry_time >= final_exit_time:
        return []

    while current_entry_time < final_exit_time:
        # ----------------------------------------------------
        # 1. FIND ENTRY
        # ----------------------------------------------------
        entry_mask = day_data['time_only'] == current_entry_time
        entry_rows = day_data[entry_mask]
        
        if entry_rows.empty:
            # If no candle at exact entry time, we can't trade this slot.
            # In a real system we might look for next candle, but backtester is strict.
            break
            
        index_at_entry = entry_rows.iloc[0]['index_close']
        if pd.isna(index_at_entry):
            break
            
        atm_strike = round(index_at_entry / 50) * 50
        pe_strike = int(atm_strike - (params['pe_offset'] * 50))
        ce_strike = int(atm_strike + (params['ce_offset'] * 50))
        
        pe_entry = entry_rows[(entry_rows['strike_price'] == pe_strike) & (entry_rows['option_type'] == 'PE')]
        ce_entry = entry_rows[(entry_rows['strike_price'] == ce_strike) & (entry_rows['option_type'] == 'CE')]
        
        if pe_entry.empty or ce_entry.empty:
            break
            
        pe_entry_price = pe_entry.iloc[0]['close']
        ce_entry_price = ce_entry.iloc[0]['close']
        total_entry_premium = pe_entry_price + ce_entry_price
        
        # ----------------------------------------------------
        # 2. MONITOR & EXIT
        # ----------------------------------------------------
        stoploss_price = total_entry_premium * (1 + params['stoploss_pct'] / 100)
        profit_target_price = total_entry_premium * (1 - params['profit_target_pct'] / 100)
        index_upper = index_at_entry + params['index_movement']
        index_lower = index_at_entry - params['index_movement']
        
        # Filter data from entry onwards
        monitoring_mask = (day_data['time_only'] >= current_entry_time) & (day_data['time_only'] <= final_exit_time)
        monitoring_data = day_data[monitoring_mask]
        
        if monitoring_data.empty:
            break
            
        pe_data = monitoring_data[(monitoring_data['strike_price'] == pe_strike) & (monitoring_data['option_type'] == 'PE')]
        ce_data = monitoring_data[(monitoring_data['strike_price'] == ce_strike) & (monitoring_data['option_type'] == 'CE')]
        
        if pe_data.empty or ce_data.empty:
            break
            
        merged = pd.merge(
            pe_data[['timestamp', 'time_only', 'close', 'index_close']], 
            ce_data[['timestamp', 'close']], 
            on='timestamp', 
            suffixes=('_pe', '_ce'),
            how='inner'
        )
        merged['total_premium'] = merged['close_pe'] + merged['close_ce']
        merged.sort_values('timestamp', inplace=True)
        
        # Skip the entry candle itself for exit checking (unless immediate SL?)
        # Standard: Check exits from next candle or current? 
        # Optimize_backtest uses strict next candle: merged['time_only'] > current_entry_time
        merged = merged[merged['time_only'] > current_entry_time]
        
        if merged.empty:
            break
            
        sl_hit = merged['total_premium'] >= stoploss_price
        pt_hit = merged['total_premium'] <= profit_target_price
        idx_breach = (merged['index_close'] >= index_upper) | (merged['index_close'] <= index_lower)
        time_exit = merged['time_only'] >= final_exit_time
        
        exit_mask = sl_hit | pt_hit | idx_breach | time_exit
        
        exit_reason = "EOD"
        if exit_mask.any():
            first_idx = exit_mask.idxmax()
            exit_row = merged.loc[first_idx]
            
            if sl_hit[first_idx]: exit_reason = "Stoploss"
            elif pt_hit[first_idx]: exit_reason = "Target"
            elif idx_breach[first_idx]: exit_reason = "Index Breach"
            elif time_exit[first_idx]: exit_reason = "Time Exit"
        else:
            # Reached end of data without triggers
            exit_row = merged.iloc[-1]
            # If last row time >= exit time, it's Time Exit, else EOD from data end
            if exit_row['time_only'] >= final_exit_time:
                exit_reason = "Time Exit"
            else:
                exit_reason = "EOD (Data End)"
            
        exit_premium = exit_row['total_premium']
        trade_exit_time = exit_row['time_only']
        pnl = (total_entry_premium - exit_premium) * params['lot_size'] * 50
        
        # ----------------------------------------------------
        # 3. LOG TRADE
        # ----------------------------------------------------
        trade_log = {
            'param_set_rank': params.get('rank', 0),
            'trade_no': trade_number,
            'date': date,
            'dte': dte,
            'entry_time': current_entry_time,
            'exit_time': trade_exit_time,
            'exit_reason': exit_reason,
            'index_at_entry': index_at_entry,
            'pe_strike': pe_strike,
            'ce_strike': ce_strike,
            'pe_entry_price': pe_entry_price,
            'ce_entry_price': ce_entry_price,
            'total_entry_premium': total_entry_premium,
            'exit_premium': exit_premium,
            'pnl': pnl,
            'stoploss_price': stoploss_price,
            'target_price': profit_target_price
        }
        trade_logs_list.append(trade_log)
        
        trade_number += 1
        # Re-entry after 5 minutes
        current_entry_time = add_minutes(trade_exit_time, 0)

    return trade_logs_list

# ============================================================================
# MULTIPROCESSING WORKER
# ============================================================================

def process_file_chunk(filepath, top_results_df):
    """
    Process a single weekly file against ALL top strategies.
    Returns a list of trade dicts.
    """
    filename = Path(filepath).name
    # print(f"Processing {filename}...") # Noise in multiprocessing
    
    file_trades = []
    try:
        # Load Data
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['time_only'] = df['timestamp'].dt.time
        
        dates = sorted(df['date'].unique())
        dtes = sorted(df['DTE'].unique(), reverse=True) if 'DTE' in df.columns else []
        
        # Iterate Strategies
        for _, param_row in top_results_df.iterrows():
            params = param_row.to_dict()
            rank = params.get('rank') # Helper added in main
            
            for date in dates:
                for dte in dtes:
                    # Filter DTE
                    valid_dtes = ['0DTE', '1DTE', '2DTE', '3DTE', '4DTE', 'ODTE']
                    if str(dte) not in valid_dtes:
                         try:
                            if int(str(dte).replace('DTE','').replace('O','0')) > 4: continue
                         except: continue
                    
                    trades = simulate_trade(df, date, dte, params)
                    if trades:
                        for t in trades:
                            t['filename'] = filename
                            file_trades.append(t)
                            
    except Exception as e:
        print(f"❌ Error in {filename}: {e}")
        
    return file_trades

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=int, default=0, help='Number of top results to process')
    parser.add_argument('--start_rank', type=int, default=1)
    parser.add_argument('--end_rank', type=int, default=1)
    parser.add_argument('--input', type=str, help='Path to optimization CSV')
    parser.add_argument('--cores', type=int, default=0, help='Number of cores (0=auto)')
    args = parser.parse_args()
    
    # 1. Determine Rank Range
    start_rank = args.start_rank
    end_rank = args.end_rank
    if args.top > 0:
        start_rank = 1
        end_rank = args.top
        
    print(f"🎯 Generating Logs for Ranks: {start_rank} to {end_rank}")
    
    # 2. Load Optimization Results
    if args.input:
        opt_file = args.input
    else:
        files = glob.glob("backtest_results/optimization_results_*.csv")
        if not files:
            print("❌ No results found.")
            return
        opt_file = max(files, key=lambda f: Path(f).stat().st_mtime)
        
    print(f"📂 Loading Params from: {opt_file}")
    opt_df = pd.read_csv(opt_file)
    opt_df = opt_df.sort_values('total_pnl', ascending=False).reset_index(drop=True)
    
    # Slice
    start_idx = start_rank - 1
    top_results = opt_df.iloc[start_idx:end_rank].copy()
    
    # Add Rank column explicitly
    top_results['rank'] = range(start_rank, end_rank + 1)
    
    print(f"📊 Strategies to process: {len(top_results)}")
    
    # 3. Find Data Files
    csv_files = sorted(glob.glob(str(Path(FOLDER_PATH) / "*.csv")))
    print(f"📂 Data Files: {len(csv_files)}")
    
    # 4. Prepare Multiprocessing
    num_cores = args.cores if args.cores > 0 else multiprocessing.cpu_count() - 1
    num_cores = max(1, num_cores)
    print(f"🚀 Starting Multiprocessing on {num_cores} cores...")
    
    pool = multiprocessing.Pool(processes=num_cores)
    
    # Partial function to pass the strategy dataframe to all workers
    worker = functools.partial(process_file_chunk, top_results_df=top_results)
    
    # Map workers to files
    all_trade_batches = pool.map(worker, csv_files)
    
    pool.close()
    pool.join()
    
    # Flatten results
    all_detailed_trades = [t for batch in all_trade_batches for t in batch]
    print(f"\n✅ Generation Complete! Total Trades: {len(all_detailed_trades)}")
    
    if not all_detailed_trades:
        print("❌ No trades generated.")
        return

    # 5. Save Results
    results_df = pd.DataFrame(all_detailed_trades)
    output_dir = Path("./backtest_results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Unique Params
    unique_ranks = results_df['param_set_rank'].unique()
    print(f"💾 Saving {len(unique_ranks)} log files...")
    
    for rank in unique_ranks:
        rank_df = results_df[results_df['param_set_rank'] == rank]
        filename = f"detailed_trades_rank_{rank}_{timestamp}.csv"
        path = output_dir / filename
        rank_df.to_csv(path, index=False)
        
    print("🎉 All log files saved.")

if __name__ == "__main__":
    main()
