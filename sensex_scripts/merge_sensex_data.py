"""
=============================================================================
SENSEX Options Data Merger
=============================================================================
Merges individual weekly SENSEX expiry CSV files into a single combined
dataset, similar to how NIFTY data is merged.

Usage:
    python merge_sensex_data.py

This reads all sensex_options_YYYYMMDD.csv files from the sensex_weekly_data
directory, combines them, and saves the result.
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import glob
import sys
import time
from datetime import timedelta

try:
    from tqdm import tqdm
except ImportError:
    print("⚠  tqdm not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
    from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory containing weekly sensex expiry CSV files
INPUT_DIR = r"d:\Algotrade\AlgorithmicStockTrading\sensex_weekly_data"

# Output file for merged data
OUTPUT_FILE = r"d:\Algotrade\AlgorithmicStockTrading\sensex_options_merged.csv"

# =============================================================================
# MAIN MERGE LOGIC
# =============================================================================

def merge_sensex_data():
    """
    Merge all weekly SENSEX expiry CSV files into one combined dataset.
    """
    print("=" * 70)
    print("  SENSEX OPTIONS DATA MERGER")
    print("=" * 70)

    # Find all weekly expiry CSV files
    pattern   = os.path.join(INPUT_DIR, "sensex_options_*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"✗ No CSV files found matching: {pattern}")
        return

    print(f"\n📁 Found {len(csv_files)} weekly expiry files\n")

    all_dfs       = []
    total_records = 0
    start_time    = time.time()

    with tqdm(csv_files, desc="📂 Reading files", unit="file",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for filepath in pbar:
            filename = os.path.basename(filepath)
            filesize = os.path.getsize(filepath) / (1024 * 1024)  # MB
            pbar.set_postfix({"file": filename, "total_rows": f"{total_records:,}"})

            try:
                df       = pd.read_csv(filepath)
                records  = len(df)
                total_records += records
                all_dfs.append(df)
                tqdm.write(f"  ✓ {filename:45s} {records:>8,} records  ({filesize:.2f} MB)")
            except Exception as e:
                tqdm.write(f"  ✗ {filename:45s} ERROR: {str(e)[:80]}")

    if not all_dfs:
        print("\n✗ No data loaded. Exiting.")
        return

    # Combine all dataframes
    print(f"\n🔧 Merging {len(all_dfs)} files ({total_records:,} total rows)...")
    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Ensure proper datetime types
    print("🔧 Parsing datetime columns...")
    with tqdm(total=3, desc="  Parsing dtypes", leave=False) as pbar:
        if 'timestamp' in merged_df.columns:
            merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
        pbar.update(1)
        if 'expiry_date' in merged_df.columns:
            merged_df['expiry_date'] = pd.to_datetime(merged_df['expiry_date'])
        pbar.update(1)
        if 'date' in merged_df.columns:
            merged_df['date'] = pd.to_datetime(merged_df['date'])
        pbar.update(1)

    # Sort by timestamp and symbol
    sort_cols = []
    if 'timestamp' in merged_df.columns:
        sort_cols.append('timestamp')
    if 'symbol' in merged_df.columns:
        sort_cols.append('symbol')

    if sort_cols:
        print(f"🔧 Sorting by {sort_cols}...")
        merged_df = merged_df.sort_values(sort_cols).reset_index(drop=True)

    elapsed     = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed)))

    # Print summary
    print(f"\n{'='*70}")
    print(f"  ✅ MERGE COMPLETE")
    print(f"{'='*70}")
    print(f"  Total records:       {len(merged_df):,}")
    if 'symbol' in merged_df.columns:
        print(f"  Unique contracts:    {merged_df['symbol'].nunique():,}")
    if 'timestamp' in merged_df.columns:
        print(f"  Date range:          {merged_df['timestamp'].min()} → {merged_df['timestamp'].max()}")
    print(f"  Time taken:          {elapsed_str}")

    if 'expiry_date' in merged_df.columns:
        print(f"\n  Unique expiry dates:")
        for expiry in sorted(merged_df['expiry_date'].unique()):
            expiry_count = len(merged_df[merged_df['expiry_date'] == expiry])
            print(f"    - {pd.to_datetime(expiry).strftime('%Y-%m-%d'):12s} ({expiry_count:>8,} records)")

    # Display column info
    print(f"\n  Columns: {list(merged_df.columns)}")

    # Save merged file
    print(f"\n💾 Saving merged data to: {OUTPUT_FILE}")
    with tqdm(total=1, desc="  Writing CSV", unit="file") as pbar:
        merged_df.to_csv(OUTPUT_FILE, index=False)
        pbar.update(1)

    filesize = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"  ✓ Saved! File size: {filesize:.2f} MB")

    # Also display the backtest-relevant columns
    backtest_cols = ['expiry_date', 'DTE', 'time', 'symbol', 'open', 'high', 'low', 'close', 'index_close']
    available_backtest_cols = [c for c in backtest_cols if c in merged_df.columns]

    if available_backtest_cols:
        print(f"\n  📊 Backtest-relevant columns preview:")
        print(merged_df[available_backtest_cols].head(10).to_string(index=False))

    print(f"\n{'='*70}")

    return merged_df


if __name__ == "__main__":
    merge_sensex_data()
