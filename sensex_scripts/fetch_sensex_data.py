"""
=============================================================================
SENSEX Options Historical Data Fetcher (Groww API)
=============================================================================
Fetches SENSEX options data from Groww API, processes it with metadata 
(DTE, index prices, etc.), and saves each weekly expiry as a separate CSV file.

Usage:
    python fetch_sensex_data.py

Configuration:
    - Set your Groww API key and secret below
    - Set the output directory
    - Set the year/month range to fetch
=============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import sys

try:
    from tqdm import tqdm
except ImportError:
    print("⚠  tqdm not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
    from tqdm import tqdm

from growwapi import GrowwAPI

# =============================================================================
# CONFIGURATION - Modify these values as needed
# =============================================================================

# Groww API credentials
USER_API_KEY = "eyJraWQiOiJaTUtjVXciLCJhbGciOiJFUzI1NiJ9.eyJleHAiOjI1NDg2NzA2OTQsImlhdCI6MTc2MDI3MDY5NCwibmJmIjoxNzYwMjcwNjk0LCJzdWIiOiJ7XCJ0b2tlblJlZklkXCI6XCIyYWM5YmY5MS1jYzVhLTQ4ZWQtYmFiZi1lYTU0MGYxNGM2YTlcIixcInZlbmRvckludGVncmF0aW9uS2V5XCI6XCJlMzFmZjIzYjA4NmI0MDZjODg3NGIyZjZkODQ5NTMxM1wiLFwidXNlckFjY291bnRJZFwiOlwiY2E4OGIyMzYtOGViNS00YzkxLTk5YzQtYmQ4OTgzN2ZlZjVjXCIsXCJkZXZpY2VJZFwiOlwiMTc3M2Q1MGEtNzU2ZC01NWIxLWEyZDQtYmU4YzFhMmEzYmZmXCIsXCJzZXNzaW9uSWRcIjpcImE4NDdlMDA3LWM3YWUtNDY5Ny1hMDMxLTNkZWNmZDUyNDhlYVwiLFwiYWRkaXRpb25hbERhdGFcIjpcIno1NC9NZzltdjE2WXdmb0gvS0EwYkU0a1gxVTh4cGdYS1F4dER0SnZQT1JSTkczdTlLa2pWZDNoWjU1ZStNZERhWXBOVi9UOUxIRmtQejFFQisybTdRPT1cIixcInJvbGVcIjpcImF1dGgtdG90cFwiLFwic291cmNlSXBBZGRyZXNzXCI6XCIyNDA5OjQwZjI6MjA4Zjo0NTgxOjNjZDg6NDU1NDphZWQ5OmYwNDUsMTYyLjE1OC41MS4xNzUsMzUuMjQxLjIzLjEyM1wiLFwidHdvRmFFeHBpcnlUc1wiOjI1NDg2NzA2OTQ5MzF9IiwiaXNzIjoiYXBleC1hdXRoLXByb2QtYXBwIn0.4dniz_YUUXpb3FArha_ElYjTQRWB6xQTnfKXk23hQK4dwzRTZlHpPjlvMMc-VxxYonLzQTwAsEaGYiEyTxsbUg"
USER_SECRET = "U4isED)&@vy^Di9r&!hB7cw!65)z!yax"

# Date range to fetch (year, start_month, end_month)
# Set these to control which months/years to process
YEARS = [2025]               # List of years, e.g. [2024, 2025]
START_MONTH = 1              # Start month (1-12), set to 1 for full year
END_MONTH = 2                # End month (1-12), set to 12 for full year

# Output directory for weekly expiry CSV files
OUTPUT_DIR = r"d:\Algotrade\AlgorithmicStockTrading\sensex_weekly_data"

# Rate limiting parameters
REQUEST_DELAY = 1            # Base delay between API requests (seconds)
MAX_RETRIES = 3              # Maximum retries per failed request
BACKOFF_FACTOR = 2           # Exponential backoff multiplier

# Candle interval
CANDLE_INTERVAL = "5minute"  # 5-minute candles

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def make_api_request_with_retry(func, *args, **kwargs):
    """
    Make an API request with exponential backoff retry logic.

    Parameters:
    -----------
    func : callable
        The API function to call
    *args, **kwargs :
        Arguments to pass to the function

    Returns:
    --------
    Response from the API call
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = func(*args, **kwargs)
            time.sleep(REQUEST_DELAY)
            return response
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = REQUEST_DELAY * (BACKOFF_FACTOR ** attempt)
                tqdm.write(f"    ⚠ Request failed ({str(e)[:80]}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                tqdm.write(f"    ✗ Request failed after {MAX_RETRIES} attempts: {str(e)[:100]}")
                raise e


def get_sensex_index_data(groww, start_date, end_date):
    """
    Fetch SENSEX index (cash) historical data for merging with options data.
    All data sourced directly from Groww API — no synthetic values.

    Parameters:
    -----------
    groww : GrowwAPI
        Groww API client instance
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format

    Returns:
    --------
    pd.DataFrame
        DataFrame with SENSEX index data (date_only, sx_open, sx_high, sx_low, sx_close)
    """
    print(f"\n📊 Fetching SENSEX index data: {start_date} → {end_date}")

    all_index_data = []
    current_start = datetime.strptime(start_date, '%Y-%m-%d')
    final_end = datetime.strptime(end_date, '%Y-%m-%d')

    # Build list of chunks for progress bar
    chunks = []
    cs = current_start
    while cs <= final_end:
        ce = min(cs + timedelta(days=30), final_end)
        chunks.append((cs, ce))
        cs = ce + timedelta(days=1)

    with tqdm(chunks, desc="  Index chunks", unit="chunk",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        for chunk_start, chunk_end in pbar:
            start_time_str = chunk_start.strftime('%Y-%m-%d 09:15:00')
            end_time_str   = chunk_end.strftime('%Y-%m-%d 15:30:00')
            pbar.set_postfix({"period": f"{chunk_start.strftime('%b %Y')}"})

            try:
                index_response = make_api_request_with_retry(
                    groww.get_historical_candles,
                    exchange=groww.EXCHANGE_BSE,
                    segment=groww.SEGMENT_CASH,
                    groww_symbol="BSE-SENSEX",
                    start_time=start_time_str,
                    end_time=end_time_str,
                    candle_interval=groww.CANDLE_INTERVAL_DAY
                )

                if index_response and 'candles' in index_response and len(index_response['candles']) > 0:
                    for candle in index_response['candles']:
                        # Candle format from Groww API: [timestamp, open, high, low, close, ...]
                        # All values are real market data — no synthetic fallbacks used
                        all_index_data.append({
                            'date_only': pd.to_datetime(candle[0]).date(),
                            'sx_open':   candle[1],
                            'sx_high':   candle[2],
                            'sx_low':    candle[3],
                            'sx_close':  candle[4],
                        })
                    tqdm.write(f"  ✓ {len(index_response['candles'])} daily candles: "
                               f"{chunk_start.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')}")
                else:
                    tqdm.write(f"  ⚠ No index data for "
                               f"{chunk_start.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')}")

            except Exception as e:
                tqdm.write(f"  ✗ Error fetching index data: {str(e)[:100]}")

    if all_index_data:
        index_df = pd.DataFrame(all_index_data)
        index_df['date_only'] = pd.to_datetime(index_df['date_only'])
        index_df = (index_df
                    .drop_duplicates(subset=['date_only'])
                    .sort_values('date_only')
                    .reset_index(drop=True))
        print(f"  ✓ Total index data: {len(index_df)} trading days")
        return index_df
    else:
        print("  ✗ No SENSEX index data collected")
        return pd.DataFrame()


def fetch_sensex_options_for_expiry(groww, expiry_date):
    """
    Fetch all option contract candle data for a single SENSEX expiry date.
    Data is fetched directly from Groww API — no synthetic/dummy values added.

    Parameters:
    -----------
    groww : GrowwAPI
        Groww API client instance
    expiry_date : str
        Expiry date in 'YYYY-MM-DD' format

    Returns:
    --------
    pd.DataFrame
        DataFrame with all option candle data for this expiry
    """
    tqdm.write(f"\n  📅 Processing expiry: {expiry_date}")

    # Get all contracts for this expiry
    try:
        contracts_response = make_api_request_with_retry(
            groww.get_contracts,
            exchange=groww.EXCHANGE_BSE,
            underlying_symbol="SENSEX",
            expiry_date=expiry_date
        )
    except Exception as e:
        tqdm.write(f"    ✗ Error fetching contracts: {str(e)[:100]}")
        return pd.DataFrame()

    if not contracts_response or 'contracts' not in contracts_response:
        tqdm.write(f"    ⚠ No contracts found for {expiry_date}")
        return pd.DataFrame()

    contracts = contracts_response['contracts']
    # Filter out FUT contracts (only want CE/PE options)
    contracts = [c for c in contracts if not c.endswith('-FUT')]
    tqdm.write(f"    Found {len(contracts)} option contracts")

    # Calculate date range: 7 days before expiry to expiry day
    expiry_dt = datetime.strptime(expiry_date, '%Y-%m-%d')
    start_dt  = expiry_dt - timedelta(days=7)

    start_time = start_dt.strftime('%Y-%m-%d 09:15:00')
    end_time   = expiry_dt.strftime('%Y-%m-%d 15:30:00')

    all_candle_data = []
    success_count = 0
    skip_count    = 0
    error_count   = 0

    with tqdm(contracts, desc=f"    Contracts ({expiry_date})", unit="contract", leave=False,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for contract in pbar:
            pbar.set_postfix({"ok": success_count, "skip": skip_count, "err": error_count})
            try:
                candles_response = make_api_request_with_retry(
                    groww.get_historical_candles,
                    exchange=groww.EXCHANGE_BSE,
                    segment=groww.SEGMENT_FNO,
                    groww_symbol=contract,
                    start_time=start_time,
                    end_time=end_time,
                    candle_interval=CANDLE_INTERVAL
                )

                if candles_response and 'candles' in candles_response and len(candles_response['candles']) > 0:
                    candles = candles_response['candles']

                    # Parse contract symbol to extract metadata
                    # Format: BSE-SENSEX-DDMonYY-Strike-CE/PE
                    parts        = contract.split('-')
                    strike_price = parts[-2] if len(parts) >= 4 else ''
                    option_type  = parts[-1] if len(parts) >= 4 else ''
                    expiry_str   = parts[-3] if len(parts) >= 4 else ''

                    for candle in candles:
                        # All OHLCV values are from Groww API — no synthetic data
                        all_candle_data.append({
                            'timestamp':    candle[0],
                            'open':         candle[1],
                            'high':         candle[2],
                            'low':          candle[3],
                            'close':        candle[4],
                            'volume':       candle[5],
                            'oi':           candle[6] if len(candle) > 6 else np.nan,
                            'groww_symbol': contract,
                            'strike_price': strike_price,
                            'option_type':  option_type,
                            'expiry':       expiry_str,
                            'expiry_date':  expiry_date,
                        })

                    success_count += 1
                else:
                    skip_count += 1

            except Exception as e:
                error_count += 1
                continue

    tqdm.write(f"    ✓ Done: {success_count} contracts fetched, "
               f"{skip_count} empty, {error_count} errors")
    tqdm.write(f"    Total candles: {len(all_candle_data):,}")

    if all_candle_data:
        df = pd.DataFrame(all_candle_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    else:
        return pd.DataFrame()


def process_and_enrich_data(options_df, index_df):
    """
    Enrich options data with DTE calculation, index prices, and other metadata.
    Index prices are merged from real Groww API data — NaN is used when no
    real data is available; no synthetic values are filled in.

    Parameters:
    -----------
    options_df : pd.DataFrame
        Raw options candle data
    index_df : pd.DataFrame
        SENSEX index daily data

    Returns:
    --------
    pd.DataFrame
        Enriched DataFrame ready for saving
    """
    if options_df.empty:
        return pd.DataFrame()

    tqdm.write("\n🔧 Processing and enriching data...")

    df = options_df.copy()

    # Ensure datetime types
    df['timestamp']   = pd.to_datetime(df['timestamp'])
    df['expiry_date'] = pd.to_datetime(df['expiry_date'])

    # Rename for consistency with NIFTY data
    df = df.rename(columns={'groww_symbol': 'symbol'})

    # Extract date and time components
    df['date']       = df['timestamp']
    df['date_only']  = df['timestamp'].dt.normalize()
    df['time']       = df['timestamp'].dt.strftime('%H:%M')
    df['AM_PM']      = df['timestamp'].dt.strftime('%p')
    df['day']        = df['timestamp'].dt.strftime('%A')
    df['expiry_day'] = df['expiry_date'].dt.strftime('%A')
    df['month']      = df['timestamp'].dt.strftime('%B')
    df['month_num']  = df['timestamp'].dt.month

    # Calculate DTE (Days to Expiry) using business days
    df['dte_num'] = df.apply(
        lambda row: np.busday_count(
            row['timestamp'].date(),
            row['expiry_date'].date()
        ) if pd.notna(row['timestamp']) and pd.notna(row['expiry_date']) else np.nan,
        axis=1
    )

    df['DTE'] = df['dte_num'].apply(
        lambda x: 'ODTE' if x == 0 else f'{int(x)}DTE' if x <= 7 else '>7DTE'
    )

    # Merge with real SENSEX index data (from Groww API)
    # If no index data available, columns are left as NaN — no synthetic fill
    if not index_df.empty:
        index_df_merge = index_df.copy()
        index_df_merge['date_only'] = pd.to_datetime(index_df_merge['date_only'])

        df = df.merge(
            index_df_merge[['date_only', 'sx_open', 'sx_high', 'sx_low', 'sx_close']],
            on='date_only',
            how='left'
        )
        # Rename sx_close to index_close for compatibility with backtest engine
        df = df.rename(columns={'sx_close': 'index_close'})
        tqdm.write(f"  ✓ Merged with SENSEX index data (index_close column added)")
    else:
        df['index_close'] = np.nan
        df['sx_open']     = np.nan
        df['sx_high']     = np.nan
        df['sx_low']      = np.nan
        tqdm.write(f"  ⚠ No index data available for merge (index_close will be NaN)")

    # Sort by timestamp and symbol
    df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

    tqdm.write(f"  ✓ Total enriched records:  {len(df):,}")
    tqdm.write(f"  ✓ Unique contracts:        {df['symbol'].nunique():,}")
    tqdm.write(f"  ✓ Date range:              {df['timestamp'].min()} → {df['timestamp'].max()}")

    return df


def save_weekly_expiry_files(enriched_df, output_dir):
    """
    Save each weekly expiry as a separate CSV file.

    Parameters:
    -----------
    enriched_df : pd.DataFrame
        Enriched options data
    output_dir : str
        Directory to save CSV files

    Returns:
    --------
    list
        List of created file paths
    """
    if enriched_df.empty:
        tqdm.write("  ⚠ No data to save")
        return []

    os.makedirs(output_dir, exist_ok=True)

    created_files = []

    for expiry_date, group_df in enriched_df.groupby('expiry_date'):
        # Format: sensex_options_YYYYMMDD.csv
        filename = f"sensex_options_{expiry_date.strftime('%Y%m%d')}.csv"
        filepath = os.path.join(output_dir, filename)

        group_df.to_csv(filepath, index=False)
        created_files.append(filepath)

        tqdm.write(f"  💾 Created {filename} with {len(group_df):,} records")

    return created_files


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to orchestrate the entire SENSEX data fetching pipeline.
    All data is fetched from Groww API — no synthetic data is introduced.
    """
    print("=" * 70)
    print("  SENSEX OPTIONS HISTORICAL DATA FETCHER (Groww API)")
    print("=" * 70)
    print(f"  Years:      {YEARS}")
    print(f"  Months:     {START_MONTH} → {END_MONTH}")
    print(f"  Output:     {OUTPUT_DIR}")
    print(f"  Interval:   {CANDLE_INTERVAL}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STEP 1: Initialize Groww API
    # -------------------------------------------------------------------------
    print("\n🔐 Initializing Groww API...")
    try:
        access_token = GrowwAPI.get_access_token(api_key=USER_API_KEY, secret=USER_SECRET)
        groww = GrowwAPI(access_token)
        print("✅ Ready to Groww!\n")
    except Exception as e:
        print(f"✗ Failed to initialize Groww API: {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STEP 2: Fetch expiries for all requested year/month combinations
    # -------------------------------------------------------------------------
    all_expiries = []

    month_list = [(y, m) for y in YEARS for m in range(START_MONTH, END_MONTH + 1)]
    with tqdm(month_list, desc="📆 Fetching expiries", unit="month",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        for year, month in pbar:
            pbar.set_postfix({"period": f"{year}-{month:02d}"})
            try:
                expiries_response = make_api_request_with_retry(
                    groww.get_expiries,
                    exchange=groww.EXCHANGE_BSE,
                    underlying_symbol="SENSEX",
                    year=year,
                    month=month
                )

                if expiries_response and 'expiries' in expiries_response:
                    expiry_list = expiries_response['expiries']
                    all_expiries.extend(expiry_list)
                    tqdm.write(f"  ✓ {year}-{month:02d}: {len(expiry_list)} expiries → {expiry_list}")
                else:
                    tqdm.write(f"  ⚠ No expiries found for {year}-{month:02d}")

            except Exception as e:
                tqdm.write(f"  ✗ Error fetching expiries for {year}-{month:02d}: {str(e)[:100]}")
                continue

    # Deduplicate and sort expiries
    all_expiries = sorted(set(all_expiries))
    print(f"\n📊 Total unique expiries to process: {len(all_expiries)}")
    if all_expiries:
        print(f"   First: {all_expiries[0]}  |  Last: {all_expiries[-1]}")

    if not all_expiries:
        print("✗ No expiries found. Exiting.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STEP 3: Fetch SENSEX index data for the full date range
    # -------------------------------------------------------------------------
    earliest_expiry = datetime.strptime(all_expiries[0], '%Y-%m-%d')
    latest_expiry   = datetime.strptime(all_expiries[-1], '%Y-%m-%d')

    # Start 10 days before earliest expiry to cover the 7-day lookback
    index_start = (earliest_expiry - timedelta(days=10)).strftime('%Y-%m-%d')
    index_end   = latest_expiry.strftime('%Y-%m-%d')

    index_df = get_sensex_index_data(groww, index_start, index_end)

    # -------------------------------------------------------------------------
    # STEP 4: Fetch options data for each expiry and save weekly files
    # -------------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_records    = 0
    total_files      = 0
    skipped_expiries = []

    # Track timing for overall ETA
    pipeline_start = time.time()

    with tqdm(enumerate(all_expiries), total=len(all_expiries),
              desc="🗂  Processing expiries", unit="expiry",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:

        for idx, expiry_date in pbar:
            pbar.set_postfix({"expiry": expiry_date, "files": total_files,
                              "records": f"{total_records:,}"})

            # Check if file already exists (resume capability)
            expected_filename = f"sensex_options_{expiry_date.replace('-', '')}.csv"
            expected_filepath = os.path.join(OUTPUT_DIR, expected_filename)

            if os.path.exists(expected_filepath):
                existing_size = os.path.getsize(expected_filepath)
                if existing_size > 1000:  # Skip if file already has data (> 1KB)
                    tqdm.write(f"\n  ⏭ Skipping: {expected_filename} already exists "
                               f"({existing_size/1024:.1f} KB)")
                    total_files += 1
                    continue

            tqdm.write(f"\n{'='*60}")
            tqdm.write(f"  EXPIRY {idx+1}/{len(all_expiries)}: {expiry_date}")
            tqdm.write(f"{'='*60}")

            # Fetch options data for this expiry
            options_df = fetch_sensex_options_for_expiry(groww, expiry_date)

            if options_df.empty:
                skipped_expiries.append(expiry_date)
                tqdm.write(f"  ⚠ No data for expiry {expiry_date}")
                continue

            # Enrich data
            enriched_df = process_and_enrich_data(options_df, index_df)

            if enriched_df.empty:
                skipped_expiries.append(expiry_date)
                continue

            # Save this expiry's data as a separate file
            files = save_weekly_expiry_files(enriched_df, OUTPUT_DIR)
            total_files   += len(files)
            total_records += len(enriched_df)

    # -------------------------------------------------------------------------
    # STEP 5: Summary
    # -------------------------------------------------------------------------
    elapsed = time.time() - pipeline_start
    elapsed_str = str(timedelta(seconds=int(elapsed)))

    print(f"\n{'='*70}")
    print(f"  ✅ SENSEX DATA FETCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Total expiries processed: {len(all_expiries)}")
    print(f"  Total files created:      {total_files}")
    print(f"  Total records:            {total_records:,}")
    print(f"  Total time elapsed:       {elapsed_str}")
    print(f"  Output directory:         {OUTPUT_DIR}")

    if skipped_expiries:
        print(f"\n  ⚠ Skipped expiries (no data): {skipped_expiries}")

    print(f"\n  Files are saved as: sensex_options_YYYYMMDD.csv")
    print(f"  Each file contains one weekly expiry with all contract data.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
