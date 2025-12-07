import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import io
import base64
import os
from pathlib import Path
import json
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL STATE MANAGEMENT
# ============================================================================

class BacktestState:
    """Centralized state management for the backtest system"""
    def __init__(self):
        self.uploaded_files = {}
        self.processed_data = {}
        self.strategy_configs = {}
        self.all_results = {}
        self.consolidated_results = None
        self.folder_path = None
        self.entry_time_override = None
        self.exit_dte_override = None
        self.exit_time_override = None
        self.strategy_mode = 'Weekly'
        
    def reset(self):
        self.__init__()
        
    def add_file(self, filename, content):
        self.uploaded_files[filename] = content
        
    def get_file_count(self):
        return len(self.uploaded_files)
    
    def get_filenames(self):
        return list(self.uploaded_files.keys())
    
    def add_strategy_config(self, strategy_name, config):
        self.strategy_configs[strategy_name] = config
    
    def get_strategy_names(self):
        return list(self.strategy_configs.keys())

state = BacktestState()

# ============================================================================
# STRATEGY PRESETS
# ============================================================================

STRATEGY_PRESETS = {
    "Conservative": {
        "description": "Lower risk with tighter stops and wider strikes",
        "stoploss_pct": [80, 80, 80, 80, 80],
        "profit_target_pct": [40, 40, 40, 40, 40],
        "pe_strike_offset": [7, 7, 6, 6, 5],
        "ce_strike_offset": [7, 7, 6, 6, 5],
        "index_movement": [70, 75, 80, 85, 90],
        "lot_size": [1, 1, 1, 1, 1]
    },
    "Moderate": {
        "description": "Balanced approach with standard parameters",
        "stoploss_pct": [60, 60, 60, 65, 65],
        "profit_target_pct": [50, 50, 50, 45, 45],
        "pe_strike_offset": [4, 4, 4, 3, 3],
        "ce_strike_offset": [4, 4, 4, 3, 3],
        "index_movement": [100, 110, 115, 120, 125],
        "lot_size": [1, 1, 1, 1, 1]
    },
    "Aggressive": {
        "description": "Higher risk with wider stops and tighter strikes",
        "stoploss_pct": [75, 75, 80, 80, 85],
        "profit_target_pct": [45, 45, 40, 40, 35],
        "pe_strike_offset": [4, 4, 3, 3, 2],
        "ce_strike_offset": [4, 4, 3, 3, 2],
        "index_movement": [120, 125, 130, 135, 140],
        "lot_size": [1, 1, 1, 1, 1]
    },
    "Scalper": {
        "description": "Quick profits with tight targets and stops",
        "stoploss_pct": [30, 30, 35, 35, 40],
        "profit_target_pct": [35, 35, 30, 30, 25],
        "pe_strike_offset": [6, 6, 5, 5, 4],
        "ce_strike_offset": [6, 6, 5, 5, 4],
        "index_movement": [50, 55, 60, 65, 70],
        "lot_size": [1, 1, 1, 1, 1]
    },
    "Custom": {
        "description": "User-defined parameters",
        "stoploss_pct": [60, 60, 60, 50, 50],
        "profit_target_pct": [50, 50, 50, 50, 50],
        "pe_strike_offset": [4, 4, 4, 4, 4],
        "ce_strike_offset": [4, 4, 4, 4, 4],
        "index_movement": [100, 100, 200, 150, 100],
        "lot_size": [1, 1, 1, 1, 1]
    }
}

DTE_LABELS = ['4 DTE', '3 DTE', '2 DTE', '1 DTE', '0 DTE']

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_data(df):
    """Enhanced data preprocessing"""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    
    required_cols = ['expiry_date', 'date_only', 'time', 'symbol', 'open', 'high', 'low', 'close', 'index_close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df['expiry_date'] = pd.to_datetime(df['expiry_date']).dt.date
    df['date_only'] = pd.to_datetime(df['date_only']).dt.date
    df['days_to_expiry'] = (pd.to_datetime(df['expiry_date']) - pd.to_datetime(df['date_only'])).dt.days
    df = df[df['days_to_expiry'] <= 6].copy()
    
    cols = ['expiry_date', "date_only", 'days_to_expiry', 'time', 'symbol', 'open', 'high', 'low', 'close', 'index_close']
    if 'DTE' in df.columns:
        cols.append('DTE')
    
    backtest_data = df[cols].copy()
    
    if 'DTE' in backtest_data.columns:
        def extract_dte_format(dte_str):
            if pd.isna(dte_str):
                return None
            dte_str = str(dte_str).strip().upper()
            if dte_str == 'ODTE' or dte_str == '0DTE':
                return '0 DTE'
            import re
            match = re.search(r'(\d+)', dte_str)
            if match:
                return f'{match.group(1)} DTE'
            return None
        
        backtest_data['dte_format'] = backtest_data['DTE'].apply(extract_dte_format)
        if backtest_data['dte_format'].isna().all() or backtest_data['dte_format'].isna().any():
            backtest_data['dte_format'] = backtest_data['days_to_expiry'].apply(lambda x: f'{x} DTE')
    else:
        backtest_data['dte_format'] = backtest_data['days_to_expiry'].apply(lambda x: f'{x} DTE')
    
    return backtest_data

# ============================================================================
# FILE HANDLING
# ============================================================================

def load_files_from_folder(folder_path):
    """Load all CSV and Excel files from a folder"""
    loaded_files = {}
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder path does not exist: {folder_path}")
    
    supported_extensions = ['.csv', '.xlsx', '.xls']
    files = [f for f in folder.iterdir() if f.suffix.lower() in supported_extensions]
    
    if not files:
        raise ValueError(f"No CSV or Excel files found in: {folder_path}")
    
    print(f"📂 Found {len(files)} files in folder")
    
    for file_path in files:
        try:
            filename = file_path.name
            print(f"   Loading: {filename}...", end=" ")
            
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            loaded_files[filename] = df
            print(f"✅ ({len(df)} rows)")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    return loaded_files

# ============================================================================
# STRATEGY CONFIGURATION (MODIFIED)
# ============================================================================

def create_base_strategy_config(backtest_data, force_standard_dtes=True):
    """Create base strategy configuration"""
    if force_standard_dtes:
        standard_dtes = ['4 DTE', '3 DTE', '2 DTE', '1 DTE', '0 DTE']
        base_config = pd.DataFrame({
            'DTE': standard_dtes,
            'Stoploss %': [60, 60, 60, 50, 50],
            'Profit Target %': [50, 50, 50, 50, 50],
            'lot_size': [1, 1, 1, 1, 1],
            'PE Strike': [4, 4, 4, 4, 4],
            'CE Strike': [4, 4, 4, 4, 4],
            'Index Movement +/-': [100, 100, 200, 150, 100]
        })
        
        def get_index_close_at_915(dte):
            filtered = backtest_data[
                (backtest_data['dte_format'] == dte) & (backtest_data['time'] == '09:15')
            ]
            return filtered['index_close'].iloc[0] if not filtered.empty else 23000.0
        
        base_config['Index Close'] = base_config['DTE'].apply(get_index_close_at_915)
        base_config['Index Close'] = (base_config['Index Close'] / 50).round() * 50
    else:
        unique_dte = sorted(backtest_data['dte_format'].dropna().unique(), reverse=True)
        base_config = pd.DataFrame({
            'DTE': unique_dte,
            'Stoploss %': [60]*len(unique_dte),
            'Profit Target %': [50]*len(unique_dte),
            'lot_size': [1]*len(unique_dte),
            'PE Strike': [4]*len(unique_dte),
            'CE Strike': [4]*len(unique_dte),
            'Index Movement +/-': [100]*len(unique_dte)
        })
        
        def get_index_close_at_915(dte):
            filtered = backtest_data[
                (backtest_data['dte_format'] == dte) & (backtest_data['time'] == '09:15')
            ]
            return filtered['index_close'].iloc[0] if not filtered.empty else np.nan
        
        base_config['Index Close'] = base_config['DTE'].apply(get_index_close_at_915)
        base_config['Index Close'] = (base_config['Index Close'] / 50).round() * 50
    
    return base_config

def apply_preset_to_config(base_config, preset_params):
    """Apply preset parameters to configuration"""
    modified_config = base_config.copy()
    num_dtes = len(modified_config)
    
    def extend_to_length(arr, length):
        if len(arr) >= length:
            return arr[:length]
        return arr + [arr[-1]] * (length - len(arr))
    
    modified_config['Stoploss %'] = extend_to_length(preset_params['stoploss_pct'], num_dtes)
    modified_config['Profit Target %'] = extend_to_length(preset_params['profit_target_pct'], num_dtes)
    modified_config['PE Strike'] = extend_to_length(preset_params['pe_strike_offset'], num_dtes)
    modified_config['CE Strike'] = extend_to_length(preset_params['ce_strike_offset'], num_dtes)
    modified_config['Index Movement +/-'] = extend_to_length(preset_params['index_movement'], num_dtes)
    modified_config['lot_size'] = extend_to_length(preset_params['lot_size'], num_dtes)
    
    return modified_config

def generate_strategy_combinations(base_config, selected_presets_with_params):
    """Generate strategy configurations for selected presets"""
    strategies = {}
    for strategy_name, preset_params in selected_presets_with_params.items():
        config = apply_preset_to_config(base_config, preset_params)
        strategies[strategy_name] = config
    return strategies

def parse_strategy_excel(file_content, base_config):
    """
    Parse strategy parameters from uploaded Excel file.
    Expected columns: Strategy, DTE, Stoploss %, Profit Target %, lot_size, PE Strike, CE Strike, Index Movement +/-
    """
    try:
        df = pd.read_excel(io.BytesIO(file_content))
        required_cols = ['Strategy', 'DTE', 'Stoploss %', 'Profit Target %', 'lot_size', 'PE Strike', 'CE Strike', 'Index Movement +/-']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Normalize DTE format
        def normalize_dte(val):
            val = str(val).upper().strip()
            if 'DTE' not in val:
                val = f"{val} DTE"
            return val
        
        df['DTE'] = df['DTE'].apply(normalize_dte)
        
        # Group by Strategy
        strategies = {}
        grouped = df.groupby('Strategy')
        
        for strategy_name, group in grouped:
            # Merge with base config structure to ensure DTEs exist
            # We take base_config structure (DTEs) and update values from group
            config = base_config[['DTE', 'Index Close']].copy()
            
            # Merge uploaded params
            merged = pd.merge(config, group, on='DTE', how='left')
            
            # Fill NaNs with defaults if any DTE is missing in the upload
            defaults = {
                'Stoploss %': 60, 'Profit Target %': 50, 'lot_size': 1,
                'PE Strike': 4, 'CE Strike': 4, 'Index Movement +/-': 100
            }
            for col, val in defaults.items():
                if col in merged.columns:
                    merged[col] = merged[col].fillna(val)
            
            # Keep only relevant columns
            final_cols = ['DTE', 'Stoploss %', 'Profit Target %', 'lot_size', 'PE Strike', 'CE Strike', 'Index Movement +/-', 'Index Close']
            strategies[strategy_name] = merged[final_cols]
            
        return strategies
            
    except Exception as e:
        raise ValueError(f"Error parsing Strategy Excel: {str(e)}")

# ============================================================================
# HELPER: Get DTE number from string
# ============================================================================

def get_dte_number(dte_string):
    """Extract numeric DTE from string like '4 DTE' -> 4"""
    import re
    match = re.search(r'(\d+)', dte_string)
    return int(match.group(1)) if match else 0

# ============================================================================
# INTRADAY STRATEGY EXECUTION (MODIFIED)
# ============================================================================

def execute_intraday_strategy(strategy_row, backtest_data, strategy_config, filename, strategy_name,
                               entry_exit_times, entry_dte, exit_dte):
    """
    Execute INTRADAY strategy with per-DTE entry/exit times.
    entry_exit_times: dict like {'4 DTE': {'entry': '09:15', 'exit': '15:20'}, ...}
    """
    all_trades = []
    trade_number = 1
    
    entry_dte_num = get_dte_number(entry_dte)
    # If exit_dte is 'Default (Expiry)', treat as 0
    exit_dte_num = 0 if exit_dte in ['At Expiry', '0 DTE', 'Default (Expiry)'] else get_dte_number(exit_dte)
    
    if entry_dte_num < exit_dte_num:
        print(f"⚠️  Entry DTE ({entry_dte_num}) must be >= Exit DTE ({exit_dte_num})")
        return all_trades
    
    for current_dte_num in range(entry_dte_num, exit_dte_num - 1, -1):
        current_dte = f'{current_dte_num} DTE'
        
        # Get times for this DTE
        dte_times = entry_exit_times.get(current_dte)
        if not dte_times:
            # Skip if no config for this DTE (though UI should prevent this or provide defaults)
            continue
            
        entry_time = dte_times.get('entry', '09:15')
        exit_time = dte_times.get('exit', '15:20')
        
        matching_strategy = strategy_config[strategy_config['DTE'] == current_dte]
        if matching_strategy.empty:
            continue
        
        current_strategy = matching_strategy.iloc[0]
        pe_strike_offset = current_strategy['PE Strike']
        ce_strike_offset = current_strategy['CE Strike']
        lot_size = current_strategy['lot_size']
        stoploss_pct = current_strategy['Stoploss %']
        profit_target_pct = current_strategy['Profit Target %']
        index_movement = current_strategy['Index Movement +/-']
        
        dte_data = backtest_data[backtest_data['dte_format'] == current_dte].copy()
        if dte_data.empty:
            continue
            
        unique_dates = sorted(dte_data['date_only'].unique())
        
        for trade_date in unique_dates:
            date_data = dte_data[dte_data['date_only'] == trade_date].copy()
            entry_data = date_data[date_data['time'] == entry_time]
            
            if entry_data.empty:
                continue
            
            index_close_at_entry = entry_data.iloc[0]['index_close']
            index_close_for_entry = round(index_close_at_entry / 50) * 50
            
            pe_strike = index_close_for_entry - (pe_strike_offset * 50)
            ce_strike = index_close_for_entry + (ce_strike_offset * 50)
            
            pe_symbol = f"{int(pe_strike)}-PE"
            ce_symbol = f"{int(ce_strike)}-CE"
            
            pe_entry = entry_data[entry_data['symbol'].str.contains(pe_symbol, na=False)]
            ce_entry = entry_data[entry_data['symbol'].str.contains(ce_symbol, na=False)]
            
            if pe_entry.empty or ce_entry.empty:
                continue
            
            pe_entry_price = pe_entry.iloc[0]['close']
            ce_entry_price = ce_entry.iloc[0]['close']
            total_entry_price = pe_entry_price + ce_entry_price
            entry_index_close = pe_entry.iloc[0]['index_close']
            
            stoploss_price = total_entry_price * (1 + stoploss_pct / 100)
            profit_target_price = total_entry_price * (1 - profit_target_pct / 100)
            index_upper_limit = entry_index_close + index_movement
            index_lower_limit = entry_index_close - index_movement
            
            monitoring_pe = date_data[(date_data['symbol'].str.contains(pe_symbol, na=False)) & (date_data['time'] >= entry_time) & (date_data['time'] <= exit_time)].copy()
            monitoring_ce = date_data[(date_data['symbol'].str.contains(ce_symbol, na=False)) & (date_data['time'] >= entry_time) & (date_data['time'] <= exit_time)].copy()
            
            if monitoring_pe.empty or monitoring_ce.empty:
                continue
            
            monitoring_data = pd.merge(
                monitoring_pe[['date_only', 'time', 'close', 'index_close']],
                monitoring_ce[['date_only', 'time', 'close']],
                on=['date_only', 'time'], suffixes=('_pe', '_ce')
            )
            
            if monitoring_data.empty:
                continue
            
            monitoring_data['total_close'] = monitoring_data['close_pe'] + monitoring_data['close_ce']
            monitoring_data = monitoring_data.sort_values('time')
            
            exit_reason = None
            exit_price = None
            pe_exit_price = None
            ce_exit_price = None
            
            exit_time_actual = None
            exit_index_close = None
            
            for idx, row in monitoring_data.iterrows():
                if row['time'] == entry_time:
                    continue
                
                check = False
                if row['total_close'] >= stoploss_price:
                    exit_reason = 'Stoploss Hit'
                    check = True
                elif row['total_close'] <= profit_target_price:
                    exit_reason = 'Profit Target Hit'
                    check = True
                elif row['index_close'] >= index_upper_limit or row['index_close'] <= index_lower_limit:
                    exit_reason = 'Index Movement Breach'
                    check = True
                elif row['time'] >= exit_time:
                    exit_reason = 'Time-Based Exit'
                    check = True
                
                if check:
                    exit_price = row['total_close']
                    pe_exit_price = row['close_pe']
                    ce_exit_price = row['close_ce']
                    exit_time_actual = row['time']
                    exit_index_close = row['index_close']
                    break
            
            if exit_reason is None:
                exit_row = monitoring_data[monitoring_data['time'] == exit_time]
                if not exit_row.empty:
                    exit_reason = 'Time-Based Exit'
                    exit_price = exit_row.iloc[0]['total_close']
                    pe_exit_price = exit_row.iloc[0]['close_pe']
                    ce_exit_price = exit_row.iloc[0]['close_ce']
                    exit_time_actual = exit_time
                    exit_index_close = exit_row.iloc[0]['index_close']
                else:
                    last_row = monitoring_data.iloc[-1]
                    exit_reason = 'End of Day Exit'
                    exit_price = last_row['total_close']
                    pe_exit_price = last_row['close_pe']
                    ce_exit_price = last_row['close_ce']
                    exit_time_actual = last_row['time']
                    exit_index_close = last_row['index_close']
            
            pnl = (total_entry_price - exit_price) * lot_size * 50
            
            all_trades.append({
                'Strategy': strategy_name, 'Mode': 'Intraday', 'Filename': filename,
                'Trade_Number': trade_number, 'Entry_DTE': current_dte, 'Exit_DTE': current_dte,
                'Entry_Date': trade_date, 'Entry_Time': entry_time, 'Exit_Date': trade_date,
                'Exit_Time': exit_time_actual, 'PE_Strike': pe_strike, 'CE_Strike': ce_strike,
                'PE_Entry_Price': pe_entry_price, 'CE_Entry_Price': ce_entry_price, 'Total_Entry_Price': total_entry_price,
                'PE_Exit_Price': pe_exit_price, 'CE_Exit_Price': ce_exit_price, 'Total_Exit_Price': exit_price,
                'Exit_Price': exit_price, 'Exit_Reason': exit_reason, 'PnL': pnl,
                'Entry_Index_Close': entry_index_close, 'Entry_Index_Close_at_Entry': index_close_for_entry,
                'Exit_Index_Close': exit_index_close
            })
            trade_number += 1
            
    return all_trades

# ============================================================================
# INTERDAY STRATEGY EXECUTION (MODIFIED)
# ============================================================================

def execute_interday_strategy(strategy_row, backtest_data, strategy_config, filename, strategy_name,
                               entry_dte=None, entry_time='09:15', exit_dte=None, exit_time='15:20'):
    """
    Execute WEEKLY (formerly Interday) strategy with continuous Re-entry loop.
    Enters at entry_dte @ entry_time.
    If trade exits, re-enters immediately (next candle) until exit_dte @ exit_time.
    """
    all_trades = []
    all_dtes = sorted(backtest_data['dte_format'].dropna().unique(), reverse=True)
    
    # Resolve DTEs
    if entry_dte is None: entry_dte = strategy_row['DTE']
    if exit_dte is None or exit_dte == 'At Expiry': exit_dte = '0 DTE'
    
    try:
        start_index = all_dtes.index(entry_dte)
        end_index = all_dtes.index(exit_dte)
    except ValueError:
        print(f"⚠️ DTEs {entry_dte} or {exit_dte} not found in data for {filename}")
        return all_trades

    current_dte_index = start_index
    current_time_cursor = entry_time
    last_exit_index_close = None
    continue_trading = True
    trade_number = 1
    
    just_exited = False
    
    # Loop through days from Entry DTE to Exit DTE
    while continue_trading and current_dte_index <= end_index:
        current_dte = all_dtes[current_dte_index]
        
        # Check Global Exit Condition (if we started on the exit day already past time)
        if current_dte == exit_dte:
             # This check is more relevant inside the candle loop, but good to have
             pass

        matching_strategy = strategy_config[strategy_config['DTE'] == current_dte]
        if matching_strategy.empty:
            # Fallback to configured strategy row if specific DTE config missing? 
            # Or skip? Let's use the passed strategy_row as fallback or skip.
            # ideally strategy_config has all DTEs.
            if strategy_row['DTE'] == current_dte:
                entry_strategy = strategy_row
            else:
                entry_strategy = strategy_row # Fallback to initial config if exact DTE missing
        else:
             entry_strategy = matching_strategy.iloc[0]

        pe_strike_offset = entry_strategy['PE Strike']
        ce_strike_offset = entry_strategy['CE Strike']
        lot_size = entry_strategy['lot_size']
        stoploss_pct = entry_strategy['Stoploss %']
        profit_target_pct = entry_strategy['Profit Target %']
        index_movement = entry_strategy['Index Movement +/-']

        dte_data = backtest_data[backtest_data['dte_format'] == current_dte].copy()
        if dte_data.empty:
            current_dte_index += 1
            current_time_cursor = '09:15' # Reset time for next day
            just_exited = False
            continue
            
        # Filter for data >= current_time_cursor
        # We need to sort by time to find the first available candle
        dte_data = dte_data.sort_values(['date_only', 'time'])
        
        # 1. FIND ENTRY
        # We find the first candle at or after current_time_cursor
        # Even if just exited, we allow immediate re-entry on the same candle (user request)
        candidate_data = dte_data[dte_data['time'] >= current_time_cursor]
        
        if candidate_data.empty:
            # No more data today, move to next DTE
            current_dte_index += 1
            current_time_cursor = '09:15'
            just_exited = False
            continue
            
        # We take the first available timestamp as entry
        entry_row = candidate_data.iloc[0]
        actual_entry_time = entry_row['time']
        actual_entry_date = entry_row['date_only']
        
        # Reset flag
        just_exited = False
        
        # Check Global Exit before entering
        if current_dte == exit_dte and actual_entry_time >= exit_time:
            continue_trading = False
            break
            
        # Determine ATM Strike
        index_close_for_entry = entry_row['index_close']
        index_close_for_entry = round(index_close_for_entry / 50) * 50
        
        pe_strike = index_close_for_entry - (pe_strike_offset * 50)
        ce_strike = index_close_for_entry + (ce_strike_offset * 50)
        
        pe_symbol = f"{int(pe_strike)}-PE"
        ce_symbol = f"{int(ce_strike)}-CE"
        
        # Get Entry Prices
        pe_entry_row = candidate_data[(candidate_data['time'] == actual_entry_time) & (candidate_data['symbol'].str.contains(pe_symbol, na=False))]
        ce_entry_row = candidate_data[(candidate_data['time'] == actual_entry_time) & (candidate_data['symbol'].str.contains(ce_symbol, na=False))]
        
        if pe_entry_row.empty or ce_entry_row.empty:
            # Strikes missing? Advance time slightly?
            # Creating a "next tick" logic is complex with just string times.
            # We'll try to find the next valid timestamp in candidate_data
            unique_times = candidate_data['time'].unique()
            valid_found = False
            for t in unique_times:
                pe_check = candidate_data[(candidate_data['time'] == t) & (candidate_data['symbol'].str.contains(pe_symbol, na=False))]
                ce_check = candidate_data[(candidate_data['time'] == t) & (candidate_data['symbol'].str.contains(ce_symbol, na=False))]
                if not pe_check.empty and not ce_check.empty:
                    actual_entry_time = t
                    pe_entry_row = pe_check
                    ce_entry_row = ce_check
                    valid_found = True
                    break
            
            if not valid_found:
                current_dte_index += 1
                current_time_cursor = '09:15'
                continue

        pe_entry_price = pe_entry_row.iloc[0]['close']
        ce_entry_price = ce_entry_row.iloc[0]['close']
        total_entry_price = pe_entry_price + ce_entry_price
        entry_index_close = pe_entry_row.iloc[0]['index_close']
        
        # 2. MONITOR TRADE
        stoploss_price = total_entry_price * (1 + stoploss_pct / 100)
        profit_target_price = total_entry_price * (1 - profit_target_pct / 100)
        index_upper_limit = entry_index_close + index_movement
        index_lower_limit = entry_index_close - index_movement
        
        # Monitor from *next* candle onwards, possibly spanning multiple days/DTEs
        # We need a monitoring loop that goes through DTEs starting from current
        
        trade_exited = False
        exit_reason = None
        
        # Iterate through remaining data for this cycle
        # We need to stitch data from current_dte to exit_dte
        
        # Optimization: Build a monitoring dataset
        # This might be heavy if lots of days. But usually it's just 5 days.
        
        for mon_dte_idx in range(current_dte_index, end_index + 1):
            mon_dte = all_dtes[mon_dte_idx]
            mon_data = backtest_data[backtest_data['dte_format'] == mon_dte].copy()
            if mon_data.empty: continue
            
            # Filter times: if first day, time > actual_entry_time
            if mon_dte_idx == current_dte_index:
                mon_data = mon_data[mon_data['time'] > actual_entry_time]
            
            if mon_data.empty: continue
            
            # Sort
            mon_data = mon_data.sort_values(['date_only', 'time'])
            
            # Get PE/CE data
            pe_mon = mon_data[mon_data['symbol'].str.contains(pe_symbol, na=False)]
            ce_mon = mon_data[mon_data['symbol'].str.contains(ce_symbol, na=False)]
            
            if pe_mon.empty or ce_mon.empty: continue
            
            full_mon = pd.merge(pe_mon[['date_only', 'time', 'close', 'index_close']],
                                ce_mon[['date_only', 'time', 'close']],
                                on=['date_only', 'time'], suffixes=('_pe', '_ce'))
            
            full_mon['total_close'] = full_mon['close_pe'] + full_mon['close_ce']
            full_mon = full_mon.sort_values(['date_only', 'time'])
            
            # Scan candles
            for idx, row in full_mon.iterrows():
                # Check Time-Based Global Exit
                if mon_dte == exit_dte and row['time'] >= exit_time:
                    exit_reason = 'Global Exit Time Reached'
                    # Force exit here
                    trade_exited = True
                    continue_trading = False # Stop outer loop
                
                # Check Strategy Exits
                elif row['total_close'] >= stoploss_price:
                    exit_reason = 'Stoploss Hit'
                elif row['total_close'] <= profit_target_price:
                    exit_reason = 'Profit Target Hit'
                elif row['index_close'] >= index_upper_limit or row['index_close'] <= index_lower_limit:
                    exit_reason = 'Index Movement Breach'
                
                if exit_reason:
                    # Execute Exit
                    pnl = (total_entry_price - row['total_close']) * lot_size * 50
                    all_trades.append({
                        'Strategy': strategy_name, 'Mode': 'Weekly', 'Filename': filename,
                        'Trade_Number': trade_number, 'Entry_DTE': current_dte, 'Exit_DTE': mon_dte,
                        'Entry_Date': actual_entry_date, 'Entry_Time': actual_entry_time, 
                        'Exit_Date': row['date_only'], 'Exit_Time': row['time'], 
                        'PE_Strike': pe_strike, 'CE_Strike': ce_strike,
                        'PE_Entry_Price': pe_entry_price, 'CE_Entry_Price': ce_entry_price, 'Total_Entry_Price': total_entry_price,
                        'PE_Exit_Price': row['close_pe'], 'CE_Exit_Price': row['close_ce'], 'Total_Exit_Price': row['total_close'],
                        'Exit_Price': row['total_close'], 'Exit_Reason': exit_reason, 'PnL': pnl,
                        'Entry_Index_Close': entry_index_close, 'Entry_Index_Close_at_Entry': index_close_for_entry,
                        'Exit_Index_Close': row['index_close']
                    })
                    trade_number += 1
                    trade_exited = True
                    
                    # Update State for Re-entry
                    last_exit_index_close = row['index_close']
                    current_dte_index = mon_dte_idx # Resume from this DTE
                    current_time_cursor = row['time'] # Resume from this time
                    just_exited = True
                    
                    break
            
            if trade_exited:
                break
                
        # End of Monitoring Loop
        if not trade_exited:
            # If we ran out of data without exiting?
            # Force close at last available point or mark as open?
            # For backtest, force close at end of data
            continue_trading = False 
            
    return all_trades

# ============================================================================
# UNIFIED STRATEGY EXECUTION
# ============================================================================

def execute_strategy(strategy_row, backtest_data, strategy_config, filename, strategy_name,
                     mode='Weekly', entry_time='09:15', exit_time='15:20', 
                     entry_dte=None, exit_dte=None, entry_exit_times=None):
    if mode == 'Intraday':
        # Default behavior validation
        if entry_exit_times is None:
            # Fallback to using scalar times for all DTEs if dict not provided
            dtes = ['4 DTE', '3 DTE', '2 DTE', '1 DTE', '0 DTE']
            entry_exit_times = {dte: {'entry': entry_time, 'exit': exit_time} for dte in dtes}
            
        return execute_intraday_strategy(strategy_row, backtest_data, strategy_config, filename, strategy_name,
                                          entry_exit_times, entry_dte or strategy_row['DTE'], exit_dte or '0 DTE')
    else:
        return execute_interday_strategy(strategy_row, backtest_data, strategy_config, filename, strategy_name,
                                         entry_dte=entry_dte, entry_time=entry_time, exit_dte=exit_dte, exit_time=exit_time)

# ============================================================================
# MULTI-STRATEGY BACKTEST EXECUTION
# ============================================================================

def run_multi_strategy_backtest(files_dict, strategies_dict, progress_callback=None,
                                 mode='Weekly', entry_time='09:15', exit_time='15:20', 
                                 entry_dte='4 DTE', exit_dte='0 DTE', entry_exit_times=None):
    all_results = {}
    total = len(files_dict) * len(strategies_dict)
    count = 0
    
    for strategy_name, strategy_config in strategies_dict.items():
        all_results[strategy_name] = {}
        for filename, df in files_dict.items():
            count += 1
            if progress_callback:
                progress_callback(f"[{mode}] {strategy_name} on {filename} ({count}/{total})")
            
            try:
                if filename not in state.processed_data:
                    backtest_data = preprocess_data(df)
                    state.processed_data[filename] = backtest_data
                else:
                    backtest_data = state.processed_data[filename]
                
                entry_strategy_row = strategy_config[strategy_config['DTE'] == entry_dte]
                if entry_strategy_row.empty:
                    entry_strategy_row = strategy_config.iloc[[0]]
                else:
                    entry_strategy_row = entry_strategy_row.iloc[0]
                
                results = execute_strategy(entry_strategy_row, backtest_data, strategy_config, filename, strategy_name,
                                           mode, entry_time=entry_time, exit_time=exit_time, entry_dte=entry_dte, exit_dte=exit_dte,
                                           entry_exit_times=entry_exit_times)
                all_results[strategy_name][filename] = results
            except Exception as e:
                print(f"❌ Error: {strategy_name} on {filename}: {str(e)}")
                all_results[strategy_name][filename] = []
    
    return all_results

# ============================================================================
# RESULTS & EXCEL
# ============================================================================

def consolidate_results(all_results):
    all_trades = []
    for s, file_results in all_results.items():
        for f, trades in file_results.items():
            all_trades.extend(trades)
    return pd.DataFrame(all_trades) if all_trades else None

def calculate_strategy_metrics(results_df, strategy_name):
    strategy_df = results_df[results_df['Strategy'] == strategy_name]
    if len(strategy_df) == 0: return None
    
    total_pnl = strategy_df['PnL'].sum()
    avg_pnl = strategy_df['PnL'].mean()
    win_rate = (len(strategy_df[strategy_df['PnL'] > 0]) / len(strategy_df) * 100)
    
    return {
        'Strategy': strategy_name, 'Total_Trades': len(strategy_df),
        'Win_Rate_%': round(win_rate, 2), 'Total_PnL': round(total_pnl, 2),
        'Avg_PnL': round(avg_pnl, 2)
    }

def generate_strategy_comparison(results_df):
    strategies = results_df['Strategy'].unique()
    data = [calculate_strategy_metrics(results_df, s) for s in strategies]
    return pd.DataFrame([d for d in data if d])

def export_comprehensive_excel(results_df, strategies_dict, output_filename, mode='Weekly'):
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        generate_strategy_comparison(results_df).to_excel(writer, sheet_name='Strategy_Comparison', index=False)
        results_df.to_excel(writer, sheet_name='All_Trades', index=False)
        
        config_rows = []
        for name, config in strategies_dict.items():
            c = config.copy()
            c.insert(0, 'Strategy', name)
            config_rows.append(c)
        pd.concat(config_rows).to_excel(writer, sheet_name='Strategy_Configs', index=False)
        
        for name in results_df['Strategy'].unique():
            results_df[results_df['Strategy'] == name].to_excel(writer, sheet_name=f"{name[:25]}_Details", index=False)

# ============================================================================
# UI COMPONENTS (MODIFIED)
# ============================================================================

def create_enhanced_file_upload_ui():
    upload_widget = widgets.FileUpload(accept='.csv,.xlsx,.xls', multiple=True, description='Upload Files')
    
    # Folder Load Option
    folder_path_input = widgets.Text(placeholder='Enter Folder Path', description='Folder Path:')
    load_folder_btn = widgets.Button(description='Load from Folder', button_style='info')
    
    file_list_output = widgets.Output()
    
    def on_upload_change(change):
        with file_list_output:
            clear_output()
            for f in change['new']:
                try:
                    content = f['content']
                    df = pd.read_csv(io.BytesIO(content)) if f['name'].endswith('.csv') else pd.read_excel(io.BytesIO(content))
                    state.add_file(f['name'], df)
                    print(f"✅ Uploaded: {f['name']}")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    
    def on_load_folder_click(b):
        with file_list_output:
            clear_output()
            try:
                path = folder_path_input.value
                loaded = load_files_from_folder(path)
                for name, df in loaded.items():
                    state.add_file(name, df)
                print(f"✅ Loaded {len(loaded)} files from {path}")
            except Exception as e:
                print(f"❌ Error loading folder: {str(e)}")
    
    upload_widget.observe(on_upload_change, names='value')
    load_folder_btn.on_click(on_load_folder_click)
    
    return widgets.VBox([
        widgets.HTML("<h3>Step 1: Load Data</h3>"),
        widgets.HBox([widgets.Label("Method A: File Upload"), upload_widget]),
        widgets.HBox([widgets.Label("Method B: Folder Load"), folder_path_input, load_folder_btn]),
        file_list_output
    ])

def generate_cartesian_strategies(ranges_dict, dtes=['4 DTE', '3 DTE', '2 DTE', '1 DTE', '0 DTE'], mode='Weekly', dte_specific_ranges=None):
    """
    Generate strategy combinations using Cartesian product of ranges.
    ranges_dict: { 'Stoploss %': [start, end, step], ... }
    mode: 'Weekly' (Bundled IDs) or 'DTE' (Unique IDs per DTE)
    dte_specific_ranges: dict { '4 DTE': {'PE Strike': [s,e,st], 'CE Strike': [s,e,st]}, ... } (Optional, for DTE mode)
    Returns: DataFrame ready for Excel export
    """
    import itertools
    import numpy as np
    
    # helper to expand range
    def expand_range(start, end, step):
        if start == end: return [start]
        if step == 0: step = 1
        vals = np.arange(start, end + step/1000, step).tolist()
        return [round(x, 2) if isinstance(x, float) else x for x in vals]

    # 1. Expand Global Ranges (Standard Params)
    # If in DTE mode with specific ranges, we might exclude PE/CE from global expansion 
    # and handle them in the inner loop.
    
    global_params = {}
    for param, (start, end, step) in ranges_dict.items():
        # Optimization: In DTE mode with overrides, skip PE/CE here if overridden
        if mode == 'DTE' and dte_specific_ranges and param in ['PE Strike', 'CE Strike']:
            continue
        global_params[param] = expand_range(start, end, step)
    
    # Calculate Global Combinations
    global_keys = list(global_params.keys())
    global_values = list(global_params.values())
    global_combinations = list(itertools.product(*global_values))
    
    print(f"Generating strategies (Mode: {mode}, Base Combos: {len(global_combinations)})...")
    
    rows = []
    
    for i, global_combo in enumerate(global_combinations):
        base_strat_id = f"GenStrat_{i+1:04d}"
        global_dict = dict(zip(global_keys, global_combo))
        
        # Create a row for each DTE
        for dte in dtes:
            if mode == 'DTE':
                strat_id = base_strat_id
                
                # Check for DTE-specific overrides
                current_dte_params = global_dict.copy()
                
                if dte_specific_ranges and dte in dte_specific_ranges:
                    # We have overrides (e.g. PE/CE)
                    overrides = dte_specific_ranges[dte]
                    override_params = {}
                    for p, (s, e, st) in overrides.items():
                        override_params[p] = expand_range(s, e, st)
                        
                    ov_keys = list(override_params.keys())
                    ov_vals = list(override_params.values())
                    ov_combos = list(itertools.product(*ov_vals))
                    
                    # Generate rows for EACH override combo
                    for j, ov_combo in enumerate(ov_combos):
                        final_dict = current_dte_params.copy()
                        final_dict.update(dict(zip(ov_keys, ov_combo)))
                        
                        # Variant Suffix: Append _v{j+1} if variations exist
                        # If only 1 variation (single value override), keep base ID
                        variant_strat_id = strat_id if len(ov_combos) == 1 else f"{strat_id}_v{j+1}"
                        
                        row = {'Strategy': variant_strat_id, 'DTE': dte}
                        row.update(final_dict)
                        rows.append(row)
                else:
                    # No overrides
                    row = {'Strategy': strat_id, 'DTE': dte}
                    row.update(current_dte_params)
                    rows.append(row)
                    
            else:
                # Weekly Mode (Bundled) - Use Global Dict as is
                strat_id = base_strat_id
                row = {'Strategy': strat_id, 'DTE': dte}
                row.update(global_dict)
                rows.append(row)
            
    return pd.DataFrame(rows)

def create_strategy_selection_ui():
    """Create strategy selection UI with Triple Configuration Options"""
    
    # === OPTION A: MANUAL UI ===
    preset_checkboxes = {}
    
    for preset_name in STRATEGY_PRESETS.keys():
        preset_checkboxes[preset_name] = widgets.Checkbox(value=False, description=preset_name)
    
    manual_ui = widgets.VBox([
        widgets.HTML("<h4>Option A: Select Presets (Manual)</h4>"),
        widgets.VBox([widgets.HBox([cb]) for cb in preset_checkboxes.values()])
    ])
    
    # === OPTION B: EXCEL UPLOAD ===
    excel_upload = widgets.FileUpload(accept='.xlsx', description='Upload Strategy config')
    excel_status = widgets.Output()
    
    excel_ui = widgets.VBox([
        widgets.HTML("<h4>Option B: Upload Strategy Excel</h4>"),
        excel_upload, excel_status
    ])
    
    # === OPTION C: GENERATOR ===
    # Ranges: Stoploss, Profit, Lot, PE Strike, CE Strike, Index Move
    # Inputs: Start, End, Step
    
    gen_params = [
        ('Stoploss %', 30, 100, 5),
        ('Profit Target %', 20, 100, 5),
        ('lot_size', 1, 1, 1),
        ('PE Strike', 0, 6, 1),
        ('CE Strike', 0, 6, 1),
        ('Index Movement +/-', 0, 400, 50)
    ]
    
    gen_widgets = {}
    gen_rows = [widgets.HBox([
        widgets.Label("Parameter", layout=widgets.Layout(width='150px', font_weight='bold')),
        widgets.Label("Start", layout=widgets.Layout(width='80px', font_weight='bold')),
        widgets.Label("End", layout=widgets.Layout(width='80px', font_weight='bold')),
        widgets.Label("Step", layout=widgets.Layout(width='80px', font_weight='bold'))
    ])]
    
    for label, def_start, def_end, def_step in gen_params:
        w_start = widgets.FloatText(value=def_start, layout=widgets.Layout(width='80px'))
        w_end = widgets.FloatText(value=def_end, layout=widgets.Layout(width='80px'))
        w_step = widgets.FloatText(value=def_step, layout=widgets.Layout(width='80px'))
        
        gen_widgets[label] = {'start': w_start, 'end': w_end, 'step': w_step}
        gen_rows.append(widgets.HBox([
            widgets.Label(label, layout=widgets.Layout(width='150px')),
            w_start, w_end, w_step
        ]))
    
    # Constraints
    max_combos = widgets.IntText(value=10000, description='Max Combos:')
    
    # Generator UI
    gen_mode_toggle = widgets.ToggleButtons(
        options=['Weekly', 'DTE'],
        description='Gen Mode:',
        value='Weekly',
        button_style='warning',
        tooltips=['Weekly: Same Strategy ID for all DTEs', 'DTE: Unique Strategy ID per DTE']
    )

    gen_action_btn = widgets.Button(description='Generate & Save Excel', button_style='info')
    gen_status = widgets.Output()
    
    def on_gen_click(b):
        with gen_status:
            clear_output()
            try:
                mode = gen_mode_toggle.value
                
                # 1. Collect Ranges
                ranges = {}
                for label, w_dict in gen_widgets.items():
                    start = w_dict['start'].value
                    end = w_dict['end'].value
                    step = w_dict['step'].value
                    ranges[label] = [start, end, step]
                
                dte_specific_ranges = None
                
                # 2. Collect DTE Specifics if DTE Mode
                if mode == 'DTE':
                    dte_specific_ranges = {}
                    # Iterate through dte_table_widgets
                    for dte, w_row in dte_table_widgets.items():
                        # PE
                        pe_s = w_row['pe_start'].value
                        pe_e = w_row['pe_end'].value
                        pe_st = w_row['pe_step'].value
                        # CE
                        ce_s = w_row['ce_start'].value
                        ce_e = w_row['ce_end'].value
                        ce_st = w_row['ce_step'].value
                        
                        dte_specific_ranges[dte] = {
                            'PE Strike': [pe_s, pe_e, pe_st],
                            'CE Strike': [ce_s, ce_e, ce_st]
                        }
                
                # 3. Generate
                df_gen = generate_cartesian_strategies(ranges, mode=mode, dte_specific_ranges=dte_specific_ranges)
                
                # 4. Save
                fname = f"Generated_Strategies_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                with pd.ExcelWriter(fname) as writer:
                    df_gen.to_excel(writer, index=False)
                
                print(f"✅ Success! Generated {len(df_gen)} rows.")
                print(f"📁 Saved to: {fname}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
            
    gen_action_btn.on_click(on_gen_click)
    
    # === DTE Parameter Table UI ===
    dte_table_widgets = {}
    dte_rows = []
    
    # Header
    dte_header = widgets.HBox([
        widgets.Label("DTE", layout=widgets.Layout(width='60px', font_weight='bold')), 
        widgets.Label("PE Start", layout=widgets.Layout(width='60px')), 
        widgets.Label("PE End", layout=widgets.Layout(width='60px')), 
        widgets.Label("Step", layout=widgets.Layout(width='40px')),
        widgets.Label("|", layout=widgets.Layout(width='20px')),
        widgets.Label("CE Start", layout=widgets.Layout(width='60px')), 
        widgets.Label("CE End", layout=widgets.Layout(width='60px')), 
        widgets.Label("Step", layout=widgets.Layout(width='40px'))
    ])
    dte_rows.append(dte_header)
    
    # Rows for each DTE
    for dte in ['4 DTE', '3 DTE', '2 DTE', '1 DTE', '0 DTE']:
        pe_s = widgets.FloatText(value=0, layout=widgets.Layout(width='60px'))
        pe_e = widgets.FloatText(value=4, layout=widgets.Layout(width='60px'))
        pe_st = widgets.FloatText(value=1, layout=widgets.Layout(width='40px'))
        
        ce_s = widgets.FloatText(value=0, layout=widgets.Layout(width='60px'))
        ce_e = widgets.FloatText(value=4, layout=widgets.Layout(width='60px'))
        ce_st = widgets.FloatText(value=1, layout=widgets.Layout(width='40px'))
        
        dte_table_widgets[dte] = {
            'pe_start': pe_s, 'pe_end': pe_e, 'pe_step': pe_st,
            'ce_start': ce_s, 'ce_end': ce_e, 'ce_step': ce_st
        }
        
        row = widgets.HBox([
            widgets.Label(dte, layout=widgets.Layout(width='60px')),
            pe_s, pe_e, pe_st,
            widgets.Label("|", layout=widgets.Layout(width='20px')),
            ce_s, ce_e, ce_st
        ])
        dte_rows.append(row)
        
    dte_table_container = widgets.VBox([
        widgets.HTML("<b>Per-DTE PE/CE Configuration:</b>"),
        widgets.VBox(dte_rows)
    ], layout=widgets.Layout(border='1px solid #ccc', padding='5px', display='none')) # Hidden by default
    
    # PE/CE Global Rows container (to hide them in DTE mode)
    # We need to identify PE/CE rows from gen_rows list. 
    # gen_widgets has keys. We can wrap PE/CE rows in a container.
    # But gen_rows is already a list of HBoxes.
    # Let's rebuild the main layout slightly to separate Common vs PE/CE global.
    
    # HACK: Toggle display of dynamic elements
    def on_gen_mode_change(change):
        if change['new'] == 'DTE':
            dte_table_container.layout.display = 'block'
            # Optional: Hide global PE/CE inputs to avoid confusion?
            # User might still want Global settings for other params.
            # Ideally we hide the PE/CE lines from the main list.
            # But accessing them in `gen_rows` list is tricky without re-rendering.
            # Pass choice to user: "DTE Mode uses this table for PE/CE, ignores Global PE/CE".
        else:
            dte_table_container.layout.display = 'none'
            
    gen_mode_toggle.observe(on_gen_mode_change, names='value')

    generator_ui = widgets.VBox([
        widgets.HTML("<h4>Option C: Generate Strategy Excel</h4>"),
        gen_mode_toggle,
        widgets.VBox(gen_rows), # Global Params
        dte_table_container,    # DTE Specific Table
        widgets.HBox([max_combos]),
        widgets.HTML("<br>"),
        gen_action_btn,
        gen_status
    ])
    
    # === TABS ===
    tabs = widgets.Tab(children=[manual_ui, excel_ui, generator_ui])
    tabs.set_title(0, 'Manual Presets')
    tabs.set_title(1, 'Upload Excel')
    tabs.set_title(2, 'Generator')
    
    generate_config_btn = widgets.Button(description='🔧 Load/Parse Configs', button_style='success')
    config_output = widgets.Output()
    
    def on_load_config_click(b):
        with config_output:
            clear_output()
            if state.get_file_count() == 0:
                print("❌ Please upload data files first!")
                return
            
            first_filename = state.get_filenames()[0]
            processed = preprocess_data(state.uploaded_files[first_filename])
            state.processed_data[first_filename] = processed
            base_config = create_base_strategy_config(processed)
            
            active_tab = tabs.selected_index
            strategies = {}
            
            if active_tab == 0: # Manual
                selected = [name for name, cb in preset_checkboxes.items() if cb.value]
                if not selected:
                    print("❌ Please select a strategy!")
                    return
                selected_presets_with_params = {}
                for name in selected:
                    selected_presets_with_params[name] = STRATEGY_PRESETS[name] 
                strategies = generate_strategy_combinations(base_config, selected_presets_with_params)
                
            elif active_tab == 1: # Excel
                if not excel_upload.value:
                    print("❌ Please upload a strategy Excel file!")
                    return
                uploaded_file = excel_upload.value[0]
                strategies = parse_strategy_excel(uploaded_file['content'], base_config)
            
            else: # Generator
                 print("⚠️ Please generate the Excel first, then upload it in the 'Upload Excel' tab.")
                 return
            
            for name, config in strategies.items():
                state.add_strategy_config(name, config)
            
            print(f"✅ Loaded {len(strategies)} strategy configurations!")
            for name in list(strategies.keys())[:10]:
                print(f"  • {name}")
            if len(strategies) > 10: print(f"  ... and {len(strategies)-10} more.")

    generate_config_btn.on_click(on_load_config_click)
    
    return widgets.VBox([
        widgets.HTML("<h3>Step 2: Configure Strategies</h3>"),
        tabs,
        widgets.HTML("<br>"),
        generate_config_btn,
        config_output
    ])

def create_backtest_execution_ui():
    # Execution Parameters
    mode_toggle = widgets.ToggleButtons(
        options=['Weekly', 'Intraday'],
        description='Mode:',
        value='Weekly',
        button_style='primary'
    )
    
    dte_options = ['4 DTE', '3 DTE', '2 DTE', '1 DTE', '0 DTE']
    
    # === WEEKLY MODE WIDGETS ===
    # Entry Parameters
    w_entry_dte = widgets.Dropdown(options=dte_options, value='4 DTE', layout=widgets.Layout(width='100px'))
    w_entry_time = widgets.Text(value='09:30', layout=widgets.Layout(width='100px'))
    
    # Exit Parameters
    # "At Expiry" logic will be handled as 0 DTE + Time check or End of Data
    w_exit_dte = widgets.Dropdown(options=['At Expiry'] + dte_options, value='At Expiry', layout=widgets.Layout(width='100px'))
    w_exit_time = widgets.Text(value='15:20', layout=widgets.Layout(width='100px'))
    
    weekly_ui = widgets.VBox([
        widgets.HTML("<b>Entry Parameters:</b>"),
        widgets.HBox([
            widgets.Label("Entry DTE:", layout=widgets.Layout(width='80px')), w_entry_dte,
            widgets.Label("Entry Time:", layout=widgets.Layout(width='80px')), w_entry_time
        ]),
        widgets.HTML("<br><b>Exit Parameters:</b>"),
        widgets.HBox([
            widgets.Label("Exit DTE:", layout=widgets.Layout(width='80px')), w_exit_dte,
            widgets.Label("Exit Time:", layout=widgets.Layout(width='80px')), w_exit_time
        ])
    ])
    
    # === INTRADAY MODE WIDGETS ===
    # Per-DTE Time Configuration (Intraday)
    time_config_widgets = {}
    time_rows = []
    
    header = widgets.HBox([
        widgets.Label("DTE", layout=widgets.Layout(width='60px', font_weight='bold')), 
        widgets.Label("Entry Time", layout=widgets.Layout(width='100px', font_weight='bold')), 
        widgets.Label("Exit Time", layout=widgets.Layout(width='100px', font_weight='bold'))
    ])
    time_rows.append(header)
    
    defaults = {
        '4 DTE': ('09:15', '15:20'), '3 DTE': ('09:15', '15:20'),
        '2 DTE': ('09:15', '15:20'), '1 DTE': ('09:15', '15:20'),
        '0 DTE': ('09:15', '15:20')
    }
    
    for dte in dte_options:
        ent_default, ext_default = defaults.get(dte)
        ent_w = widgets.Text(value=ent_default, layout=widgets.Layout(width='100px'))
        ext_w = widgets.Text(value=ext_default, layout=widgets.Layout(width='100px'))
        time_config_widgets[dte] = {'entry': ent_w, 'exit': ext_w}
        
        row = widgets.HBox([
            widgets.Label(dte, layout=widgets.Layout(width='60px')),
            ent_w, ext_w
        ])
        time_rows.append(row)
        
    intraday_ui = widgets.VBox([
        widgets.HTML("<b>Time Configuration (Per DTE):</b>"),
        widgets.VBox(time_rows)
    ])
    
    # Dynamic display container
    settings_container = widgets.VBox([weekly_ui])
    
    def on_mode_change(change):
        if change['new'] == 'Weekly':
            settings_container.children = [weekly_ui]
        else:
            settings_container.children = [intraday_ui]
    
    mode_toggle.observe(on_mode_change, names='value')
    
    run_btn = widgets.Button(description='🚀 Run Backtest', button_style='success')
    output = widgets.Output()
    
    def on_run_click(b):
        with output:
            clear_output()
            if not state.strategy_configs:
                print("❌ No strategy configs! Please generate or upload strategies first.")
                return
            
            mode = mode_toggle.value
            
            # Common Params defaults
            entry_dte = '4 DTE'
            exit_dte = '0 DTE'
            entry_time = '09:15'
            exit_time = '15:20'
            entry_exit_times = None
            
            if mode == 'Weekly':
                entry_dte = w_entry_dte.value
                entry_time = w_entry_time.value
                
                raw_exit = w_exit_dte.value
                exit_dte = '0 DTE' if raw_exit == 'At Expiry' else raw_exit
                exit_time = w_exit_time.value
                
                print(f"Running Weekly Mode | Entry: {entry_dte} @ {entry_time} -> Exit: {exit_dte} @ {exit_time} (Continuous Re-entry)")
                
            else: # Intraday
                entry_dte = '4 DTE' # Default/Placeholder, engine iterates all
                exit_dte = '0 DTE'
                
                entry_exit_times = {}
                for dte, w_dict in time_config_widgets.items():
                    e_t = w_dict['entry'].value.strip()
                    x_t = w_dict['exit'].value.strip()
                    if e_t and x_t:
                        entry_exit_times[dte] = {'entry': e_t, 'exit': x_t}
                
                print(f"Running Intraday Mode | Per-DTE Configuration")
            
            all_results = run_multi_strategy_backtest(
                state.uploaded_files, 
                state.strategy_configs, 
                lambda x: print(x),
                mode=mode,
                entry_time=entry_time,
                exit_time=exit_time,
                entry_dte=entry_dte,
                exit_dte=exit_dte,
                entry_exit_times=entry_exit_times
            )
            
            res = consolidate_results(all_results)
            if res is not None:
                fname = f"Backtest_Result_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                export_comprehensive_excel(res, state.strategy_configs, fname, mode=mode)
                print(f"✅ Saved: {fname}")
                state.consolidated_results = res
            else:
                print("No trades generated.")

    run_btn.on_click(on_run_click)
    
    return widgets.VBox([
        widgets.HTML("<h3>Step 3: Execute</h3>"),
        mode_toggle,
        widgets.HTML("<hr>"),
        settings_container,
        widgets.HTML("<hr>"),
        run_btn, output
    ])

def launch():
    display(widgets.VBox([
        create_enhanced_file_upload_ui(),
        create_strategy_selection_ui(),
        create_backtest_execution_ui()
    ]))

if __name__ == '__main__':
    launch()
