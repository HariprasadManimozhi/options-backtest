
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
import time
from itertools import product
import warnings
import concurrent.futures
import concurrent.futures
import multiprocessing

try:
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
    CUDA_AVAILABLE = cuda.is_available()
    if CUDA_AVAILABLE:
        print("✅ NVIDIA GPU Detected & Enabled (CUDA)")
    else:
        print("✅ CPU Acceleration Enabled (Numba)")
        
except ImportError:
    NUMBA_AVAILABLE = False
    CUDA_AVAILABLE = False
    print("⚠️ Acceleration Library Not Found. Using Standard CPU Mode.")
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    # Dummy cuda object
    class CudaMock:
        def jit(self, *args, **kwargs):
             def decorator(func): return func
             return decorator
        def is_available(self): return False
    cuda = CudaMock()

warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL STATE MANAGEMENT
# ============================================================================

class BacktestState:
    """Centralized state management for the backtest system"""
    def __init__(self):
        self.uploaded_files = {} # Stores {filename: filepath_str} or {filename: dataframe}
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
        "stoploss_pct": [100, 100, 100, 100, 100],
        "profit_target_pct": [75, 75, 75, 75, 75],
        "pe_strike_offset": [1, 1, 1, 1, 1],
        "ce_strike_offset": [0, 0, 0, 0, 0],
        "index_movement": [250, 250, 250, 250, 250],
        "lot_size": [1, 1, 1, 1, 1]
    },
    "Custom1": {
        "description": "User-defined parameters",
        "stoploss_pct": [80, 80, 80, 80, 80],
        "profit_target_pct": [75, 75, 75, 75, 75],
        "pe_strike_offset": [0, 0, 0, 0, 0],
        "ce_strike_offset": [0, 0, 0, 0, 0],
        "index_movement": [250, 250, 250, 250, 250],
        "lot_size": [1, 1, 1, 1, 1]
    }

}

DTE_LABELS = ['4 DTE', '3 DTE', '2 DTE', '1 DTE', '0 DTE']

# ============================================================================
# DATA PREPROCESSING (OPTIMIZED)
# ============================================================================

def preprocess_data(source):
    """
    Enhanced data preprocessing with memory optimization.
    source: Can be a DataFrame or a file path (str/Path).
    """
    
    # 1. Load Data if source is a path
    if isinstance(source, (str, Path)):
        file_path = Path(source)
        if file_path.suffix.lower() == '.csv':
             # Optimize: Load only necessary columns to save memory
             # Check if we can determine columns first? No, just load.
             df = pd.read_csv(file_path) 
        else:
             df = pd.read_excel(file_path)
    else:
        df = source.copy()

    required_cols = ['expiry_date', 'date_only', 'time', 'symbol', 'open', 'high', 'low', 'close', 'index_close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # 2. Type Optimization (Memory Reduction)
    float_cols = ['open', 'high', 'low', 'close', 'index_close']
    for col in float_cols:
        if col in df.columns:
             df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
             
    # Date Handling
    df['expiry_date'] = pd.to_datetime(df['expiry_date']).dt.date
    df['date_only'] = pd.to_datetime(df['date_only']).dt.date
    
    # Calculate Days to Expiry
    df['days_to_expiry'] = (pd.to_datetime(df['expiry_date']) - pd.to_datetime(df['date_only'])).dt.days
    
    # Filter for relevant days (<= 6 days usually for weekly)
    df = df[df['days_to_expiry'] <= 6].copy()
    
    cols = ['expiry_date', "date_only", 'days_to_expiry', 'time', 'symbol', 'open', 'high', 'low', 'close', 'index_close']
    if 'DTE' in df.columns:
        cols.append('DTE')
    
    backtest_data = df[cols].copy()
    
    # DTE Formatter
    if 'DTE' in backtest_data.columns:
        def extract_dte_format(dte_str):
            if pd.isna(dte_str): return None
            dte_str = str(dte_str).strip().upper()
            if dte_str in ['ODTE', '0DTE']: return '0 DTE'
            import re
            match = re.search(r'(\d+)', dte_str)
            if match: return f'{match.group(1)} DTE'
            return None
        
        backtest_data['dte_format'] = backtest_data['DTE'].apply(extract_dte_format)
        # Fallback if DTE col was empty/invalid
        if backtest_data['dte_format'].isna().all():
             backtest_data['dte_format'] = backtest_data['days_to_expiry'].apply(lambda x: f'{x} DTE')
    else:
        backtest_data['dte_format'] = backtest_data['days_to_expiry'].apply(lambda x: f'{x} DTE')
    
    # Convert dte_format to category to save memory
    backtest_data['dte_format'] = backtest_data['dte_format'].astype('category')
    
    return backtest_data

# ============================================================================
# FILE HANDLING (LAZY LOAD)
# ============================================================================

def load_files_from_folder(folder_path):
    """
    Identify all CSV and Excel files in a folder. 
    Returns a DICTIONARY of {filename: absolute_path} (Lazy Loading).
    Does NOT load content into RAM yet.
    """
    loaded_files = {}
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder path does not exist: {folder_path}")
    
    supported_extensions = ['.csv', '.xlsx', '.xls']
    files = [f for f in folder.iterdir() if f.suffix.lower() in supported_extensions]
    
    if not files:
        raise ValueError(f"No CSV or Excel files found in: {folder_path}")
    
    print(f"📂 Found {len(files)} files in folder (Metadata only, content lazily loaded)")
    
    for file_path in files:
        filename = file_path.name
        # Store PATH using str() for serialization compatibility
        loaded_files[filename] = str(file_path.absolute()) 
        
    return loaded_files

# ============================================================================
# STRATEGY CONFIGURATION
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
    """Parse strategy parameters from uploaded Excel file."""
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
            config = base_config[['DTE', 'Index Close']].copy()
            merged = pd.merge(config, group, on='DTE', how='left')
            
            defaults = {
                'Stoploss %': 60, 'Profit Target %': 50, 'lot_size': 1,
                'PE Strike': 4, 'CE Strike': 4, 'Index Movement +/-': 100
            }
            for col, val in defaults.items():
                if col in merged.columns:
                    merged[col] = merged[col].fillna(val)
            
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
# FAST PATH EXECUTION (NUMBA/JIT)
# ============================================================================

@jit(nopython=True)
def scan_trades_fast(time_mins, total_closes, index_closes, close_pes, close_ces,
                     entry_time_min, exit_time_min,
                     stoploss_price, profit_target_price, 
                     index_upper, index_lower):
    """
    Optimized scan loop. 
    Returns: (index_in_array, exit_reason_code)
    Exit Reasons: 1=Stoploss, 2=Target, 3=IndexBreach, 4=TimeExit
    """
    n = len(time_mins)
    
    for i in range(n):
        t = time_mins[i]
        
        # Skip if before entry (though input array should already be filtered)
        if t <= entry_time_min: continue
        
        tc = total_closes[i]
        ic = index_closes[i]
        
        # Check Exits
        if tc >= stoploss_price:
            return i, 1
        elif tc <= profit_target_price:
            return i, 2
        elif ic >= index_upper or ic <= index_lower:
            return i, 3
        elif t >= exit_time_min:
            return i, 4
            
    return -1, 0

if CUDA_AVAILABLE:
    @cuda.jit
    def scan_trades_cuda(time_mins, total_closes, index_closes, 
                         entry_time_min, exit_time_min,
                         stoploss_price, profit_target_price, 
                         index_upper, index_lower, 
                         result_arr):
        """
        CUDA Kernel for trade scanning.
        Operates on single thread per block or similar, simplistic for demonstration.
        Ideally we batch many trades, but here we accelerate the Loop over time steps.
        """
        # Linear thread index doesn't map perfectly to time series loop.
        # But we can use one thread to scan the array? No, that's not parallelism.
        # We use grid-stride loop or just parallel threads checking different indices?
        # Parallel threads over time points:
        # If thread i finds exit, it writes to result using atomicMin? 
        # We need FIRST exit. AtomicMin on index is good.
        
        i = cuda.grid(1)
        if i < time_mins.shape[0]:
            t = time_mins[i]
            
            # Valid time range check?
            if t > entry_time_min:
                tc = total_closes[i]
                ic = index_closes[i]
                
                reason = 0
                if tc >= stoploss_price:
                    reason = 1
                elif tc <= profit_target_price:
                    reason = 2
                elif ic >= index_upper or ic <= index_lower:
                    reason = 3
                elif t >= exit_time_min:
                    reason = 4
                
                if reason > 0:
                    # Found an exit trigger at index i.
                    # We want the SMALLEST index i across all threads.
                    cuda.atomic.min(result_arr, 0, i)
                    # We also need to store reason.
                    # This is tricky with race conditions on storing reason for min index.
                    # Simplified: Just find min index first. Then retrieving reason is trivial on CPU.
                    
        # result_arr[0] init to 999999 before kernel


def time_to_min(t_str):
    try:
        h, m = map(int, t_str.split(':'))
        return h * 60 + m
    except:
        return 9999

# ============================================================================
# CORE EXECUTION LOGIC (INTRADAY & INTERDAY)
# ============================================================================

def execute_intraday_strategy(strategy_row, backtest_data, strategy_config, filename, strategy_name,
                               entry_exit_times, entry_dte, exit_dte):
    
    all_trades = []
    trade_number = 1
    
    entry_dte_num = get_dte_number(entry_dte)
    exit_dte_num = 0 if exit_dte in ['At Expiry', '0 DTE', 'Default (Expiry)'] else get_dte_number(exit_dte)
    
    if entry_dte_num < exit_dte_num:
        return all_trades
    
    for current_dte_num in range(entry_dte_num, exit_dte_num - 1, -1):
        current_dte = f'{current_dte_num} DTE'
        
        dte_times = entry_exit_times.get(current_dte)
        if not dte_times: continue
            
        entry_time = dte_times.get('entry', '09:15')
        exit_time = dte_times.get('exit', '15:20')
        
        matching_strategy = strategy_config[strategy_config['DTE'] == current_dte]
        if matching_strategy.empty: continue
        
        current_strategy = matching_strategy.iloc[0]
        pe_strike_offset = current_strategy['PE Strike']
        ce_strike_offset = current_strategy['CE Strike']
        lot_size = current_strategy['lot_size']
        stoploss_pct = current_strategy['Stoploss %']
        profit_target_pct = current_strategy['Profit Target %']
        index_movement = current_strategy['Index Movement +/-']
        
        dte_data = backtest_data[backtest_data['dte_format'] == current_dte].copy()
        if dte_data.empty: continue
            
        unique_dates = sorted(dte_data['date_only'].unique())
        
        for trade_date in unique_dates:
            date_data = dte_data[dte_data['date_only'] == trade_date].copy()
            entry_data = date_data[date_data['time'] == entry_time]
            
            if entry_data.empty: continue
            
            index_close_at_entry = entry_data.iloc[0]['index_close']
            if pd.isna(index_close_at_entry): continue

            try:
                index_close_for_entry = round(index_close_at_entry / 50) * 50
                pe_strike = int(index_close_for_entry - (pe_strike_offset * 50))
                ce_strike = int(index_close_for_entry + (ce_strike_offset * 50))
                
                pe_symbol = f"{pe_strike}-PE"
                ce_symbol = f"{ce_strike}-CE"
            except Exception as e:
                continue
            
            pe_entry = entry_data[entry_data['symbol'].str.contains(pe_symbol, na=False)]
            ce_entry = entry_data[entry_data['symbol'].str.contains(ce_symbol, na=False)]
            
            if pe_entry.empty or ce_entry.empty: continue
            
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
            
            if monitoring_pe.empty or monitoring_ce.empty: continue
            
            monitoring_data = pd.merge(
                monitoring_pe[['date_only', 'time', 'close', 'index_close']],
                monitoring_ce[['date_only', 'time', 'close']],
                on=['date_only', 'time'], suffixes=('_pe', '_ce')
            )
            
            if monitoring_data.empty: continue
            
            monitoring_data['total_close'] = monitoring_data['close_pe'] + monitoring_data['close_ce']
            monitoring_data = monitoring_data.sort_values('time')
            
            exit_reason = None
            exit_price = None
            pe_exit_price = None
            ce_exit_price = None
            exit_time_actual = None
            exit_index_close = None
            
            if NUMBA_AVAILABLE and not monitoring_data.empty:
                # Prepare arrays
                times_min = monitoring_data['time'].apply(time_to_min).values
                total_closes = monitoring_data['total_close'].values.astype(np.float64)
                index_closes = monitoring_data['index_close'].values.astype(np.float64)
                
                entry_min = time_to_min(entry_time)
                exit_min = time_to_min(exit_time)
                
                idx = -1
                reason_code = 0
                
                # --- CUDA PATH ---
                if CUDA_AVAILABLE and len(times_min) > 50: # Threshold overhead
                    try:
                        import math
                        d_times = cuda.to_device(times_min)
                        d_total = cuda.to_device(total_closes)
                        d_index = cuda.to_device(index_closes)
                        d_res = cuda.to_device(np.array([999999], dtype=np.int32))
                        
                        threadsperblock = 256
                        blockspergrid = math.ceil(len(times_min) / threadsperblock)
                        
                        scan_trades_cuda[blockspergrid, threadsperblock](
                            d_times, d_total, d_index, 
                            entry_min, exit_min, 
                            stoploss_price, profit_target_price,
                            index_upper_limit, index_lower_limit,
                            d_res
                        )
                        
                        res_val = d_res.copy_to_host()[0]
                        if res_val != 999999 and res_val < len(times_min):
                            idx = res_val
                            # Recalculate reason on CPU to avoid complex kernel atomic logic
                            # Since we have the index, check logic is O(1)
                            row_t = times_min[idx]
                            row_tc = total_closes[idx]
                            row_ic = index_closes[idx]
                            if row_tc >= stoploss_price: reason_code = 1
                            elif row_tc <= profit_target_price: reason_code = 2
                            elif row_ic >= index_upper_limit or row_ic <= index_lower_limit: reason_code = 3
                            elif row_t >= exit_min: reason_code = 4
                            
                    except Exception as e:
                        print(f"CUDA Error (falling back to CPU): {e}")
                        idx = -1 # Trigger fallback
                
                # --- CPU FALLBACK (If CUDA skipped or failed or unavailable) ---
                if idx == -1 and reason_code == 0:
                     close_pes = monitoring_data['close_pe'].values.astype(np.float64) # Not used in scan but needed for signature
                     close_ces = monitoring_data['close_ce'].values.astype(np.float64)
                     idx, reason_code = scan_trades_fast(
                        times_min, total_closes, index_closes, close_pes, close_ces,
                        entry_min, exit_min,
                        stoploss_price, profit_target_price,
                        index_upper_limit, index_lower_limit
                    )
                
                if idx != -1:
                    row = monitoring_data.iloc[idx]
                    exit_price = row['total_close']
                    pe_exit_price = row['close_pe']
                    ce_exit_price = row['close_ce']
                    exit_time_actual = row['time']
                    exit_index_close = row['index_close']
                    
                    reasons = {1: 'Stoploss Hit', 2: 'Profit Target Hit', 3: 'Index Movement Breach', 4: 'Time-Based Exit'}
                    exit_reason = reasons.get(reason_code, 'Unknown')
                else:
                    # End of Data Exit
                    last_row = monitoring_data.iloc[-1]
                    exit_reason = 'End of Day Exit'
                    exit_price = last_row['total_close']
                    pe_exit_price = last_row['close_pe']
                    ce_exit_price = last_row['close_ce']
                    exit_time_actual = last_row['time']
                    exit_index_close = last_row['index_close']

            else:
                # Fallback / Slow Path (Itertuples for better speed than iterrows)
                for row in monitoring_data.itertuples():
                    if row.time == entry_time: continue
                    
                    check = False
                    if row.total_close >= stoploss_price:
                        exit_reason = 'Stoploss Hit'
                        check = True
                    elif row.total_close <= profit_target_price:
                        exit_reason = 'Profit Target Hit'
                        check = True
                    elif row.index_close >= index_upper_limit or row.index_close <= index_lower_limit:
                        exit_reason = 'Index Movement Breach'
                        check = True
                    elif row.time >= exit_time:
                        exit_reason = 'Time-Based Exit'
                        check = True
                    
                    if check:
                        exit_price = row.total_close
                        pe_exit_price = row.close_pe
                        ce_exit_price = row.close_ce
                        exit_time_actual = row.time
                        exit_index_close = row.index_close
                        break
                
                if exit_reason is None:
                    # Force close at end
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

def execute_interday_strategy(strategy_row, backtest_data, strategy_config, filename, strategy_name,
                               entry_dte=None, entry_time='09:15', exit_dte=None, exit_time='15:20'):
    
    all_trades = []
    all_dtes = sorted(backtest_data['dte_format'].dropna().unique(), reverse=True)
    
    if entry_dte is None: entry_dte = strategy_row['DTE']
    if exit_dte is None or exit_dte == 'At Expiry': exit_dte = '0 DTE'
    
    def parse_dte(d_str):
        try: return int(str(d_str).split()[0])
        except: return -1

    req_entry_val = parse_dte(entry_dte)
    req_exit_val = parse_dte(exit_dte)
    
    start_index = -1
    for i, d in enumerate(all_dtes):
        if parse_dte(d) <= req_entry_val:
            start_index = i
            break
            
    end_index = -1
    for i in range(len(all_dtes)-1, -1, -1):
        if parse_dte(all_dtes[i]) >= req_exit_val:
            end_index = i
            break
            
    if start_index == -1 or end_index == -1 or start_index > end_index:
        return all_trades

    current_dte_index = start_index
    current_time_cursor = entry_time
    continue_trading = True
    trade_number = 1
    
    while continue_trading and current_dte_index <= end_index:
        current_dte = all_dtes[current_dte_index]
        
        matching_strategy = strategy_config[strategy_config['DTE'] == current_dte]
        if matching_strategy.empty:
            if strategy_row['DTE'] == current_dte:
                entry_strategy = strategy_row
            else:
                entry_strategy = strategy_row
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
            current_time_cursor = '09:15'
            continue
            
        dte_data = dte_data.sort_values(['date_only', 'time'])
        
        candidate_data = dte_data[
            (dte_data['time'] >= current_time_cursor) & 
            (dte_data['index_close'].notna())
        ]
        
        if candidate_data.empty:
            current_dte_index += 1
            current_time_cursor = '09:15'
            continue
            
        entry_row = candidate_data.iloc[0]
        actual_entry_time = entry_row['time']
        actual_entry_date = entry_row['date_only']
        
        if current_dte == exit_dte and actual_entry_time >= exit_time:
            continue_trading = False
            break
            
        index_close_for_entry = entry_row['index_close']
        try:
             index_close_for_entry = round(index_close_for_entry / 50) * 50
             pe_strike = int(index_close_for_entry - (pe_strike_offset * 50))
             ce_strike = int(index_close_for_entry + (ce_strike_offset * 50))
             pe_symbol = f"{pe_strike}-PE"
             ce_symbol = f"{ce_strike}-CE"
        except:
             current_dte_index += 1
             continue
        
        pe_entry_row = candidate_data[(candidate_data['time'] == actual_entry_time) & (candidate_data['symbol'].str.contains(pe_symbol, na=False))]
        ce_entry_row = candidate_data[(candidate_data['time'] == actual_entry_time) & (candidate_data['symbol'].str.contains(ce_symbol, na=False))]
        
        if pe_entry_row.empty or ce_entry_row.empty:
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
        
        stoploss_price = total_entry_price * (1 + stoploss_pct / 100)
        profit_target_price = total_entry_price * (1 - profit_target_pct / 100)
        index_upper_limit = entry_index_close + index_movement
        index_lower_limit = entry_index_close - index_movement
        
        trade_exited = False
        exit_reason = None
        
        for mon_dte_idx in range(current_dte_index, end_index + 1):
            mon_dte = all_dtes[mon_dte_idx]
            mon_data = backtest_data[backtest_data['dte_format'] == mon_dte].copy()
            if mon_data.empty: continue
            
            if mon_dte_idx == current_dte_index:
                mon_data = mon_data[mon_data['time'] > actual_entry_time]
            
            if mon_data.empty: continue
            
            mon_data = mon_data.sort_values(['date_only', 'time'])
            
            pe_mon = mon_data[mon_data['symbol'].str.contains(pe_symbol, na=False)]
            ce_mon = mon_data[mon_data['symbol'].str.contains(ce_symbol, na=False)]
            
            if pe_mon.empty or ce_mon.empty: continue
            
            full_mon = pd.merge(pe_mon[['date_only', 'time', 'close', 'index_close']],
                                ce_mon[['date_only', 'time', 'close']],
                                on=['date_only', 'time'], suffixes=('_pe', '_ce'))
            
            full_mon['total_close'] = full_mon['close_pe'] + full_mon['close_ce']
            full_mon = full_mon.sort_values(['date_only', 'time'])
            
            if NUMBA_AVAILABLE and not full_mon.empty:
                # Fast Path
                times_min = full_mon['time'].apply(time_to_min).values
                total_closes = full_mon['total_close'].values.astype(np.float64)
                index_closes = full_mon['index_close'].values.astype(np.float64)
                    
                entry_min = -1 
                exit_min = time_to_min(exit_time) 
                active_exit_min = exit_min if mon_dte == exit_dte else 9999
                
                idx = -1
                reason_code = 0
                
                if CUDA_AVAILABLE and len(times_min) > 50:
                    try:
                         import math
                         d_times = cuda.to_device(times_min)
                         d_total = cuda.to_device(total_closes)
                         d_index = cuda.to_device(index_closes)
                         d_res = cuda.to_device(np.array([999999], dtype=np.int32))
                        
                         threadsperblock = 256
                         blockspergrid = math.ceil(len(times_min) / threadsperblock)
                         
                         scan_trades_cuda[blockspergrid, threadsperblock](
                            d_times, d_total, d_index, 
                            entry_min, active_exit_min, 
                            stoploss_price, profit_target_price,
                            index_upper_limit, index_lower_limit,
                            d_res
                         )
                         res_val = d_res.copy_to_host()[0]
                         if res_val != 999999 and res_val < len(times_min):
                            idx = res_val
                            # Recalculate reason
                            row_t = times_min[idx]
                            row_tc = total_closes[idx]
                            row_ic = index_closes[idx]
                            if row_tc >= stoploss_price: reason_code = 1
                            elif row_tc <= profit_target_price: reason_code = 2
                            elif row_ic >= index_upper_limit or row_ic <= index_lower_limit: reason_code = 3
                            elif row_t >= active_exit_min: reason_code = 4
                    except Exception as e:
                         # print(f"CUDA Error: {e}") 
                         idx = -1 

                if idx == -1 and reason_code == 0:
                     close_pes = full_mon['close_pe'].values.astype(np.float64)
                     close_ces = full_mon['close_ce'].values.astype(np.float64)
                     idx, reason_code = scan_trades_fast(
                        times_min, total_closes, index_closes, close_pes, close_ces,
                        entry_min, active_exit_min,
                        stoploss_price, profit_target_price,
                        index_upper_limit, index_lower_limit
                    )
                
                if idx != -1:
                    row = full_mon.iloc[idx]
                    reasons = {1: 'Stoploss Hit', 2: 'Profit Target Hit', 3: 'Index Movement Breach', 4: 'Global Exit Time Reached'}
                    exit_reason = reasons.get(reason_code, 'Unknown')
                    
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
                    
                    current_dte_index = mon_dte_idx
                    current_time_cursor = row['time']
                    break

            else:
                # Slow Path
                for idx, row in full_mon.iterrows():
                    if mon_dte == exit_dte and row['time'] >= exit_time:
                        exit_reason = 'Global Exit Time Reached'
                        trade_exited = True
                        continue_trading = False 
                        # Note: original code set continue_trading=False here. 
                        # To match that behavior, if we exit due to time on exit day, we notify outer loop?
                        # In fast path `trade_exited = True` breaks the inner loop. 
                        # Outer loop logic: `if trade_exited: break`.
                        # Then `if not trade_exited: continue_trading = False`.
                        # Wait, logic check:
                        # Original: sets `continue_trading = False` inside loop.
                        # Fast path: sets `trade_exited = True`, breaks inner.
                        # Then outer checks trade_exited. It breaks DTE loop.
                        # Then the while loop checks `continue_trading`. 
                        # If fast path hit Time Exit, we need to ensure `continue_trading` becomes False if it was the FINAL exit.
                        # `mon_dte == exit_dte` check handles that. 
                        # If we exited on exit_dte, we are done.
                        # If we exited on stoploss (prior to exit dte), we continue to next trade setup?
                        # Yes, trade_number += 1.
                        pass # Handled by setting variables below
                    elif row['total_close'] >= stoploss_price:
                        exit_reason = 'Stoploss Hit'
                    elif row['total_close'] <= profit_target_price:
                        exit_reason = 'Profit Target Hit'
                    elif row['index_close'] >= index_upper_limit or row['index_close'] <= index_lower_limit:
                        exit_reason = 'Index Movement Breach'
                    
                    if exit_reason:
                        if exit_reason == 'Global Exit Time Reached':
                             continue_trading = False
                        
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
                        
                        current_dte_index = mon_dte_idx
                        current_time_cursor = row['time']
                        break
            
            if trade_exited:
                break
                
        if not trade_exited:
            continue_trading = False 
            
    return all_trades

def execute_strategy(strategy_row, backtest_data, strategy_config, filename, strategy_name,
                     mode='Weekly', entry_time='09:15', exit_time='15:20', 
                     entry_dte=None, exit_dte=None, entry_exit_times=None):
    if mode == 'Intraday':
        if entry_exit_times is None:
            dtes = ['4 DTE', '3 DTE', '2 DTE', '1 DTE', '0 DTE']
            entry_exit_times = {dte: {'entry': entry_time, 'exit': exit_time} for dte in dtes}
            
        return execute_intraday_strategy(strategy_row, backtest_data, strategy_config, filename, strategy_name,
                                          entry_exit_times, entry_dte or strategy_row['DTE'], exit_dte or '0 DTE')
    else:
        return execute_interday_strategy(strategy_row, backtest_data, strategy_config, filename, strategy_name,
                                         entry_dte=entry_dte, entry_time=entry_time, exit_dte=exit_dte, exit_time=exit_time)

# ============================================================================
# MULTI-STRATEGY PARALLEL EXECUTION
# ============================================================================

def process_file_strategies_wrapper(args):
    """
    Worker function for parallel processing.
    args: (file_identifier, file_source, strategies_list, run_params)
    file_source: can be a DataFrame or a Path string.
    """
    file_identifier, file_source, strategies_list, run_params = args
    results = {}
    
    try:
        # Load data once per file per worker
        backtest_data = preprocess_data(file_source)
        
        for strategy_name, strategy_config in strategies_list:
            # Extract params
            mode = run_params['mode']
            entry_dte = run_params['entry_dte']
            entry_time = run_params['entry_time']
            exit_time = run_params['exit_time']
            exit_dte = run_params['exit_dte']
            entry_exit_times = run_params['entry_exit_times']
            
            entry_strategy_row = strategy_config[strategy_config['DTE'] == entry_dte]
            if entry_strategy_row.empty:
                entry_strategy_row = strategy_config.iloc[[0]]
            else:
                entry_strategy_row = entry_strategy_row.iloc[0]
            
            res = execute_strategy(entry_strategy_row, backtest_data, strategy_config, 
                                   file_identifier, strategy_name,
                                   mode = mode,
                                   entry_time=entry_time, exit_time=exit_time, 
                                   entry_dte=entry_dte, exit_dte=exit_dte,
                                   entry_exit_times=entry_exit_times)
            
            results[strategy_name] = res
            
    except Exception as e:
        print(f"Error processing {file_identifier}: {e}")
        # Return empty list for strategies on failure
        for strategy_name, _ in strategies_list:
             results[strategy_name] = []
             
    return file_identifier, results

def run_multi_strategy_backtest(files_dict, strategies_dict, progress_callback=None, stop_callback=None,
                                 mode='Weekly', entry_time='09:15', exit_time='15:20', 
                                 entry_dte='4 DTE', exit_dte='0 DTE', entry_exit_times=None):
    
    # Structure for final aggregation
    all_results = {s: {} for s in strategies_dict.keys()}
    
    strategies_list = list(strategies_dict.items())
    run_params = {
        'mode': mode,
        'entry_time': entry_time,
        'exit_time': exit_time,
        'entry_dte': entry_dte,
        'exit_dte': exit_dte,
        'entry_exit_times': entry_exit_times
    }
    
    tasks = []
    for filename, source in files_dict.items():
        tasks.append((filename, source, strategies_list, run_params))
        
    total_files = len(tasks)
    completed_files = 0
    start_time = time.time()
    
    # Determine max workers
    max_workers = min(os.cpu_count(), len(tasks)) or 1
    print(f"🚀 Starting Parallel Execution with {max_workers} workers on {total_files} files...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_file_strategies_wrapper, task): task[0] for task in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            fname = futures[future]
            completed_files += 1
            
            if stop_callback and stop_callback():
                print("🛑 Stopping execution...")
                executor.shutdown(wait=False, cancel_futures=True)
                break
                
            try:
                processed_fname, file_results = future.result()
                
                # Aggregate results back
                for strat, trades in file_results.items():
                    all_results[strat][processed_fname] = trades
                    
            except Exception as e:
                print(f"❌ Worker Error on {fname}: {e}")
            
            if progress_callback:
                elapsed = time.time() - start_time
                if completed_files > 0:
                    avg_time = elapsed / completed_files
                    remaining_sec = avg_time * (total_files - completed_files)
                    mins, secs = divmod(int(remaining_sec), 60)
                    time_str = f"{mins}m {secs}s"
                else:
                    time_str = "..."
                
                pct = int((completed_files / total_files) * 100)
                progress_callback(f"Processed {fname} ({completed_files}/{total_files}) | ETC: {time_str}", pct)
                
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
    """Calculate comprehensive metrics for a strategy"""
    
    strategy_df = results_df[results_df['Strategy'] == strategy_name]
    
    if len(strategy_df) == 0:
        return None
    
    total_pnl = strategy_df['PnL'].sum()
    avg_pnl = strategy_df['PnL'].mean()
    winning_trades = len(strategy_df[strategy_df['PnL'] > 0])
    losing_trades = len(strategy_df[strategy_df['PnL'] < 0])
    win_rate = (winning_trades / len(strategy_df) * 100) if len(strategy_df) > 0 else 0
    
    avg_win = strategy_df[strategy_df['PnL'] > 0]['PnL'].mean() if winning_trades > 0 else 0
    avg_loss = strategy_df[strategy_df['PnL'] < 0]['PnL'].mean() if losing_trades > 0 else 0
    
    profit_factor = (strategy_df[strategy_df['PnL'] > 0]['PnL'].sum() / 
                     abs(strategy_df[strategy_df['PnL'] < 0]['PnL'].sum())) if losing_trades > 0 else float('inf')
    
    # Drawdown calculation
    if 'Entry_Date' in strategy_df.columns:
         strategy_sorted = strategy_df.sort_values(['Filename', 'Entry_Date', 'Entry_Time']).copy()
    else:
         strategy_sorted = strategy_df.copy() # Fallback order
         
    strategy_sorted['Cumulative_PnL'] = strategy_sorted['PnL'].cumsum()
    strategy_sorted['Running_Max'] = strategy_sorted['Cumulative_PnL'].cummax()
    strategy_sorted['Drawdown'] = strategy_sorted['Cumulative_PnL'] - strategy_sorted['Running_Max']
    max_drawdown = strategy_sorted['Drawdown'].min()
    
    # Sharpe ratio (simplified)
    sharpe = (avg_pnl / strategy_df['PnL'].std()) if strategy_df['PnL'].std() > 0 else 0
    
    return {
        'Strategy': strategy_name,
        'Total_Trades': len(strategy_df),
        'Winning_Trades': winning_trades,
        'Losing_Trades': losing_trades,
        'Win_Rate_%': round(win_rate, 2),
        'Total_PnL': round(total_pnl, 2),
        'Avg_PnL': round(avg_pnl, 2),
        'Avg_Win': round(avg_win, 2),
        'Avg_Loss': round(avg_loss, 2),
        'Best_Trade': round(strategy_df['PnL'].max(), 2),
        'Worst_Trade': round(strategy_df['PnL'].min(), 2),
        'Profit_Factor': round(profit_factor, 2) if profit_factor != float('inf') else 'Inf',
        'Max_Drawdown': round(max_drawdown, 2),
        'Sharpe_Ratio': round(sharpe, 2),
        'Files_Processed': strategy_df['Filename'].nunique()
    }

def generate_strategy_comparison(results_df):
    """Generate comparison metrics across all strategies"""
    
    strategies = results_df['Strategy'].unique()
    comparison_data = []
    
    for strategy in strategies:
        metrics = calculate_strategy_metrics(results_df, strategy)
        if metrics:
            comparison_data.append(metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Rank strategies
    comparison_df['Rank_by_PnL'] = comparison_df['Total_PnL'].rank(ascending=False)
    comparison_df['Rank_by_WinRate'] = comparison_df['Win_Rate_%'].rank(ascending=False)
    comparison_df['Rank_by_Sharpe'] = comparison_df['Sharpe_Ratio'].rank(ascending=False)
    
    comparison_df['Rank_by_ProfitFactor'] = comparison_df['Profit_Factor'].apply(
        lambda x: float(x) if x != 'Inf' else 999
    ).rank(ascending=False)
    
    comparison_df['Overall_Rank_Score'] = (
        comparison_df['Rank_by_PnL'] + 
        comparison_df['Rank_by_WinRate'] + 
        comparison_df['Rank_by_Sharpe'] + 
        comparison_df['Rank_by_ProfitFactor']
    ) / 4
    
    comparison_df = comparison_df.sort_values('Overall_Rank_Score')
    
    return comparison_df

def export_comprehensive_excel(results_df, strategies_dict, output_filename, mode='Weekly'):
    EXCEL_ROW_LIMIT = 900000 
    
    if len(results_df) > EXCEL_ROW_LIMIT:
        csv_filename = output_filename.replace('.xlsx', '.csv')
        if not csv_filename.endswith('.csv'): csv_filename += '.csv'
        
        print(f"⚠️ Warning: Dataset has {len(results_df)} rows, exceeding Excel limit.")
        print(f"🔄 Switching to CSV format...")
        
        results_df.to_csv(csv_filename, index=False)
        print(f"✅ Saved Results to: {csv_filename}")
        
        config_filename = output_filename.replace('.xlsx', '_Configs.xlsx')
        config_rows = []
        for name, config in strategies_dict.items():
            c = config.copy()
            c.insert(0, 'Strategy', name)
            config_rows.append(c)
        pd.concat(config_rows).to_excel(config_filename, index=False)
        print(f"📁 Strategy Configs saved to: {config_filename}")
        
        return

    print(f"\\n📊 Generating comprehensive Excel report: {output_filename}")
    
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        comparison_df = generate_strategy_comparison(results_df)
        comparison_df.to_excel(writer, sheet_name='Strategy_Comparison', index=False)
        print("   ✓ Sheet 1: Strategy_Comparison (Rankings)")
        
        all_trades_export = results_df.copy()
        if 'Entry_Date' in all_trades_export.columns:
            all_trades_export['Week_Number'] = pd.to_datetime(all_trades_export['Entry_Date']).dt.isocalendar().week
            all_trades_export['Month'] = pd.to_datetime(all_trades_export['Entry_Date']).dt.to_period('M').astype(str)
            all_trades_export = all_trades_export.sort_values(['Strategy', 'Filename', 'Entry_Date', 'Entry_Time'])
        
        all_trades_export.to_excel(writer, sheet_name='All_Trades', index=False)
        print("   ✓ Sheet 2: All_Trades")
        
        config_rows = []
        for strategy_name, config_df in strategies_dict.items():
            config_with_name = config_df.copy()
            config_with_name.insert(0, 'Strategy', strategy_name)
            config_rows.append(config_with_name)
        
        if config_rows:
            all_configs = pd.concat(config_rows, ignore_index=True)
            all_configs.to_excel(writer, sheet_name='Strategy_Configurations', index=False)
            print("   ✓ Sheet 3: Strategy_Configurations")
        
        if 'Exit_Reason' in results_df.columns:
            exit_analysis = results_df.groupby(['Strategy', 'Exit_Reason']).agg({
                'PnL': ['sum', 'mean', 'count']
            }).round(2)
            exit_analysis.columns = ['Total_PnL', 'Avg_PnL', 'Count']
            exit_analysis = exit_analysis.reset_index()
            exit_analysis.to_excel(writer, sheet_name='Exit_Reason_Analysis', index=False)
            print(f"   ✓ Sheet 4: Exit_Reason_Analysis")
        
        dte_col = 'Entry_DTE' if 'Entry_DTE' in results_df.columns else 'DTE'
        if dte_col in results_df.columns:
            dte_analysis = results_df.groupby(['Strategy', dte_col]).agg({
                'PnL': ['sum', 'mean', 'std', 'count']
            }).round(2)
            dte_analysis.columns = ['Total_PnL', 'Avg_PnL', 'Std_Dev', 'Count']
            dte_analysis = dte_analysis.reset_index()
            dte_analysis.to_excel(writer, sheet_name='DTE_Analysis', index=False)
            print(f"   ✓ Sheet 5: DTE_Analysis")

        dictionary_data = [
            {'Term': 'Total_PnL', 'Description': 'Sum of Profit/Loss from all trades', 'Calculation': 'SUM(PnL)'},
            {'Term': 'Avg_PnL', 'Description': 'Average Profit/Loss per trade', 'Calculation': 'Total_PnL / Total_Trades'},
            {'Term': 'Win_Rate_%', 'Description': 'Percentage of trades that resulted in a profit', 'Calculation': '(Winning_Trades / Total_Trades) * 100'},
            {'Term': 'Profit_Factor', 'Description': 'Ratio of Gross Profit to Gross Loss. Measures payout efficiency. >1.5 is good.', 'Calculation': 'Gross_Profit / ABS(Gross_Loss)'},
            {'Term': 'Sharpe_Ratio', 'Description': 'Risk-adjusted return metric. Higher is better.', 'Calculation': 'Avg_PnL / Std_Dev(PnL)'},
            {'Term': 'Max_Drawdown', 'Description': 'Maximum peak-to-valley decline in cumulative PnL during the period.', 'Calculation': 'MIN(Cumulative_PnL - Running_Max_PnL)'},
            {'Term': 'Cumulative_PnL', 'Description': 'Running total of PnL over time, showing the equity curve.', 'Calculation': 'Running Sum of PnL'},
            {'Term': 'Rank_by_[Metric]', 'Description': 'Ranking of strategy based on the specific metric (1 is best).', 'Calculation': 'RANK(Metric, Ascending/Descending)'},
            {'Term': 'Overall_Rank_Score', 'Description': 'Composite score derived from average of all rankings. Lower is better.', 'Calculation': 'AVG(Rank_PnL, Rank_WinRate, Rank_Sharpe, Rank_PF)'}
        ]
        pd.DataFrame(dictionary_data).to_excel(writer, sheet_name='Data_Dictionary', index=False)
        print(f"   ✓ Sheet 6: Data_Dictionary")
    
    print(f"\\n✅ Excel report generated successfully!")
