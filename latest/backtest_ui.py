
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import io
from datetime import datetime
import backtest_engine as engine

def create_enhanced_file_upload_ui():
    header = widgets.HTML('''
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0;">📈 Options Backtest System</h1>
            <p style="color: #e0e0e0; margin: 10px 0 0 0;">
                Multi-Strategy | Intraday & Interday Modes | Parallel Execution
            </p>
        </div>
    ''')
    upload_widget = widgets.FileUpload(accept='.csv,.xlsx,.xls', multiple=True, description='Upload Files')
    
    folder_path_input = widgets.Text(placeholder='Enter Folder Path', description='Folder Path:')
    load_folder_btn = widgets.Button(description='Load from Folder', button_style='info')
    
    file_list_output = widgets.Output()
    
    processed_files = set()

    def on_upload_change(change):
        with file_list_output:
            # Do not clear output, append instead
            for f in change['new']:
                fname = f['name']
                
                # Logic: Ensure file is in engine state (in case of reload wipe)
                # But only LOG if we haven't logged it for this UI instance yet.
                
                if fname not in engine.state.uploaded_files:
                    try:
                        content = f['content']
                        df = pd.read_csv(io.BytesIO(content)) if fname.endswith('.csv') else pd.read_excel(io.BytesIO(content))
                        engine.state.add_file(fname, df)
                        
                        if fname not in processed_files:
                            print(f"✅ Uploaded: {fname}")
                            processed_files.add(fname)
                            
                    except Exception as e:
                        print(f"❌ Error processing {fname}: {e}")
                else:
                    # Already in engine. Mark as processed in UI to be safe.
                    if fname not in processed_files:
                        # It was in engine (maybe from folder load or previous session) 
                        # but new to this upload widget interaction?
                        # If widget sends it, and it's in engine, we probably already logged it?
                        # Or maybe engine persisted but widget is new?
                        # Let's not log to be conservative and avoid duplicates.
                        processed_files.add(fname)
                    
    def on_load_folder_click(b):
        with file_list_output:
            # clear_output() # Removed to preserve history
            try:
                path = folder_path_input.value
                loaded_paths = engine.load_files_from_folder(path)
                for name, fpath in loaded_paths.items():
                    engine.state.add_file(name, fpath)
                print(f"✅ Registered {len(loaded_paths)} files from {path}")
                print("   (Lazy Loading enabled for performance)")
            except Exception as e:
                print(f"❌ Error loading folder: {str(e)}")
    
    upload_widget.observe(on_upload_change, names='value')
    load_folder_btn.on_click(on_load_folder_click)
    
    return widgets.VBox([
        header,
        widgets.HTML("<h3>Step 1: Load Data</h3>"),
        widgets.HBox([widgets.Label("Method A: File Upload"), upload_widget]),
        widgets.HBox([widgets.Label("Method B: Folder Load"), folder_path_input, load_folder_btn]),
        file_list_output
    ])

def create_strategy_selection_ui():
    # === OPTION A: MANUAL UI ===
    preset_checkboxes = {}
    for preset_name in engine.STRATEGY_PRESETS.keys():
        preset_checkboxes[preset_name] = widgets.Checkbox(value=False, description=preset_name)
    
    manual_ui = widgets.VBox([
        widgets.HTML("<h4>Option A: Select Presets (Manual)</h4>"),
        widgets.VBox([widgets.HBox([cb]) for cb in preset_checkboxes.values()])
    ])
    
    # === OPTION B: EXCEL UPLOAD ===
    excel_upload = widgets.FileUpload(accept='.xlsx', description='Upload Strategy config')
    excel_status = widgets.Output()
    
    def on_excel_upload_change(change):
        with excel_status:
            clear_output()
            if change['new']:
                 uploaded_filename = next(iter(change['new']))['name']
                 print(f"✅ Strategy Excel '{uploaded_filename}' Uploaded Successfully!")
                 print("   (Click 'Load/Parse Configs' to process it)")

    excel_upload.observe(on_excel_upload_change, names='value')
    
    excel_ui = widgets.VBox([
        widgets.HTML("<h4>Option B: Upload Strategy Excel</h4>"),
        excel_upload, excel_status
    ])
    
    # === OPTION C: GENERATOR ===
    gen_params = [
        ('Stoploss %', 70, 100, 5),
        ('Profit Target %', 35, 55, 5),
        ('lot_size', 1, 1, 1),
        ('PE Strike', 2, 4, 1),
        ('CE Strike', 2, 4, 1),
        ('Index Movement +/-', 100, 300, 50)
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
        common_row = widgets.HBox([
            widgets.Label(label, layout=widgets.Layout(width='150px')),
            w_start, w_end, w_step
        ])
        
    max_combos = widgets.IntText(value=1000000, description='Max Combos:')
    
    gen_mode_toggle = widgets.ToggleButtons(
        options=['Weekly', 'DTE'],
        description='Gen Mode:',
        value='Weekly',
        button_style='warning',
        tooltips=['Weekly: Same Strategy ID for all DTEs', 'DTE: Unique Strategy ID per DTE']
    )

    gen_action_btn = widgets.Button(description='Generate & Save Excel', button_style='info')
    gen_status = widgets.Output()
    
    # DTE Table Widgets construction
    dte_table_widgets = {}
    dte_rows = []
    
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
    ], layout=widgets.Layout(border='1px solid #ccc', padding='5px', display='none'))
    
    # Splitting parameters for display
    common_params = []
    pe_ce_params = []
    
    for label, w_dict in gen_widgets.items():
        row_box = widgets.HBox([
            widgets.Label(label, layout=widgets.Layout(width='150px')),
            w_dict['start'], w_dict['end'], w_dict['step']
        ])
        if label in ['PE Strike', 'CE Strike']:
            pe_ce_params.append(row_box)
        else:
            common_params.append(row_box)
            
    common_container = widgets.VBox(common_params)
    global_pe_ce_container = widgets.VBox(pe_ce_params)
    
    def on_gen_mode_change(change):
        if change['new'] == 'DTE':
            dte_table_container.layout.display = 'block'
            global_pe_ce_container.layout.display = 'none'
        else:
            dte_table_container.layout.display = 'none'
            global_pe_ce_container.layout.display = 'block'
            
    gen_mode_toggle.observe(on_gen_mode_change, names='value')

    def on_gen_click(b):
        with gen_status:
            clear_output()
            try:
                mode = gen_mode_toggle.value
                
                ranges = {}
                for label, w_dict in gen_widgets.items():
                    start = w_dict['start'].value
                    end = w_dict['end'].value
                    step = w_dict['step'].value
                    ranges[label] = [start, end, step]
                
                dte_specific_ranges = None
                if mode == 'DTE':
                    dte_specific_ranges = {}
                    for dte, w_row in dte_table_widgets.items():
                        pe_s = w_row['pe_start'].value
                        pe_e = w_row['pe_end'].value
                        pe_st = w_row['pe_step'].value
                        ce_s = w_row['ce_start'].value
                        ce_e = w_row['ce_end'].value
                        ce_st = w_row['ce_step'].value
                        dte_specific_ranges[dte] = {
                            'PE Strike': [pe_s, pe_e, pe_st],
                            'CE Strike': [ce_s, ce_e, ce_st]
                        }
                
                # IMPORTANT: need to import helper from engine or implement here?
                # Helper is `generate_cartesian_strategies` which I did NOT include in engine export list but put in engine file.
                # Assuming engine has it.
                # Wait, I did include it in engine.
                
                # I need to implement generate_cartesian_strategies inside engine or here.
                # It was a standalone function in notebook. I likely put it in engine?
                # Let me check my memory. I put `generate_strategy_combinations` but NOT `generate_cartesian_strategies` in `backtest_engine.py`.
                # Oops. `generate_cartesian_strategies` is for the Generator UI.
                # I will implement it here in UI file or add to engine. 
                # Better to put here if it's UI specific, but it generates data.
                
                # I'll implement it here for now to avoid re-writing engine again.
                df_gen = generate_cartesian_strategies(ranges, mode=mode, dte_specific_ranges=dte_specific_ranges)
                
                if len(df_gen) > 900000:
                    fname = f"Generated_Strategies_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    df_gen.to_csv(fname, index=False)
                    print(f"⚠️ Warning: Large dataset ({len(df_gen)} rows). Switched to CSV.")
                else:
                    fname = f"Generated_Strategies_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    with pd.ExcelWriter(fname) as writer:
                        df_gen.to_excel(writer, index=False)
                
                print(f"✅ Success! Generated {len(df_gen)} rows.")
                print(f"📁 Saved to: {fname}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    gen_action_btn.on_click(on_gen_click)

    generator_ui = widgets.VBox([
        widgets.HTML("<h4>Option C: Generate Strategy Excel</h4>"),
        gen_mode_toggle,
        widgets.VBox([widgets.HBox([
             widgets.Label("Parameter", layout=widgets.Layout(width='150px', font_weight='bold')),
             widgets.Label("Start", layout=widgets.Layout(width='80px', font_weight='bold')),
             widgets.Label("End", layout=widgets.Layout(width='80px', font_weight='bold')),
             widgets.Label("Step", layout=widgets.Layout(width='80px', font_weight='bold'))
        ])]),
        common_container,
        global_pe_ce_container,
        dte_table_container,
        widgets.HBox([max_combos]),
        widgets.HTML("<br>"),
        gen_action_btn,
        gen_status
    ])
    
    tabs = widgets.Tab(children=[manual_ui, excel_ui, generator_ui])
    tabs.set_title(0, 'Manual Presets')
    tabs.set_title(1, 'Upload Excel')
    tabs.set_title(2, 'Generator')
    
    generate_config_btn = widgets.Button(description='🔧 Load/Parse Configs', button_style='success')
    config_output = widgets.Output()
    
    def on_load_config_click(b):
        with config_output:
            clear_output()
            if engine.state.get_file_count() == 0:
                print("❌ Please upload data files first!")
                return
            
            # Use ANY uploaded file for base config (just need DTEs/Index Close sample)
            first_filename = engine.state.get_filenames()[0]
            # Need to preprocess ONE file to get base config structure
            file_source = engine.state.uploaded_files[first_filename]
            processed = engine.preprocess_data(file_source)
            engine.state.processed_data[first_filename] = processed # Cache it
            
            base_config = engine.create_base_strategy_config(processed)
            
            active_tab = tabs.selected_index
            strategies = {}
            
            if active_tab == 0: # Manual
                selected = [name for name, cb in preset_checkboxes.items() if cb.value]
                if not selected:
                    print("❌ Please select a strategy!")
                    return
                selected_presets_with_params = {}
                for name in selected:
                    selected_presets_with_params[name] = engine.STRATEGY_PRESETS[name] 
                strategies = engine.generate_strategy_combinations(base_config, selected_presets_with_params)
                
            elif active_tab == 1: # Excel
                if not excel_upload.value:
                    print("❌ Please upload a strategy Excel file!")
                    return
                uploaded_file = excel_upload.value[0]
                strategies = engine.parse_strategy_excel(uploaded_file['content'], base_config)
            
            else: # Generator
                 print("⚠️ Please generate the Excel first, then upload it in the 'Upload Excel' tab.")
                 return
            
            for name, config in strategies.items():
                engine.state.add_strategy_config(name, config)
            
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

# Helper for generator
def generate_cartesian_strategies(ranges_dict, dtes=['4 DTE', '3 DTE', '2 DTE', '1 DTE', '0 DTE'], mode='Weekly', dte_specific_ranges=None):
    import itertools
    import numpy as np
    
    def expand_range(start, end, step):
        if start == end: return [start]
        if step == 0: step = 1
        vals = np.arange(start, end + step/1000, step).tolist()
        return [round(x, 2) if isinstance(x, float) else x for x in vals]

    global_params = {}
    for param, (start, end, step) in ranges_dict.items():
        if mode == 'DTE' and dte_specific_ranges and param in ['PE Strike', 'CE Strike']:
            continue
        global_params[param] = expand_range(start, end, step)
    
    global_keys = list(global_params.keys())
    global_values = list(global_params.values())
    global_combinations = list(itertools.product(*global_values))
    
    print(f"Generating strategies (Mode: {mode}, Base Combos: {len(global_combinations)})...")
    
    rows = []
    
    for i, global_combo in enumerate(global_combinations):
        base_strat_id = f"Strat_{i+1:04d}"
        global_dict = dict(zip(global_keys, global_combo))
        
        for dte in dtes:
            if mode == 'DTE':
                strat_id = base_strat_id
                current_dte_params = global_dict.copy()
                
                if dte_specific_ranges and dte in dte_specific_ranges:
                    overrides = dte_specific_ranges[dte]
                    override_params = {}
                    for p, (s, e, st) in overrides.items():
                        override_params[p] = expand_range(s, e, st)
                        
                    ov_keys = list(override_params.keys())
                    ov_vals = list(override_params.values())
                    ov_combos = list(itertools.product(*ov_vals))
                    
                    for j, ov_combo in enumerate(ov_combos):
                        final_dict = current_dte_params.copy()
                        final_dict.update(dict(zip(ov_keys, ov_combo)))
                        variant_strat_id = strat_id if len(ov_combos) == 1 else f"{strat_id}_v{j+1}"
                        row = {'Strategy': variant_strat_id, 'DTE': dte}
                        row.update(final_dict)
                        rows.append(row)
                else:
                    row = {'Strategy': strat_id, 'DTE': dte}
                    row.update(current_dte_params)
                    rows.append(row)
            else:
                strat_id = base_strat_id
                row = {'Strategy': strat_id, 'DTE': dte}
                row.update(global_dict)
                rows.append(row)
            
    return pd.DataFrame(rows)

def create_backtest_execution_ui():
    mode_toggle = widgets.ToggleButtons(
        options=['Weekly', 'Intraday'],
        description='Mode:',
        value='Weekly',
        button_style='primary'
    )
    
    dte_options = ['4 DTE', '3 DTE', '2 DTE', '1 DTE', '0 DTE']
    
    w_entry_dte = widgets.Dropdown(options=dte_options, value='4 DTE', layout=widgets.Layout(width='100px'))
    w_entry_time = widgets.Text(value='09:30', layout=widgets.Layout(width='100px'))
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
    
    settings_container = widgets.VBox([weekly_ui])
    
    def on_mode_change(change):
        if change['new'] == 'Weekly':
            settings_container.children = [weekly_ui]
        else:
            settings_container.children = [intraday_ui]
    
    mode_toggle.observe(on_mode_change, names='value')
    
    prog_bar = widgets.IntProgress(value=0, min=0, max=100, step=1, description='Progress:', bar_style='info', layout=widgets.Layout(width='300px'))
    prog_label = widgets.Label(value="")
    stop_btn = widgets.Button(description='🛑 Stop', button_style='danger', disabled=True)
    run_btn = widgets.Button(description='🚀 Run Backtest', button_style='success')
    output = widgets.Output()
    
    execution_state = {'stop': False}
    
    def on_stop_click(b):
        execution_state['stop'] = True
        prog_label.value = "Stopping..."
        stop_btn.disabled = True
        
    stop_btn.on_click(on_stop_click)
    
    def on_run_click(b):
        with output:
            clear_output()
            if not engine.state.strategy_configs:
                print("❌ No strategy configs! Please generate or upload strategies first.")
                return
            
            prog_bar.value = 0
            prog_label.value = "Starting..."
            stop_btn.disabled = False
            run_btn.disabled = True
            execution_state['stop'] = False
            
            mode = mode_toggle.value
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
                print(f"Running Weekly Mode | Entry: {entry_dte} @ {entry_time} -> Exit: {exit_dte} @ {exit_time}")
            else:
                entry_dte = '4 DTE'
                exit_dte = '0 DTE'
                entry_exit_times = {}
                for dte, w_dict in time_config_widgets.items():
                    e_t = w_dict['entry'].value.strip()
                    x_t = w_dict['exit'].value.strip()
                    if e_t and x_t:
                        entry_exit_times[dte] = {'entry': e_t, 'exit': x_t}
                print(f"Running Intraday Mode | Per-DTE Configuration")
            
            def update_progress(msg, pct):
                prog_label.value = msg
                prog_bar.value = pct
            
            def check_stop():
                return execution_state['stop']

            try:
                all_results = engine.run_multi_strategy_backtest(
                    engine.state.uploaded_files, 
                    engine.state.strategy_configs, 
                    progress_callback=update_progress,
                    stop_callback=check_stop,
                    mode=mode,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    entry_dte=entry_dte,
                    exit_dte=exit_dte,
                    entry_exit_times=entry_exit_times
                )
            except Exception as e:
                print(f"❌ Critical Error during execution: {e}")
                all_results = {}
            
            run_btn.disabled = False
            stop_btn.disabled = True
            
            if execution_state['stop']:
                print("\n⚠️ Execution stopped by user. Partial results will be processed.")
            
            results_df = engine.consolidate_results(all_results)
            
            if results_df is not None:
                engine.state.consolidated_results = results_df
                
                selected_entry_dte = entry_dte
                selected_entry_time = entry_time
                actual_exit_dte = exit_dte
                actual_exit_time = exit_time
                
                # Report Generation
                print("\n" + "="*80)
                print("📊 MULTI-STRATEGY BACKTEST COMPLETED")
                print("="*80)
                
                comparison_df = engine.generate_strategy_comparison(results_df)
                
                print("\n🏆 STRATEGY RANKINGS:")
                cols_to_show = ['Strategy', 'Total_PnL', 'Win_Rate_%', 'Profit_Factor', 'Sharpe_Ratio', 'Overall_Rank_Score']
                valid_cols = [c for c in cols_to_show if c in comparison_df.columns]
                print(comparison_df[valid_cols].to_string(index=False))
                
                if not comparison_df.empty:
                    print("\n📈 BEST PERFORMING STRATEGY:")
                    best_strategy = comparison_df.iloc[0]
                    print(f"Strategy: {best_strategy['Strategy']}")
                    print(f"Total P&L: ₹{best_strategy['Total_PnL']:,.2f}")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                entry_dte_clean = str(selected_entry_dte).replace(" ", "")
                entry_time_clean = str(selected_entry_time).replace(":", "")
                exit_dte_clean = str(actual_exit_dte).replace(" ", "")
                exit_time_clean = str(actual_exit_time).replace(":", "")
                
                excel_filename = f'Backtest_Entry{entry_dte_clean}_{entry_time_clean}_Exit{exit_dte_clean}_{exit_time_clean}_{timestamp}.xlsx'
                
                engine.export_comprehensive_excel(results_df, engine.state.strategy_configs, excel_filename, mode=mode)
                
                print(f"\n📊 Excel report: {excel_filename}")
            else:
                print("No trades generated.")
    
    run_btn.on_click(on_run_click)
    
    return widgets.VBox([
        widgets.HTML("<h3>Step 3: Execute</h3>"),
        mode_toggle,
        widgets.HTML("<hr>"),
        settings_container,
        widgets.HTML("<hr>"),
        widgets.HBox([run_btn, stop_btn]),
        prog_bar,
        prog_label,
        output
    ])

def launch():
    display(widgets.VBox([
        create_enhanced_file_upload_ui(),
        create_strategy_selection_ui(),
        create_backtest_execution_ui()
    ]))

if __name__ == '__main__':
    launch()
