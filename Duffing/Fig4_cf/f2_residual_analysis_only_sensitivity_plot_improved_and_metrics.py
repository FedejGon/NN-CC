import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import scipy.stats as stats
from io import StringIO
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
BIN_WIDTH = 0.03 
FILENAME = 'f2_residuals_data_10_augmentation.txt'
#OUTPUT_FOLDER = 'plots_results'
OUTPUT_FOLDER = './'

# --- FONT SIZES ---
LABEL_SIZE = 24         # Axis labels (x, y)
TICK_SIZE = 24          # Numbers on axes
LEGEND_SIZE = 24        # Legend text
CORNER_LABEL_SIZE = 28  # The (b), (c) text
METRIC_TEXT_SIZE = 18   # Size of the new metric text on plot

# --- DATASET SELECTION & LABELS ---
PLOTS_TO_GENERATE = [
    (0,  "(l)", "(m)"),  # Database 1
    (1,  "(h)", "(i)"),  # Database 2
    (4,  "(j)", "(k)"),  # Database 5
    #(49, "(l)", "(m)"),  # Database 50
]

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Enable LaTeX if available
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}"
    })
except:
    pass

# ==========================================
# 2. DATA PROCESSING
# ==========================================

def parse_stacked_data(filename, bin_width=0.03):
    """ Parses file into a list of DataFrames. """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return []

    datasets = []
    current_data = []
    current_header = None
    
    def process_block(header, data):
        if not data: return None
        cols = header.strip().replace('#', '').split()
        content = '\n'.join(data)
        try:
            df = pd.read_csv(StringIO(content), sep='\s+', names=cols, header=None)
            if not df.empty:
                df['x_binned'] = (df[df.columns[0]] / bin_width).round() * bin_width
                return df
        except:
            pass
        return None

    for line in lines:
        stripped = line.strip()
        if not stripped: continue
        if stripped.startswith('#'):
            if current_data:
                df = process_block(current_header, current_data)
                if df is not None: datasets.append(df)
                current_data = []
            current_header = stripped
        else:
            current_data.append(stripped)

    if current_data and current_header:
        df = process_block(current_header, current_data)
        if df is not None: datasets.append(df)

    return datasets

def get_ci_99(data): 
    a = 1.0 * np.array(data)
    n = len(a)
    if n < 2: return 0
    return stats.sem(a) * stats.t.ppf((1 + 0.99) / 2., n-1)

def calculate_global_stats(all_datasets_list):
    full_df = pd.concat(all_datasets_list)
    y_cols = full_df.columns[1:-1]
    grouped = full_df.groupby('x_binned')
    
    df_mean = grouped[y_cols].mean()
    df_min = grouped[y_cols].min()
    df_max = grouped[y_cols].max()
    df_ci = pd.DataFrame({c: grouped[c].apply(get_ci_99) for c in y_cols})
    
    return df_mean, df_min, df_max, df_ci

def get_shade_metrics(df_upper, df_lower, col_name, mode="range"):
    """
    Calculates the average vertical span of the shaded region.
    mode="range": computes mean(Upper - Lower)
    mode="ci": computes mean(CI_value), assuming input is just the CI magnitude
    """
    if mode == "range":
        # Average total height of the shade
        avg_span = (df_upper[col_name] - df_lower[col_name]).mean()
        return avg_span
    elif mode == "ci":
        # Average distance from mean (the CI radius)
        return df_upper[col_name].mean()
    return 0.0

# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================

def custom_ticks(ax, maj_x, maj_y, min_x, min_y):
    ax.xaxis.set_major_locator(ticker.MultipleLocator(maj_x))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(maj_y))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(min_x))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(min_y))
    
    ax.tick_params(axis='both', direction='in', which='major', length=8, width=1.5, labelsize=TICK_SIZE, top=True, right=True)
    ax.tick_params(axis='both', direction='in', which='minor', length=5, width=1, top=True, right=True)

def plot_single_frame(x, ys, labels, colors, y_label, 
                      maj_x=1.0, maj_y=0.02, min_x=0.5, min_y=0.01,
                      corner_label=None, y_lim=None, 
                      fill_bounds=None, fill_alpha=0.2,
                      metrics_text=None, # <--- NEW ARGUMENT
                      save_filename=None):
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for i, y_data in enumerate(ys):
        ax.plot(x, y_data, label=labels[i], ls='-', lw=2.5, color=colors[i])
        
        if fill_bounds is not None:
            lower, upper = fill_bounds[i]
            ax.fill_between(x, lower, upper, color=colors[i], alpha=fill_alpha)

    custom_ticks(ax, maj_x, maj_y, min_x, min_y)
    
    ax.set_xlabel(r"$x$", fontsize=LABEL_SIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc='best', frameon=False, labelspacing=0.05)
    ax.axhline(0, color='black', lw=1.5, ls='--', alpha=0.6, dashes=(4, 6))
    
    # Corner Label (e.g., (a), (b))
    if corner_label:
        ax.text(0.96, 0.15, corner_label, transform=ax.transAxes, fontsize=CORNER_LABEL_SIZE, va='top', ha='right')
    
    # NEW: Metrics Text (The average shade values)
    if metrics_text:
        # Placed in top-left or wherever fits best (using relative coordinates)
        ax.text(0.04, 0.96, metrics_text, transform=ax.transAxes, 
                fontsize=METRIC_TEXT_SIZE, va='top', ha='left', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    if y_lim:
        ax.set_ylim(y_lim)

    plt.tight_layout()
    
    if save_filename:
        full_path = os.path.join(OUTPUT_FOLDER, save_filename)
        plt.savefig(full_path, format='pdf', bbox_inches='tight')
        print(f"Saved: {full_path}")
    
    plt.show()

# ==========================================
# 4. EXECUTION
# ==========================================

# Load Data
all_datasets = parse_stacked_data(FILENAME, bin_width=BIN_WIDTH)
print(f"Total Datasets Loaded: {len(all_datasets)}")

c_map = {'no': 'magenta', 'sym': 'teal', 'sr': 'blue'}
l_map = {
    'no': r'NN-CC',
    'sym': r'NN-CC$_{\text{+sym}}$',
    'sr': r'NN-CC$_{\text{+sym+post-SR}}$'
}

# --- PHASE 1: INDIVIDUAL DATASETS (RAW) ---
print("\n=== PHASE 1: Individual Dataset Plots (Raw Values) ===")

for idx, label_nosym, label_sym in PLOTS_TO_GENERATE:
    if idx >= len(all_datasets): continue
    
    df = all_datasets[idx]
    x = df[df.columns[0]]
    
    # No Sym
    plot_single_frame(
        x, [df['res_f2_nosym']], 
        [l_map['no']], [c_map['no']], 
        r"NN$_2(x) - f_2^{\text{th.}}(x)$",
        maj_x=1.0, maj_y=0.02, min_x=0.2, min_y=0.01,
        corner_label=label_nosym, y_lim=(-0.055, 0.05),
        save_filename=f"dataset_{idx+1}_nosym.pdf"
    )

    # Sym & SR
    plot_single_frame(
        x, [df['res_f2_sym'], df['res_f2_NN_SR']], 
        [l_map['sym'], l_map['sr']], [c_map['sym'], c_map['sr']], 
        r"NN$_2(x) - f_2^{\text{th.}}(x)$",
        maj_x=1.0, maj_y=0.02, min_x=0.2, min_y=0.01,
        corner_label=label_sym, y_lim=(-0.025, 0.025),
        save_filename=f"dataset_{idx+1}_sym.pdf"
    )

# --- PHASE 2: GLOBAL STATISTICS ---
print("\n=== PHASE 2: Whole Database Statistics ===")

if all_datasets:
    df_mean, df_min, df_max, df_ci = calculate_global_stats(all_datasets)
    x_stats = df_mean.index
    
    LABEL_STATS_B = "(c) 99\% CI"
    LABEL_STATS_C = "(f) 99\% CI"
    LABEL_STATS_D = "(c)"
    LABEL_STATS_E = "(f)"

    # ---------------------------------------------------------
    # 1. Global 99% CI (No Sym)
    # ---------------------------------------------------------
    # Calculate Metric: Average CI radius
    avg_ci_no = get_shade_metrics(df_ci, None, 'res_f2_nosym', mode="ci")
    msg_ci_no = f"Avg CI: {avg_ci_no:.3f}" # Scientific notation for compactness
    
    print(f"Stats (NoSym CI): {msg_ci_no}")

    plot_single_frame(
        x_stats, [df_mean['res_f2_nosym']], 
        [l_map['no']], [c_map['no']], 
        r"NN$_2(x) - f_2^{\text{th.}}(x)$ (99\% CI)",
        maj_x=1.0, maj_y=0.02, min_x=0.2, min_y=0.01,
        corner_label=LABEL_STATS_B, y_lim=(-0.055, 0.05),
        metrics_text=msg_ci_no,  # <--- PASSING METRIC TEXT
        fill_bounds=[(df_mean['res_f2_nosym'] - df_ci['res_f2_nosym'], 
                      df_mean['res_f2_nosym'] + df_ci['res_f2_nosym'])],
        save_filename="global_stats_CI_nosym.pdf"
    )

    # ---------------------------------------------------------
    # 2. Global 99% CI (Sym & SR)
    # ---------------------------------------------------------
    avg_ci_sym = get_shade_metrics(df_ci, None, 'res_f2_sym', mode="ci")
    avg_ci_sr = get_shade_metrics(df_ci, None, 'res_f2_NN_SR', mode="ci")
    
    msg_ci_sym = f"Avg CI (Sym): {avg_ci_sym:.3f}\nAvg CI (SR): {avg_ci_sr:.1e}"
    print(f"Stats (Sym/SR CI):\n{msg_ci_sym}")

    plot_single_frame(
        x_stats, [df_mean['res_f2_sym'], df_mean['res_f2_NN_SR']], 
        [l_map['sym'], l_map['sr']], [c_map['sym'], c_map['sr']], 
        r"NN$_2(x) - f_2^{\text{th.}}(x)$ (99\% CI)",
        maj_x=1.0, maj_y=0.02, min_x=0.2, min_y=0.01,
        corner_label=LABEL_STATS_C, y_lim=(-0.025, 0.025),
        metrics_text=msg_ci_sym, # <--- PASSING METRIC TEXT
        fill_bounds=[
            (df_mean['res_f2_sym'] - df_ci['res_f2_sym'], df_mean['res_f2_sym'] + df_ci['res_f2_sym']),
            (df_mean['res_f2_NN_SR'] - df_ci['res_f2_NN_SR'], df_mean['res_f2_NN_SR'] + df_ci['res_f2_NN_SR'])
        ],
        save_filename="global_stats_CI_sym.pdf"
    )

    # ---------------------------------------------------------
    # 3. Global Min/Max (No Sym)
    # ---------------------------------------------------------
    # Calculate Metric: Average Range Width (Max - Min)
    avg_w_no = get_shade_metrics(df_max, df_min, 'res_f2_nosym', mode="range")
    msg_rng_no = f"Avg Range: {avg_w_no:.3f}"

    print(f"Stats (NoSym Range): {msg_rng_no}")

    plot_single_frame(
        x_stats, [df_mean['res_f2_nosym']], 
        [l_map['no']], [c_map['no']], 
        r"NN$_2(x) - f_2^{\text{th.}}(x)$",
        maj_x=1.0, maj_y=0.02, min_x=0.2, min_y=0.01,
        corner_label=LABEL_STATS_D, y_lim=(-0.055, 0.05), fill_alpha=0.15,
        #metrics_text=msg_rng_no, # <--- PASSING METRIC TEXT
        fill_bounds=[(df_min['res_f2_nosym'], df_max['res_f2_nosym'])],
        save_filename="Fig4c_global_stats_range_nosym.pdf"
    )

    # ---------------------------------------------------------
    # 4. Global Min/Max (Sym & SR)
    # ---------------------------------------------------------
    avg_w_sym = get_shade_metrics(df_max, df_min, 'res_f2_sym', mode="range")
    avg_w_sr = get_shade_metrics(df_max, df_min, 'res_f2_NN_SR', mode="range")
    
    msg_rng_sym = f"Avg Range (Sym): {avg_w_sym:.3f}\nAvg Range (SR): {avg_w_sr:.1e}"
    print(f"Stats (Sym/SR Range):\n{msg_rng_sym}")

    plot_single_frame(
        x_stats, [df_mean['res_f2_sym'], df_mean['res_f2_NN_SR']], 
        [l_map['sym'], l_map['sr']], [c_map['sym'], c_map['sr']], 
        r"NN$_2(x) - f_2^{\text{th.}}(x)$",
        maj_x=1.0, maj_y=0.02, min_x=0.2, min_y=0.01,
        corner_label=LABEL_STATS_E, y_lim=(-0.025, 0.025), fill_alpha=0.15,
        #metrics_text=msg_rng_sym, # <--- PASSING METRIC TEXT
        fill_bounds=[
            (df_min['res_f2_sym'], df_max['res_f2_sym']),
            (df_min['res_f2_NN_SR'], df_max['res_f2_NN_SR'])
        ],
        save_filename="Fig4f_global_stats_range_sym.pdf"
    )
