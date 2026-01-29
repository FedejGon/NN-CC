import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter, FixedLocator, FuncFormatter
from matplotlib.ticker import LogFormatterSciNotation

# ==========================================
# CONFIGURATION
# ==========================================
FIX_B_TO_ONE = True  # Set to True to force B=1, False for free fit

# Replace with your filename
filename = 'rmse_results_for_f1_and_f2.txt'
#filename = 'rmse_results_for_f1_and_f2_symmetries_only_eval_data.txt'
fontsize = 16

col_names = [
    "noise_perc_th", "noise_perc", "SNR_dB", "NN_f1", "NN_f2",
    "NN_sym_f1", "NN_sym_f2","NN_symSR_f1", "NN_symSR_f2",  "SINDy_f1", "SINDy_f2",
    "LS_f1", "LS_f2", "Param_f1", "Param_f2", "SR_f1", "SR_f2"
]

# Create dummy data for demonstration if file doesn't exist
try:
    df = pd.read_csv(filename, skiprows=6, delim_whitespace=True, header=None, names=col_names)
except FileNotFoundError:
    print("Warning: File not found. Creating dummy data for testing.")
    data = {col: np.random.rand(10) * 0.1 for col in col_names}
    data['noise_perc_th'] = np.logspace(0, 3, 10)
    data['SNR_dB'] = np.linspace(40, -20, 10)
    df = pd.DataFrame(data)

grouped = df.groupby('noise_perc_th').mean().reset_index()
# Applies specific modifications as per your original script
#grouped['SR_f1'] = grouped['SR_f1'] * 1.0
#grouped['SR_f2'] = grouped['SR_f2'] * 1.0
#grouped['SINDy_f2'] = grouped['LS_f2'] * 1.0 #(0.9 + np.random.rand(len(grouped['LS_f2'])))

x_bottom = grouped['noise_perc_th']
x_top = grouped['SNR_dB']

# ---------------------------------------------------------
# Define Eta for Fitting
# eta = noise_percentage / 100
# ---------------------------------------------------------
eta = x_bottom / 100.0
log_eta = np.log(eta)

# Your desired SNR_dB labels to show on top
desired_top_labels = [40, 30, 20, 10, 0, -10, -20]
tick_positions = []
tick_labels = []
for label in desired_top_labels:
    idx = (np.abs(x_top - label)).argmin()
    pos = x_bottom.iloc[idx]
    tick_positions.append(pos)
    tick_labels.append(str(label))

# =========================================================
# PLOT 1: F1
# =========================================================
plot1_cols = ['Param_f1', 'NN_symSR_f1', 'SR_f1', 'LS_f1', 'SINDy_f1']

label_map_f1 = {
    'NN_symSR_f1': 'NN-CC$_{+sym+post\!\!-\!\!SR}$', 
    'SINDy_f1': 'SINDy-CC',
    'LS_f1': 'Poly-CC',
    'Param_f1': 'Parametric',
    'SR_f1': 'SR'
}
plot_colors = ['black', 'blue', 'darkred', 'darkgreen', 'darkorange']
plot_markers = ['o', 's', '^', 'D', 'p', '*']
plot_ms = [5, 5, 6, 5, 6]

fig, ax_bottom = plt.subplots(figsize=(6, 6))

print("--- Fits for F1 (RMSE = A * eta^B) ---")
if FIX_B_TO_ONE:
    print("(NOTE: B is fixed to 1.0)")
print(f"{'Method':<20} | {'A (Prefactor)':<15} | {'B (Exponent)':<15}")
print("-" * 55)

for col, color, marker, ms in zip(plot1_cols, plot_colors, plot_markers, plot_ms):
    y_data = grouped[col]
    
    # 1. Plot original data (all points)
    ax_bottom.plot(x_bottom, y_data, marker=marker, markersize=ms, 
                   color=color, label=label_map_f1[col], linewidth=0, alpha=0.8)

    # 2. Prepare Data for Fitting
    log_y = np.log(y_data)
    
    # --- SR MODEL CONSTRAINT: Fit only on x <= 60 ---
    if col == 'SR_f1':
        mask = x_bottom <= 60.0
        fit_log_eta = log_eta[mask]
        fit_log_y = log_y[mask]
    else:
        fit_log_eta = log_eta
        fit_log_y = log_y

    # 3. Perform Fit
    if FIX_B_TO_ONE:
        B = 1.0
        log_A = np.mean(fit_log_y - fit_log_eta)
        A = np.exp(log_A)
    else:
        if len(fit_log_eta) > 0:
            coeffs = np.polyfit(fit_log_eta, fit_log_y, 1) 
            B = coeffs[0]
            log_A = coeffs[1]
            A = np.exp(log_A)
        else:
            A, B = 0, 0
    
    print(f"{label_map_f1[col]:<20} | {A:<15.4e} | {B:<15.4f}")
    
    # 4. Plot Fitted Curve
    if col == 'SR_f1':
        # -- Special Handling for SR --
        # 1. Generate fit line ONLY up to 60
        x_fit_sr = np.linspace(x_bottom.min(), 60, 50) 
        eta_fit_sr = x_fit_sr / 100.0
        y_fit_sr = A * (eta_fit_sr ** B)
        
        # Plot the SR fit line (standard color)
        ax_bottom.plot(x_fit_sr, y_fit_sr, linestyle='--', color=color, linewidth=3, alpha=0.9)
        
        # 2. Add Divergence Line (Red vertical line)
        # Starts from the last point of the fit (at 60%) and goes up
        y_divergence_start = y_fit_sr[-1]
        ax_bottom.plot([60, 60], [y_divergence_start, 2.0], color='darkred', linestyle='--', linewidth=3.0)
        
    else:
        # -- Standard Handling --
        # Plot fit line across the whole range
        y_fit = A * (eta ** B)
        ax_bottom.plot(x_bottom, y_fit, linestyle='--', color=color, linewidth=3, alpha=0.9)


ax_bottom.set_xlabel(r'Percentage of noise, $100\,\eta$ (%)', fontsize=fontsize)
ax_bottom.set_ylabel(r'RMSE [ f$_1$($\dot{x}$) ]', fontsize=fontsize)
# Horizontal reference line
ax_bottom.axhline(y=20.0, color='darkred', linestyle='--')

# --- LEGEND REORDERING ---
handles, labels = ax_bottom.get_legend_handles_labels()
desired_order = [
    'SINDy-CC',
    'Poly-CC',
    'NN-CC$_{+sym+post\!\!-\!\!SR}$',
    'SR',
    'Parametric'
]
label_handle_map = dict(zip(labels, handles))
ordered_handles = [label_handle_map[l] for l in desired_order if l in label_handle_map]
ordered_labels = [l for l in desired_order if l in label_handle_map]

ax_bottom.legend(ordered_handles, ordered_labels, labelspacing=0.1, fontsize=fontsize)
# -------------------------

ax_bottom.text(0.02, 0.98, '(a)', transform=ax_bottom.transAxes, fontsize=20, verticalalignment='top')

# Log scales
ax_bottom.set_xscale('log')
ax_bottom.set_yscale('log')

# Tick appearance
ax_bottom.tick_params(axis='x', which='major', direction='in', length=12, top=True, labelsize=fontsize)
ax_bottom.tick_params(axis='x', which='minor', direction='in', length=4, top=True)
ax_bottom.tick_params(axis='y', which='major', direction='in', length=12, right=True, labelsize=fontsize)
ax_bottom.tick_params(axis='y', which='minor', direction='in', length=4, right=True)

# Top x-axis
ax_top = ax_bottom.twiny()
ax_top.yaxis.set_major_locator(LogLocator(base=10.0))
ax_top.yaxis.set_major_formatter(ScalarFormatter())
ax_top.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.1, 10) / 1.0, numticks=10))
ax_top.yaxis.set_minor_formatter(NullFormatter())

# Set y-axis limits
ax_bottom.set_ylim(1e-3, 1e1)

# Format y-axis
ax_bottom.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
minor_tick_positions = []
for decade in [1, 10]:
    for sub in [2, 3, 4, 5, 6, 7, 8, 9]:
        pos = decade * sub
        if 1 <= pos <= 100:
            minor_tick_positions.append(pos)

ax_bottom.yaxis.set_minor_locator(FixedLocator(minor_tick_positions))

def minor_tick_formatter(x, pos):
    if abs(x - 2) < 0.1 or abs(x - 5) < 0.1 or abs(x - 20) < 0.1 or abs(x - 50) < 0.1:
        return str(int(round(x)))
    else:
        return ''

ax_bottom.yaxis.set_minor_formatter(FuncFormatter(minor_tick_formatter))
ax_bottom.tick_params(axis='y', which='minor', length=5, labelsize=fontsize)

# Set bottom x-axis major ticks
bottom_ticks = [1, 10, 100, 1000]
ax_bottom.set_xticks(bottom_ticks)
ax_bottom.set_xticklabels([str(tick) for tick in bottom_ticks], fontsize=fontsize)
ax_bottom.set_xlim(1, 1000)

ax_top.set_xlim(ax_bottom.get_xlim())
ax_top.set_xscale('log')
ax_top.set_xticks(tick_positions)
ax_top.set_xticklabels(tick_labels, fontsize=fontsize)
ax_top.tick_params(axis='x', which='major', direction='out', length=12, top=True, bottom=False)
ax_top.tick_params(axis='x', which='minor', direction='out', length=4, top=True, bottom=False)
ax_top.set_xlabel('SNR (dB)', fontsize=fontsize)
ax_top.minorticks_off()
ax_bottom.minorticks_on()
ax_bottom.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[2, 3, 4, 5, 6, 7, 8, 9]))
ax_bottom.yaxis.set_minor_formatter(NullFormatter())
ax_bottom.tick_params(axis='y', which='minor', direction='in', length=4, right=True)

plt.tight_layout()
plt.savefig("rmse_f1.pdf")


# =========================================================
# PLOT 2: F2
# =========================================================
plot2_cols = ['Param_f2', 'NN_symSR_f2', 'SR_f2', 'LS_f2', 'SINDy_f2']

label_map_f2 = {
    'NN_symSR_f2': 'NN-CC$_{+sym+post\!\!-\!\!SR}$',
    'SINDy_f2': 'SINDy-CC',
    'LS_f2': 'Poly-CC',
    'Param_f2': 'Parametric',
    'SR_f2': 'SR'
}

fig, ax_bottom = plt.subplots(figsize=(6, 6))

print("\n--- Fits for F2 (RMSE = A * eta^B) ---")
if FIX_B_TO_ONE:
    print("(NOTE: B is fixed to 1.0)")
print(f"{'Method':<20} | {'A (Prefactor)':<15} | {'B (Exponent)':<15}")
print("-" * 55)

for col, color, marker, ms in zip(plot2_cols, plot_colors, plot_markers, plot_ms):
    y_data = grouped[col]
    
    # 1. Plot original data
    ax_bottom.plot(x_bottom, y_data, marker=marker, markersize=ms, 
                   color=color, label=label_map_f2[col], linewidth=0, alpha=0.8)

    # 2. Prepare Data for Fitting
    log_y = np.log(y_data)
    
    # --- SR MODEL CONSTRAINT ---
    if col == 'SR_f2':
        mask = x_bottom <= 60.0
        fit_log_eta = log_eta[mask]
        fit_log_y = log_y[mask]
    else:
        fit_log_eta = log_eta
        fit_log_y = log_y

    # 3. Perform Fit
    if FIX_B_TO_ONE:
        B = 1.0
        log_A = np.mean(fit_log_y - fit_log_eta)
        A = np.exp(log_A)
    else:
        if len(fit_log_eta) > 0:
            coeffs = np.polyfit(fit_log_eta, fit_log_y, 1) 
            B = coeffs[0]
            log_A = coeffs[1]
            A = np.exp(log_A)
        else:
            A, B = 0, 0
    
    print(f"{label_map_f2[col]:<20} | {A:<15.4e} | {B:<15.4f}")
    
    # 4. Plot Fitted Curve
    if col == 'SR_f2':
        # -- Special Handling for SR --
        x_fit_sr = np.linspace(x_bottom.min(), 60, 50) 
        eta_fit_sr = x_fit_sr / 100.0
        y_fit_sr = A * (eta_fit_sr ** B)
        
        ax_bottom.plot(x_fit_sr, y_fit_sr, linestyle='--', color=color, linewidth=3, alpha=0.9)
        
        # Red Divergence Line
        y_divergence_start = y_fit_sr[-1]
        ax_bottom.plot([60, 60], [y_divergence_start, 2.0], color='darkred', linestyle='--', linewidth=3.0)
        
    else:
        # -- Standard Handling --
        y_fit = A * (eta ** B)
        ax_bottom.plot(x_bottom, y_fit, linestyle='--', color=color, linewidth=3, alpha=0.9)


ax_bottom.set_xlabel(r'Percentage of noise, $100\,\eta$ (%)', fontsize=fontsize)
ax_bottom.set_ylabel(r'RMSE [ f$_2$($x$) ]', fontsize=fontsize)
ax_bottom.axhline(y=20.0, color='darkred', linestyle='--')

# --- LEGEND REORDERING ---
handles, labels = ax_bottom.get_legend_handles_labels()
desired_order = [
    'SINDy-CC',
    'Poly-CC',
    'NN-CC$_{+sym+post\!\!-\!\!SR}$',
    'SR',
    'Parametric'
]

label_handle_map = dict(zip(labels, handles))
ordered_handles = [label_handle_map[l] for l in desired_order if l in label_handle_map]
ordered_labels = [l for l in desired_order if l in label_handle_map]

ax_bottom.legend(ordered_handles, ordered_labels, labelspacing=0.1, fontsize=fontsize)
# -------------------------

ax_bottom.text(0.02, 0.98, '(b)', transform=ax_bottom.transAxes, fontsize=20, verticalalignment='top')

# Log scales
ax_bottom.set_xscale('log')
ax_bottom.set_yscale('log')

# Tick appearance
ax_bottom.tick_params(axis='x', which='major', direction='in', length=12, top=True, labelsize=fontsize)
ax_bottom.tick_params(axis='x', which='minor', direction='in', length=4, top=True)
ax_bottom.tick_params(axis='y', which='major', direction='in', length=12, right=True, labelsize=fontsize)
ax_bottom.tick_params(axis='y', which='minor', direction='in', length=4, right=True)

# Top x-axis
ax_top = ax_bottom.twiny()
ax_top.yaxis.set_major_locator(LogLocator(base=10.0))
ax_top.yaxis.set_major_formatter(ScalarFormatter())
ax_top.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.1, 10) / 1.0, numticks=10))
ax_top.yaxis.set_minor_formatter(NullFormatter())

# Set y-axis limits
ax_bottom.set_ylim(1e-2, 1e1)
ax_bottom.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
ax_bottom.yaxis.set_minor_locator(FixedLocator(minor_tick_positions))
ax_bottom.yaxis.set_minor_formatter(FuncFormatter(minor_tick_formatter))
ax_bottom.tick_params(axis='y', which='minor', length=5, labelsize=fontsize)

# Set bottom x-axis major ticks
bottom_ticks = [1, 10, 100, 1000]
ax_bottom.set_xticks(bottom_ticks)
ax_bottom.set_xticklabels([str(tick) for tick in bottom_ticks], fontsize=fontsize)
ax_bottom.set_xlim(1, 1000)

ax_top.set_xlim(ax_bottom.get_xlim())
ax_top.set_xscale('log')
ax_top.set_xticks(tick_positions)
ax_top.set_xticklabels(tick_labels, fontsize=fontsize)
ax_top.tick_params(axis='x', which='major', direction='out', length=12, top=True, bottom=False)
ax_top.tick_params(axis='x', which='minor', direction='out', length=4, top=True, bottom=False)
ax_top.set_xlabel('SNR (dB)', fontsize=fontsize)
ax_top.minorticks_off()
ax_bottom.minorticks_on()
ax_bottom.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[2, 3, 4, 5, 6, 7, 8, 9]))
ax_bottom.yaxis.set_minor_formatter(NullFormatter())
ax_bottom.tick_params(axis='y', which='minor', direction='in', length=4, right=True)

plt.tight_layout()
plt.savefig("rmse_f2.pdf")
plt.show()
