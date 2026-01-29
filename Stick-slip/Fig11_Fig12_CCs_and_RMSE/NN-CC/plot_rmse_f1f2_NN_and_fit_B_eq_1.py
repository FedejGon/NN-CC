import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter, FixedLocator, FuncFormatter
from matplotlib.ticker import LogFormatterSciNotation

# ==========================================
# CONFIGURATION
# ==========================================
# Set this to True to force the exponent B = 1.0 during fitting.
# Set to False to allow the code to calculate B freely (standard polyfit).
FIX_B_TO_ONE = True 

filename = 'rmse_results_for_f1_and_f2.txt'
#filename2 = 'rmse_results_for_f1_and_f2_without_symmetry.txt'
fontsize = 16

col_names = [
    "noise_perc_th", "noise_perc", "SNR_dB",     
    "NN_f1", "NN_f2","NN_f1_SR", "NN_f2_SR",  "NN_f1_sym", "NN_f2_sym",  "NN_f1_symSR", "NN_f2_symSR" , "Param_f1","Param_f2"
]


# --- LOAD DATA OR CREATE DUMMY DATA FOR TESTING ---
try:
    df = pd.read_csv(filename, skiprows=6, delim_whitespace=True, header=None, names=col_names)
  #  df2 = pd.read_csv(filename2, skiprows=6, delim_whitespace=True, header=None, names=col_names2)
except FileNotFoundError:
    print("Warning: Files not found. Creating dummy data for demonstration.")
    # Create dummy data 1
    data1 = {col: np.random.rand(10) * 0.1 for col in col_names}
    data1['noise_perc_th'] = np.logspace(0, 3, 10)
    data1['SNR_dB'] = np.linspace(40, -20, 10)
    df = pd.DataFrame(data1)
    
grouped = df.groupby('noise_perc_th').mean().reset_index()

x_bottom = grouped['noise_perc_th']
x_top = grouped['SNR_dB']

# Prepare Etas for fitting
# eta = noise_percentage / 100
eta1 = x_bottom / 100.0
log_eta1 = np.log(eta1)

# Desired top labels
desired_top_labels = [40, 30, 20, 10, 0, -10, -20]
tick_positions = []
tick_labels = []
for label in desired_top_labels:
    idx = (np.abs(x_top - label)).argmin()
    pos = x_bottom.iloc[idx]
    tick_positions.append(pos)
    tick_labels.append(str(label))


# =============================================================================
# FIGURE 1: f1
# =============================================================================
plot1_cols = ['Param_f1','NN_f1_symSR','NN_f1_sym','NN_f1','NN_f1_SR']#'SR_f1','LS_f1','SINDy_f1']
# Create a dictionary to map column names to desired labels
label_map = {
    'Param_f1': 'Parametric',
    'NN_f1_symSR': r'NN-CC$_{+sym+post\!\!-\!\!SR}$',
    'NN_f1_sym': r'NN-CC$_{+sym}$',
    'NN_f1': 'NN-CC',
    'NN_f1_SR': r'NN-CC$_{+post\!\!-\!\!SR}$',
#    'SINDy_f1': 'SINDy-CC',
#    'LS_f1': 'LS-CC',
#    'SR_f1': 'SR'
}
plot_colors = ['black','blue','teal','magenta', 'gray']#, 'darkgreen', 'darkorange' ]
plot_markers = ['o', 's', '^','*','D']#, 'D', 'p', '*']
plot_ms = [5, 5, 6,7,5]#, 5, 6]  # List of marker sizes


fig, ax_bottom = plt.subplots(figsize=(6, 6))

print("--- F1 Fits (RMSE = A * eta^B) ---")
if FIX_B_TO_ONE:
    print("(NOTE: B is fixed to 1.0)")
print(f"{'Method':<25} | {'A (Prefactor)':<15} | {'B (Exponent)':<15}")
print("-" * 60)

# --- Loop 1: Data with Symmetry (Grouped 1) ---
for col, color, marker, ms in zip(plot1_cols, plot_colors, plot_markers, plot_ms):
    y_data = grouped[col]
    
    # Plot points
    ax_bottom.plot(x_bottom, y_data, marker=marker, markersize=ms, 
                   color=color, label=label_map[col], linewidth=0.0, alpha=0.9)
    
    # Fit Logic
    log_y = np.log(y_data)
    
    if FIX_B_TO_ONE:
        # Force slope = 1. 
        # In log-log space: log(y) = 1*log(x) + log(A) 
        # Therefore: log(A) = mean(log(y) - log(x))
        B = 1.0
        log_A = np.mean(log_y - log_eta1)
        A = np.exp(log_A)
    else:
        # Standard free fit
        coeffs = np.polyfit(log_eta1, log_y, 1)
        B = coeffs[0]
        A = np.exp(coeffs[1])
    
    print(f"{label_map[col]:<25} | {A:<15.4e} | {B:<15.4f}")
    
    # Plot Fit
    y_fit = A * (eta1 ** B)
    ax_bottom.plot(x_bottom, y_fit, linestyle='--', color=color, linewidth=3.5, alpha=0.8)


ax_bottom.set_xlabel(r'Percentage of noise, $100\, \eta$ (%)', fontsize=fontsize)
ax_bottom.set_ylabel(r'RMSE [ f$_1$($\dot{x}$) ]  (N)', fontsize=fontsize)
ax_bottom.axhline(y=20.0, color='red', linestyle='--')

# --- LEGEND REORDERING LOGIC ---
handles, labels = ax_bottom.get_legend_handles_labels()
desired_order = [
    'NN-CC', 
    r'NN-CC$_{+post\!\!-\!\!SR}$', 
    r'NN-CC$_{+sym}$', 
    r'NN-CC$_{+sym+post\!\!-\!\!SR}$', 
    'Parametric'
]
label_handle_map = dict(zip(labels, handles))
ordered_handles = [label_handle_map[l] for l in desired_order if l in label_handle_map]
ordered_labels = [l for l in desired_order if l in label_handle_map]
ax_bottom.legend(ordered_handles, ordered_labels,loc='lower right', labelspacing=0.1, fontsize=fontsize,frameon=False,markerfirst=False,handletextpad=0.1)
# -------------------------------

ax_bottom.text(0.02, 0.98, '(a)', transform=ax_bottom.transAxes, fontsize=20, verticalalignment='top')

# Log scales and Ticks
ax_bottom.set_xscale('log')
ax_bottom.set_yscale('log')
ax_bottom.tick_params(axis='x', which='major', direction='in', length=12, top=True, labelsize=fontsize)
ax_bottom.tick_params(axis='x', which='minor', direction='in', length=4, top=True)
ax_bottom.tick_params(axis='y', which='major', direction='in', length=12, right=True, labelsize=fontsize)
ax_bottom.tick_params(axis='y', which='minor', direction='in', length=4, right=True)

# Top Axis
ax_top = ax_bottom.twiny()
ax_top.yaxis.set_major_locator(LogLocator(base=10.0))
ax_top.yaxis.set_major_formatter(ScalarFormatter())
ax_top.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.1, 10)/1.0, numticks=10))
ax_top.yaxis.set_minor_formatter(NullFormatter())

# Axis Limits and Formatters
ax_bottom.set_ylim(1e-4, 1e1)
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
plt.savefig("Fig11a_rmse_f1.pdf")


# =============================================================================
# FIGURE 2: f2
# =============================================================================
print("\n--- F2 Fits (RMSE = A * eta^B) ---")
if FIX_B_TO_ONE:
    print("(NOTE: B is fixed to 1.0)")
print(f"{'Method':<25} | {'A (Prefactor)':<15} | {'B (Exponent)':<15}")
print("-" * 60)

plot2_cols = ['Param_f2','NN_f2_symSR','NN_f2_sym','NN_f2','NN_f2_SR']#'SR_f1','LS_f1','SINDy_f1']
# Create a dictionary to map column names to desired labels
label_map_f2 = {
    'Param_f2': 'Parametric',
    'NN_f2_symSR': r'NN-CC$_{+sym+post\!\!-\!\!SR}$',
    'NN_f2_sym': r'NN-CC$_{+sym}$',
    'NN_f2': 'NN-CC',
    'NN_f2_SR': r'NN-CC$_{+post\!\!-\!\!SR}$',
#    'SINDy_f1': 'SINDy-CC',
#    'LS_f1': 'LS-CC',
#    'SR_f1': 'SR'
}
plot_colors = ['black','blue','teal','magenta', 'gray']#, 'darkgreen', 'darkorange' ]
plot_markers = ['o', 's', '^','*','D']#, 'D', 'p', '*']
plot_ms = [5, 5, 6,7,5]#, 5, 6]  # List of marker sizes


fig, ax_bottom = plt.subplots(figsize=(6, 6))

# --- Loop 1: Data with Symmetry (Grouped 1) ---
for col, color, marker, ms in zip(plot2_cols, plot_colors, plot_markers, plot_ms):
    y_data = grouped[col]
    
    # Plot points
    ax_bottom.plot(x_bottom, y_data, marker=marker, markersize=ms, 
                   color=color, label=label_map_f2[col], linewidth=0.0, alpha=0.9)
    
    # Fit Logic
    log_y = np.log(y_data)
    
    if FIX_B_TO_ONE:
        B = 1.0
        log_A = np.mean(log_y - log_eta1)
        A = np.exp(log_A)
    else:
        coeffs = np.polyfit(log_eta1, log_y, 1)
        B = coeffs[0]
        A = np.exp(coeffs[1])
    
    print(f"{label_map_f2[col]:<25} | {A:<15.4e} | {B:<15.4f}")
    
    # Plot Fit
    y_fit = A * (eta1 ** B)
    ax_bottom.plot(x_bottom, y_fit, linestyle='--', color=color, linewidth=3.5, alpha=0.8)



ax_bottom.set_xlabel(r'Percentage of noise, $100\, \eta$ (%)', fontsize=fontsize)
ax_bottom.set_ylabel(r'RMSE [ f$_2$($x$) ]  (N)', fontsize=fontsize)
ax_bottom.axhline(y=20.0, color='red', linestyle='--')

# --- LEGEND REORDERING LOGIC ---
handles, labels = ax_bottom.get_legend_handles_labels()
desired_order = [
    'NN-CC', 
    r'NN-CC$_{+post\!\!-\!\!SR}$', 
    r'NN-CC$_{+sym}$', 
    r'NN-CC$_{+sym+post\!\!-\!\!SR}$', 
    'Parametric'
]
label_handle_map = dict(zip(labels, handles))
ordered_handles = [label_handle_map[l] for l in desired_order if l in label_handle_map]
ordered_labels = [l for l in desired_order if l in label_handle_map]
ax_bottom.legend(ordered_handles, ordered_labels,loc='lower right', labelspacing=0.1, fontsize=fontsize,frameon=False,markerfirst=False,handletextpad=0.1)
# -------------------------------

ax_bottom.text(0.02, 0.98, '(b)', transform=ax_bottom.transAxes, fontsize=20, verticalalignment='top')

# Formatting (Copy-pasted logic)
ax_bottom.set_xscale('log')
ax_bottom.set_yscale('log')
ax_bottom.tick_params(axis='x', which='major', direction='in', length=12, top=True, labelsize=fontsize)
ax_bottom.tick_params(axis='x', which='minor', direction='in', length=4, top=True)
ax_bottom.tick_params(axis='y', which='major', direction='in', length=12, right=True, labelsize=fontsize)
ax_bottom.tick_params(axis='y', which='minor', direction='in', length=4, right=True)

ax_top = ax_bottom.twiny()
ax_top.yaxis.set_major_locator(LogLocator(base=10.0))
ax_top.yaxis.set_major_formatter(ScalarFormatter())
ax_top.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.1, 10)/1.0, numticks=10))
ax_top.yaxis.set_minor_formatter(NullFormatter())

ax_bottom.set_ylim(1e-4, 1e1)
ax_bottom.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
ax_bottom.yaxis.set_minor_locator(FixedLocator(minor_tick_positions))
ax_bottom.yaxis.set_minor_formatter(FuncFormatter(minor_tick_formatter))
ax_bottom.tick_params(axis='y', which='minor', length=5, labelsize=fontsize)

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
plt.savefig("Fig11b_rmse_f2.pdf")
plt.show()
