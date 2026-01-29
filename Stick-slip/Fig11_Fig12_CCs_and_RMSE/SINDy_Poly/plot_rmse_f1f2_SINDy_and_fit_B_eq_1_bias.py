import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter, FixedLocator, FuncFormatter
from matplotlib.ticker import LogFormatterSciNotation
from scipy.optimize import curve_fit

# ==========================================
# CONFIGURATION
# ==========================================
FIX_B_TO_ONE = True  # Only affects the "Parametric" linear fit
filename = 'rmse_results_for_f1_and_f2.txt'
fontsize = 16

col_names = [
    "noise_perc_th", "noise_perc", "SNR_dB", "SINDy_f1","SINDy_f2","Poly_f1","Poly_f2","Param_f1","Param_f2"
]

# Create dummy data if file doesn't exist
try:
    df = pd.read_csv(filename, skiprows=6, delim_whitespace=True, header=None, names=col_names)
except FileNotFoundError:
    print("Warning: File not found. Creating dummy data for testing.")
    data = {col: np.random.rand(10) * 0.1 for col in col_names}
    data['noise_perc_th'] = np.logspace(0, 3, 10)
    data['SNR_dB'] = np.linspace(40, -20, 10)
    df = pd.DataFrame(data)

grouped = df.groupby('noise_perc_th').mean().reset_index()

x_bottom = grouped['noise_perc_th']
x_top = grouped['SNR_dB']

# ---------------------------------------------------------
# Define Eta for Fitting
# ---------------------------------------------------------
eta = x_bottom / 100.0
log_eta = np.log(eta)

# Desired SNR_dB labels
desired_top_labels = [40, 30, 20, 10, 0, -10, -20]
tick_positions = []
tick_labels = []
for label in desired_top_labels:
    idx = (np.abs(x_top - label)).argmin()
    pos = x_bottom.iloc[idx]
    tick_positions.append(pos)
    tick_labels.append(str(label))

# =========================================================
# HELPER FUNCTIONS FOR FITTING
# =========================================================

# 1. The Bias-Variance Model: RMSE = sqrt(Bias^2 + (Slope * eta)^2)
def bias_variance_model(eta_val, bias, slope):
    return np.sqrt(bias**2 + (slope * eta_val)**2)

# =========================================================
# PLOT 1: F1
# =========================================================
plot1_cols = ['Param_f1', 'SINDy_f1', 'Poly_f1']

label_map_f1 = {
    'Param_f1': 'Parametric',
    'SINDy_f1': 'SINDy-CC',
    'Poly_f1': 'Poly-CC',
}
plot_colors = ['black', 'darkorange','darkgreen']
plot_markers = ['o', 'D','p']
plot_ms = [5, 5, 7]

fig, ax_bottom = plt.subplots(figsize=(6, 6))

print("--- Fits for F1 ---")

for col, color, marker, ms in zip(plot1_cols, plot_colors, plot_markers, plot_ms):
    y_data = grouped[col]
    
    # 1. Plot original data
    ax_bottom.plot(x_bottom, y_data, marker=marker, markersize=ms, 
                   color=color, label=label_map_f1[col], linewidth=0, alpha=0.8)

    # 2. FIT SELECTION LOGIC
    if 'Param' in col:
        # --- LINEAR FIT (Log-Log) for Parametric ---
        # It assumes no bias, just noise scaling: y = A * x^B
        log_y = np.log(y_data)
        
        if FIX_B_TO_ONE:
            B = 1.0
            log_A = np.mean(log_y - log_eta)
            A = np.exp(log_A)
        else:
            coeffs = np.polyfit(log_eta, log_y, 1) 
            B = coeffs[0]
            log_A = coeffs[1]
            A = np.exp(log_A)
            
        print(f"{label_map_f1[col]:<20} | Linear Fit | A={A:.4e}, B={B:.4f}")
        
        # Plot Parametric Fit
        y_fit = A * (eta ** B)
        ax_bottom.plot(x_bottom, y_fit, linestyle='--', color=color, linewidth=3, alpha=0.9)

    else:
        # --- BIAS-VARIANCE FIT for SINDy/Poly ---
        # Model: sqrt(Bias^2 + (Slope * eta)^2)
        # We provide initial guesses: Bias ~ min(y), Slope ~ 1.0
        p0 = [np.min(y_data), 1.0]
        
        try:
            popt, _ = curve_fit(bias_variance_model, eta, y_data, p0=p0, bounds=(0, np.inf))
            bias_est, slope_est = popt
            
            print(f"{label_map_f1[col]:<20} | Bias-Var Fit | Bias={bias_est:.4e}, Slope={slope_est:.4f}")
            
            # Generate smooth curve for plotting
            x_smooth = np.logspace(np.log10(x_bottom.min()), np.log10(x_bottom.max()), 100)
            eta_smooth = x_smooth / 100.0
            y_fit = bias_variance_model(eta_smooth, bias_est, slope_est)
            
            ax_bottom.plot(x_smooth, y_fit, linestyle='--', color=color, linewidth=3, alpha=0.9)
            
        except Exception as e:
            print(f"Fit failed for {col}: {e}")


ax_bottom.set_xlabel(r'Percentage of noise, $100\,\eta$ (%)', fontsize=fontsize)
ax_bottom.set_ylabel(r'RMSE [ f$_1$($\dot{x}$) ]   (N)', fontsize=fontsize)
# ax_bottom.axhline(y=20.0, color='darkred', linestyle='--') # Optional reference line

# --- LEGEND REORDERING ---
handles, labels = ax_bottom.get_legend_handles_labels()
desired_order = ['SINDy-CC', 'Poly-CC', 'Parametric']
label_handle_map = dict(zip(labels, handles))
ordered_handles = [label_handle_map[l] for l in desired_order if l in label_handle_map]
ordered_labels = [l for l in desired_order if l in label_handle_map]

#ax_bottom.legend(ordered_handles, ordered_labels, loc='lower right', labelspacing=0.1, fontsize=fontsize)
ax_bottom.legend(ordered_handles, ordered_labels,loc='lower right', labelspacing=0.1, fontsize=fontsize,frameon=False,markerfirst=False,handletextpad=0.1)
ax_bottom.text(0.02, 0.98, '(c)', transform=ax_bottom.transAxes, fontsize=20, verticalalignment='top')

# Log scales
ax_bottom.set_xscale('log')
ax_bottom.set_yscale('log')

# Tick appearance
ax_bottom.tick_params(axis='x', which='major', direction='in', length=12, top=True, labelsize=fontsize)
ax_bottom.tick_params(axis='x', which='minor', direction='in', length=4, top=True)
ax_bottom.tick_params(axis='y', which='major', direction='in', length=12, right=True, labelsize=fontsize)
ax_bottom.tick_params(axis='y', which='minor', direction='in', length=4, right=True)

# Top x-axis setup
ax_top = ax_bottom.twiny()
ax_top.yaxis.set_major_locator(LogLocator(base=10.0))
ax_top.yaxis.set_major_formatter(ScalarFormatter())
ax_top.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.1, 10) / 1.0, numticks=10))
ax_top.yaxis.set_minor_formatter(NullFormatter())

# Set y-axis limits
ax_bottom.set_ylim(1e-4, 1e1)

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
plt.savefig("Fig11c_rmse_f1.pdf")


# =========================================================
# PLOT 2: F2
# =========================================================
plot2_cols = ['Param_f2',  'SINDy_f2', 'Poly_f2']

label_map_f2 = {
    'Param_f2': 'Parametric',
    'Poly_f2': 'Poly-CC',
    'SINDy_f2': 'SINDy-CC',
}

fig, ax_bottom = plt.subplots(figsize=(6, 6))

print("\n--- Fits for F2 ---")

for col, color, marker, ms in zip(plot2_cols, plot_colors, plot_markers, plot_ms):
    y_data = grouped[col]
    
    # 1. Plot original data
    ax_bottom.plot(x_bottom, y_data, marker=marker, markersize=ms, 
                   color=color, label=label_map_f2[col], linewidth=0, alpha=0.8)

    # 2. FIT SELECTION LOGIC
    if 'Param' in col:
        # --- LINEAR FIT (Log-Log) for Parametric ---
        log_y = np.log(y_data)
        
        if FIX_B_TO_ONE:
            B = 1.0
            log_A = np.mean(log_y - log_eta)
            A = np.exp(log_A)
        else:
            coeffs = np.polyfit(log_eta, log_y, 1) 
            B = coeffs[0]
            log_A = coeffs[1]
            A = np.exp(log_A)
            
        print(f"{label_map_f2[col]:<20} | Linear Fit | A={A:.4e}, B={B:.4f}")
        
        # Plot Parametric Fit
        y_fit = A * (eta ** B)
        ax_bottom.plot(x_bottom, y_fit, linestyle='--', color=color, linewidth=3, alpha=0.9)

    else:
        # --- BIAS-VARIANCE FIT for SINDy/Poly ---
        p0 = [np.min(y_data), 1.0]
        
        try:
            popt, _ = curve_fit(bias_variance_model, eta, y_data, p0=p0, bounds=(0, np.inf))
            bias_est, slope_est = popt
            
            print(f"{label_map_f2[col]:<20} | Bias-Var Fit | Bias={bias_est:.4e}, Slope={slope_est:.4f}")
            
            # Generate smooth curve for plotting
            x_smooth = np.logspace(np.log10(x_bottom.min()), np.log10(x_bottom.max()), 100)
            eta_smooth = x_smooth / 100.0
            y_fit = bias_variance_model(eta_smooth, bias_est, slope_est)
            
            ax_bottom.plot(x_smooth, y_fit, linestyle='--', color=color, linewidth=3, alpha=0.9)
            
        except Exception as e:
            print(f"Fit failed for {col}: {e}")

ax_bottom.set_xlabel(r'Percentage of noise, $100\,\eta$ (%)', fontsize=fontsize)
ax_bottom.set_ylabel(r'RMSE [ f$_2$($x$) ]  (N)', fontsize=fontsize)
# ax_bottom.axhline(y=20.0, color='darkred', linestyle='--')

# --- LEGEND REORDERING ---
handles, labels = ax_bottom.get_legend_handles_labels()
desired_order = ['SINDy-CC', 'Poly-CC', 'Parametric']
label_handle_map = dict(zip(labels, handles))
ordered_handles = [label_handle_map[l] for l in desired_order if l in label_handle_map]
ordered_labels = [l for l in desired_order if l in label_handle_map]

#ax_bottom.legend(ordered_handles, ordered_labels, loc='lower right', labelspacing=0.1, fontsize=fontsize)
ax_bottom.legend(ordered_handles, ordered_labels,loc='lower right', labelspacing=0.1, fontsize=fontsize,frameon=False,markerfirst=False,handletextpad=0.1)
ax_bottom.text(0.02, 0.98, '(d)', transform=ax_bottom.transAxes, fontsize=20, verticalalignment='top')

# Log scales and Formatting (Same as Plot 1)
ax_bottom.set_xscale('log')
ax_bottom.set_yscale('log')

ax_bottom.tick_params(axis='x', which='major', direction='in', length=12, top=True, labelsize=fontsize)
ax_bottom.tick_params(axis='x', which='minor', direction='in', length=4, top=True)
ax_bottom.tick_params(axis='y', which='major', direction='in', length=12, right=True, labelsize=fontsize)
ax_bottom.tick_params(axis='y', which='minor', direction='in', length=4, right=True)

ax_top = ax_bottom.twiny()
ax_top.yaxis.set_major_locator(LogLocator(base=10.0))
ax_top.yaxis.set_major_formatter(ScalarFormatter())
ax_top.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.1, 10) / 1.0, numticks=10))
ax_top.yaxis.set_minor_formatter(NullFormatter())

ax_bottom.set_ylim(1e-4, 1e1)
ax_bottom.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
ax_bottom.yaxis.set_minor_locator(FixedLocator(minor_tick_positions))
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
plt.savefig("Fig11d_rmse_f2.pdf")
plt.show()
