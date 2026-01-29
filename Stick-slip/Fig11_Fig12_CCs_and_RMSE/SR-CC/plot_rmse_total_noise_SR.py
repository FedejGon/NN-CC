import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter, FixedLocator, FuncFormatter
from matplotlib.ticker import LogFormatterSciNotation

# ==========================================
# CONFIGURATION
# ==========================================
FIX_B_TO_ONE = True  # Set to True to force B=1, False for free fit

# filename = 'rmse_results_duffing_NN_without_symmetry.txt'
filename = 'rmse_noise_duffing_SR_CC.txt'

fontsize = 16

col_names = [
    "noise_perc_th", "noise_perc", "SNR_dB", "xSR", "vSR", "xSR-CC", "vSR-CC", "xParam", "vParam"
]

# Load Data
df = pd.read_csv(filename, skiprows=53, delim_whitespace=True, header=None, names=col_names)
grouped = df.groupby('noise_perc_th').mean().reset_index()

x_bottom = grouped['noise_perc_th']
x_top = grouped['SNR_dB']

# ---------------------------------------------------------
# Define Eta for Fitting
# eta = noise_percentage / 100
# ---------------------------------------------------------
eta = x_bottom / 100.0
log_eta = np.log(eta)

desired_top_labels = [40, 30, 20, 10, 0, -10, -20]
tick_positions = []
tick_labels = []
for label in desired_top_labels:
    idx = (np.abs(x_top - label)).argmin()
    pos = x_bottom.iloc[idx]
    tick_positions.append(pos)
    tick_labels.append(str(label))

fig, ax_bottom = plt.subplots(figsize=(6, 6))

plot1_cols = ['xSR', 'xSR-CC' , 'xParam']
label_map = {
    'xSR': 'SR',
    'xSR-CC': 'SR-CC',
    'xParam': 'Parametric',
}

plot_colors = ['darkred', 'darkslateblue','black']
plot_markers = ['s', 'H','o']
plot_ms = [ 5, 6, 5]

# =========================================================
# PERFORM FITTING AND PLOTTING
# =========================================================
print("--- Fits for x(t) (RMSE = A * eta^B) ---")
if FIX_B_TO_ONE:
    print("(NOTE: B is fixed to 1.0)")
print(f"{'Method':<20} | {'A (Prefactor)':<15} | {'B (Exponent)':<15}")
print("-" * 55)

for col, color, marker, ms in zip(plot1_cols, plot_colors, plot_markers, plot_ms):
    y_data = grouped[col]

    # 1. Plot original data (markers and solid lines)
    # Note: kept linewidth=2 to connect dots as per original script. 
    # If you only want markers, set linestyle='None'.
    ax_bottom.plot(x_bottom, grouped[col], marker=marker, markersize=ms, 
                   color=color, label=label_map[col], linewidth=2)

    # 2. Prepare Data for Fitting
    log_y = np.log(y_data)

    # --- SR MODEL CONSTRAINT: Fit only on x <= 800 (masking high noise for SR) ---
    if col == 'xSR':
        mask = x_bottom <= 800.0
        fit_log_eta = log_eta[mask]
        fit_log_y = log_y[mask]
    else:
        fit_log_eta = log_eta
        fit_log_y = log_y

    # 3. Perform Fit
    if FIX_B_TO_ONE:
        B = 1.0
        # log(y) = log(A) + 1 * log(eta)  => log(A) = mean(log(y) - log(eta))
        log_A = np.mean(fit_log_y - fit_log_eta)
        A = np.exp(log_A)
    else:
        # Linear regression on log-log data
        if len(fit_log_eta) > 0:
            coeffs = np.polyfit(fit_log_eta, fit_log_y, 1)
            B = coeffs[0]
            log_A = coeffs[1]
            A = np.exp(log_A)
        else:
            A, B = 0, 0

    print(f"{label_map[col]:<20} | {A:<15.4e} | {B:<15.4f}")

    # 4. Plot Fitted Curve (Dashed lines)
    # Generate smooth x points for drawing the line
    x_fit_smooth = np.geomspace(x_bottom.min(), 1000, 50)
    eta_fit_smooth = x_fit_smooth / 100.0
    y_fit_smooth = A * (eta_fit_smooth ** B)

    # Plot the dashed fit line
    ax_bottom.plot(x_fit_smooth, y_fit_smooth, linestyle='--', color=color, linewidth=2, alpha=0.9)


ax_bottom.set_xlabel(r'Percentage of noise, $100\,\eta$ (%)', fontsize=fontsize)
ax_bottom.set_ylabel('RMSE [x(t)]  (m)', fontsize=fontsize)
ax_bottom.axhline(y=20.0, color='red', linestyle='--')

# Legend
ax_bottom.legend(loc='lower right', labelspacing=0.1, fontsize=fontsize, frameon=False, markerfirst=False, handletextpad=0.15)

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
ax_bottom.set_ylim(1e-4, 1e1)

ax_bottom.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))

# Custom formatter for minor ticks
def minor_tick_formatter(x, pos):
    if abs(x - 2) < 0.1 or abs(x - 5) < 0.1 or abs(x - 20) < 0.1 or abs(x - 50) < 0.1:
        return str(int(round(x)))
    else:
        return ''

ax_bottom.yaxis.set_minor_formatter(FuncFormatter(minor_tick_formatter))

# Enable minor tick labels
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
ax_bottom.text(0.02, 0.98, '(c)', transform=ax_bottom.transAxes, fontsize=20, verticalalignment='top')

# Force minor ticks to be visible
ax_bottom.minorticks_on()

# Double-check minor tick settings
ax_bottom.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[2, 3, 4, 5, 6, 7, 8, 9]))
ax_bottom.yaxis.set_minor_formatter(NullFormatter())
ax_bottom.tick_params(axis='y', which='minor', direction='in', length=4, right=True)

plt.tight_layout()
plt.savefig("Fig12c_rmse_total_noise.pdf")
plt.show()
