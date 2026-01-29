import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter, FixedLocator, FuncFormatter

#filename = 'times_noise_chaos_duffing_NN_with_symmetry.txt_old'
filename='times_noise_chaos_duffing_SR_and_param.txt'
fontsize = 16

col_names = [
    "noise_perc_th", "noise_perc", "SNR_dB", "Param","SR","SR-CC"
]

df = pd.read_csv(filename, skiprows=1, delim_whitespace=True, header=None, names=col_names)
grouped = df.groupby('noise_perc_th').mean().reset_index()

#filename_2 = './SR_and_parametric/times_noise_chaos_duffing_SR_and_param.dat'
#col_names_2 = ["noise_perc_th", "noise_perc", "SNR_dB", "SR", "Parametric"]
#df_2 = pd.read_csv(filename_2, skiprows=6, delim_whitespace=True, header=None, names=col_names_2)
#grouped_2 = df_2.groupby('noise_perc_th').mean().reset_index()
#x_bottom_2 = grouped_2['noise_perc_th']


x_bottom = grouped['noise_perc_th']
x_top = grouped['SNR_dB']

desired_top_labels = [40, 30, 20, 10, 0, -10, -20]
tick_positions = []
tick_labels = []
for label in desired_top_labels:
    idx = (np.abs(x_top - label)).argmin()
    pos = x_bottom.iloc[idx]
    tick_positions.append(pos)
    tick_labels.append(str(label))

#plot1_cols = ['NN-CC', 'NN-CC extrap','NN-CC-SR', 'LS-CC','SINDy-CC','SINDy']
#plot1_cols = ['NN-CC','SINDy-CC','LS-CC','NN-CC extrap','NN-CC-SR','SINDy','SR','Parametric']

#plot1_cols = ['NN-CC','SINDy-CC','LS-CC','NN-CC-SR','SINDy','SR','Parametric']
plot1_cols = ['Param','SR-CC','SR']
label_map = {
    #'NN_f1': 'Neural Network F1',
    #'NN-CC-SR': r'NN-CC$_{+sym+post\!\!-\!\!SR}$',
    'SR': 'SR',
    'SR-CC': 'SR-CC',
    'Param': 'Parametric',
}

plot_colors = ['black', 'darkslateblue', 'darkred']
plot_markers = ['o', 'H', 's']
plot_ms = [5, 6, 5]

#plot_colors = ['black','darkred','blue', 'darkgreen', 'darkorange','gray' ]
#plot_markers = ['o', '^','s', 'D', 'p', '*']
#plot_ms = [5, 6, 5, 5, 6,7]  # List of marker sizes

fig, ax_bottom = plt.subplots(figsize=(6,6))
#for col in plot1_cols:
#    ax_bottom.plot(x_bottom, grouped[col], marker='o', label=col)
for col, color, marker, ms in zip(plot1_cols, plot_colors, plot_markers, plot_ms):
    ax_bottom.plot(x_bottom, grouped[col], marker=marker, markersize=ms, color=color, label=label_map[col],linewidth=2)

#plot1_cols_2 = ['SR', 'Parametric']
#for col in plot1_cols_2:
#    ax_bottom.plot(x_bottom_2, grouped_2[col], marker='o', label=col)


ax_bottom.set_xlabel(r'Percentage of noise, $100\,\eta$ (%)', fontsize=fontsize)
#ax_bottom.set_ylabel('Averaged separation time', fontsize=fontsize)
ax_bottom.set_ylabel(r'Averaged separation time, $\langle t_{sep}\rangle$', fontsize=fontsize)
#ax_bottom.axhline(y=20.0, color='red', linestyle='--')
ax_bottom.axhline(y=7.8, color='red', linestyle='--')
#ax_bottom.legend(fontsize=14,loc='lower left',labelspacing=0.2)
ax_bottom.legend(loc='lower left', labelspacing=0.2, fontsize=fontsize,frameon=False,handletextpad=0.1) #markerfirst=False


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
ax_top.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.1,10)/1.0, numticks=10))
ax_top.yaxis.set_minor_formatter(NullFormatter())

# Set y-axis limits first
ax_bottom.set_ylim(1, 50)

# Major y-axis ticks only at powers of 10: 1, 10, 100
major_ticks = [1, 10, 100]
ax_bottom.yaxis.set_major_locator(FixedLocator(major_ticks))
ax_bottom.yaxis.set_major_formatter(ScalarFormatter())

# Minor y-axis ticks at intermediate values
minor_tick_positions = []
for decade in [1, 10]:
    for sub in [2, 3, 4, 5, 6, 7, 8, 9]:
        pos = decade * sub
        if 1 <= pos <= 100:
            minor_tick_positions.append(pos)

ax_bottom.yaxis.set_minor_locator(FixedLocator(minor_tick_positions))

# Custom formatter for minor ticks - show labels only for specific values
def minor_tick_formatter(x, pos):
    if abs(x - 2) < 0.1 or abs(x - 5) < 0.1 or abs(x - 20) < 0.1 or abs(x - 50) < 0.1:
        return str(int(round(x)))
    else:
        return ''

ax_bottom.yaxis.set_minor_formatter(FuncFormatter(minor_tick_formatter))

# Enable minor tick labels to be shown
ax_bottom.tick_params(axis='y', which='minor',  length=5,labelsize=fontsize)

# Explicitly add text labels for specific values
ax_bottom.text(-0.02, 2, '2', transform=ax_bottom.get_yaxis_transform(), 
               ha='right', va='center', fontsize=fontsize)
ax_bottom.text(-0.02, 5, '5', transform=ax_bottom.get_yaxis_transform(), 
               ha='right', va='center', fontsize=fontsize)
ax_bottom.text(-0.02, 20, '20', transform=ax_bottom.get_yaxis_transform(), 
               ha='right', va='center', fontsize=fontsize)
ax_bottom.text(-0.02, 50, '50', transform=ax_bottom.get_yaxis_transform(), 
               ha='right', va='center', fontsize=fontsize)

# Set bottom x-axis major ticks as powers of 10
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

# Force minor ticks to be visible
ax_bottom.minorticks_on()

# Double-check minor tick settings
ax_bottom.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[2, 3, 4, 5, 6, 7, 8, 9]))
ax_bottom.yaxis.set_minor_formatter(NullFormatter())

# Make sure minor ticks are styled properly
ax_bottom.tick_params(axis='y', which='minor', direction='in', length=4, right=True)

# Add text at relative coordinates (0.1, 0.7)
ax_bottom.text(0.08, 0.57, r'$T_L\approx 7.8$', #r'$t_{sep}^{bound}$', 
               transform=ax_bottom.transAxes, 
               fontsize=fontsize, color='red',
               verticalalignment='center')
ax_bottom.text(0.9, 0.98, '(c)', transform=ax_bottom.transAxes, fontsize=20, verticalalignment='top')



plt.tight_layout()
plt.savefig("Fig8c_time_results.pdf")
plt.show()
