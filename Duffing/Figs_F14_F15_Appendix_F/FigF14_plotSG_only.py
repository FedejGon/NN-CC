
##!pip install pysindy
# Van der Pol RN with Fext
# Importar bibliotecas necesarias
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#import pysindy as ps
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import solve_ivp
import pysindy as ps
print(ps.__version__)
#from pysindy.feature_library import PolynomialLibrary
#from pysindy.feature_library import CustomLibrary
#from pysindy.feature_library import ParameterizedLibrary
#from pysindy.feature_library import IdentityLibrary
#from pysindy.optimizers import ConstrainedSR3
#from pysindy import AxesArray
#from pysindy.optimizers import STLSQ
#from pysindy import SINDy
#from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy.polynomial.legendre import legvander
from scipy.signal import savgol_filter
from scipy.special import comb  # For binomial coefficients
from scipy.stats import gaussian_kde
import os
import copy
import time
from matplotlib.patches import Rectangle # Import Rectangle
#from google.colab import drive
#drive.mount('/content/drive')
#output_path = "/content/drive/My Drive/Colab Notebooks/Second_order_noise/Python"
#output_path = "/content/drive/Shared with me/Federico2024_System_Identification/Python"
output_path = "./"
output_file_log = open("output_log.txt", "w")

#from pysr import PySRRegressor
#import sympy as sp

# Check for GPU availability
if torch.cuda.is_available():
    print("GPU is available, using GPU")
else:
    print("GPU is not available, using CPU.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Free GPU memory if using CUDA
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
# Initialization of random number generators for reproducibility
np.random.seed(0) 
torch.manual_seed(10)  # any integer


# Place this definition before the plotting sections
def custom_ticks(ax, major_x_interval, major_y_interval, minor_x_interval, minor_y_interval):
    # Set major ticks
    ax.xaxis.set_major_locator(ticker.MultipleLocator(major_x_interval))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(major_y_interval))
    # Set minor ticks
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_x_interval))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(minor_y_interval))
  
    # Customize tick appearance (labelsize=24 requires larger axis labels too)
    ax.tick_params(axis='x', direction='in', which='major', length=8, width=1.5, labelsize=24, top=True, bottom=True)
    ax.tick_params(axis='y', direction='in', which='major', length=8, width=1.5, labelsize=24, left=True, right=True)
    ax.tick_params(axis='x', direction='in', which='minor', length=5, width=1, top=True, bottom=True)
    ax.tick_params(axis='y', direction='in', which='minor', length=5, width=1, left=True, right=True)
  
print("ODE: x'' + f1(x') + f2(x) = F_ext(t)")
print("Duffing System")
print("f1(x')= delta x'")
print("f2(x)= alpha x + beta x^3")
print("F_ext(t)=Aext cos(Omega t)")

#parameters stick slip
#m=1.0 # kg
#cval=0.1 # Ns/m (viscous damping coefficient)
#kval=1.0 # N/m (stiffness)
#Aext=2 # N (forcing amplitude)
#Omega=0.3 # 0.3 and 0.15 rad/s (forcing frequency)
#x0=0.1 # m (initial displacement)
#v0=0.1 # m/s (initial velocity)
#mu_N = 0.5 #0.5
#m=1.0

#parameters duffing
Aext=0.5
alpha=-1.05
beta=1.02
delta=0.3
Omega=1.2
x0=0.5
v0=-0.5
y0 = [x0, v0]  # [x(0), x'(0)]

Tsimul=40 # for generating the training database
Nsimul=1000
Tval=2*Tsimul # for forward simulations with trained models
Nval=2*Nsimul
NevalCC=1000 # number of evaluating points for the CCs from min to max values
t_span = (0, Tsimul)  # time interval for training dataset
t_simul = np.linspace(*t_span, Nsimul)  
t_span_val = (0, Tval)  # time interval for forward simulation
t_val = np.linspace(*t_span_val, Nval)   

# parameters for evaluating CCs for extrapolation
n_window = 50 # number of points to calculate the envelope around each edge
range_interp = 0.2 # percentage of values near edges for obtaining the envelope
range_extrap = 0.5  # percentage of extrapolated range with respect to total range
n_extra = 50 # number of new data points for each direction
deg_extrap = 1 # degree of polynomial extrapolation

# Hyperparameters for NN-CC methods
learning_rate = 1e-4
epochs_max = 20000
neurons=50
error_threshold = 1e-8
f1_symmetry='odd'
f2_symmetry='odd'
lambda_penalty = 1e-4  # You can adjust this weight if needed
lambda_penalty_symm = 1e1
apply_restriction=False #True
N_constraint = 1000 # number of points for evaluating symmetry constraints
#for testing other activation functions
#weight_decay = 1e-6 # 0.0 # 1e-6 was the better, 0.0 default
#momentum=0.99


#SNR_dB_list = [np.inf] + list(np.linspace(40, -20, 61 ))  # ∞, 20, 17.5, ..., -5
SNR_dB_list = [np.inf] + list(np.linspace(40, 0, 41 ))  # ∞, 20, 17.5, ..., -5
SNR_dB_list =  list(np.linspace(40, 0, 41 ))  # ∞, 20, 17.5, ..., -5
#SNR_dB_list = list(np.linspace(-18, -20, 3 ))  # ∞, 20, 17.5, ..., -5

#SNR_dB_list = list(np.linspace(5, -20, 26 ))  # ∞, 20, 17.5, ..., -5
SNR_dB_list = [20.0]

#repeat 3 times each value in the list
#SNR_dB_list = np.repeat(SNR_dB_list, 10)


#SNR_dB_list = np.repeat(SNR_dB_list,B_list = list(np.linspace(5, -5, 3))  # ∞, 20, 17.5, ..., -5


for SNR_dB in SNR_dB_list:
    #SNR_dB = -5.0
    #alpha=-1.0
    #beta=1.0
    #delta=0.3
    #x0=0.5
    #v0=-0.5
    #Aext=2.0
    #Omega=1.2
    
    print(f"SNR_dB={SNR_dB}")    
    print(f"alpha={alpha}, beta={beta}, delta={delta}")
    print(f"Omega={Omega}, Aext={Aext}, x₀={x0}, v₀={v0}")
    
    #Definition of the theoretical functions
    def F1(x_dot):
        return delta * x_dot 
    def F2(x):
        return alpha*x+beta*x**3 
    def F_ext(t):
        return Aext*np.cos(Omega*t)
    def eq_2nd_ord_veloc(t,y):
        x, x_dot = y  # y=[x, x']
        x_ddot = (F_ext(t) - F1(x_dot) - F2(x))*1.0
        return [x_dot, x_ddot]

    # ODE: x'' + F1(x_dot) + F2(x) = F_ext(t) 
    # ODE: x'' + delta x_dot + alpha x + beta x^3 = F_ext(t) 
    # F_ext(t) = Aext cos(Omega t)

    #Integrate forward the theoretical equation to generate training dataset 
    sol = solve_ivp(eq_2nd_ord_veloc, t_span, y0, t_eval=t_simul,method='LSODA') 


    # extract clean variables 
    x_data_clean = sol.y[0]       
    x_dot_data_clean = sol.y[1]   
    time_data = sol.t         
    
    # Calculate theoretical acceleration (for comparison only)
    x_ddot_data_clean = np.array([eq_2nd_ord_veloc(t, y)[1] for t, y in zip(sol.t, sol.y.T)])

    # Theoretical F_ext (Assuming F_ext is known and clean)
    F_ext_data = F_ext(time_data)

    # -------------------------------------------------------------------------
    # 1. ADD NOISE TO x(t)
    # -------------------------------------------------------------------------
    if np.isinf(SNR_dB):
        print("Infinite SNR: Using Clean Data")
        x_noisy = x_data_clean.copy()
        noise_percentage_th=0.0
        noise_percentage=0.0
    else:
        # Calculate signal power (variance of x)
        signal_power = np.var(x_data_clean)
        # Calculate required noise power
        noise_power = signal_power / (10**(SNR_dB / 10))
        noise_std = np.sqrt(noise_power)
        # Add noise
        noise = np.random.normal(0, noise_std, size=x_data_clean.shape)
        x_noisy = x_data_clean + noise
        
        signal_rms = np.sqrt(signal_power)
        noise_rms= np.sqrt(noise_power)
        noise_percentage_th=100*10**(-SNR_dB / 20.0)
        noise_percentage = 100 * (noise_rms / signal_rms)
        # Verify SNR
        measured_snr = 10 * np.log10(signal_power / np.var(noise))
        print(f"Measured SNR on x: {measured_snr:.2f} dB")

    # -------------------------------------------------------------------------
    # 2. COMPUTE DERIVATIVES (Savitzky-Golay)
    # -------------------------------------------------------------------------
    dt = time_data[1] - time_data[0]
    
    # SG Parameters - Tune these based on noise level!
    # Window length must be odd. 
    # High noise -> larger window (smoother, but may cut peaks).
    # Low noise -> smaller window.
    #if SNR_dB < 30:
    #    sg_window = 51 
    #else:
    sg_window = 31
    sg_poly = 3
    
    print(f"Computing derivatives with SG Filter: Window={sg_window}, Poly={sg_poly}")

    # 0th deriv (Smoothed Position)
    x_data = savgol_filter(x_noisy, window_length=sg_window, polyorder=sg_poly, deriv=0)
    
    # 1st deriv (Velocity) - IMPORTANT: delta=dt
    x_dot_data = savgol_filter(x_noisy, window_length=sg_window, polyorder=sg_poly, deriv=1, delta=dt)
    
    # 2nd deriv (Acceleration) - IMPORTANT: delta=dt
    x_ddot_data = savgol_filter(x_noisy, window_length=sg_window, polyorder=sg_poly, deriv=2, delta=dt)

# Plot Differentiation Results
    # Increased figsize to (10, 12) to accommodate labelsize=24
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Subplot (a): Position
    ax[0].plot(time_data, x_data_clean, 'k-', lw=1.5, label='True')
    ax[0].plot(time_data, x_noisy, 'r.', ms=3, alpha=1.0, label='Noisy')
    ax[0].plot(time_data, x_data, 'b--', lw=2, label='SG')
    ax[0].set_ylabel(r"$x(t)$", fontsize=24) # LaTeX label
    # Legend settings
    #ax[0].legend(fontsize=18, frameon=False, loc='lower right')
    ax[0].legend(fontsize=18, frameon=False, loc='upper right', bbox_to_anchor=(0.95, 1.0))
    # Label (a) inside plot
    ax[0].text(0.02, 0.86, '(a)', transform=ax[0].transAxes, fontsize=24 ) #, weight='bold')
    # Custom ticks: X(Major 10, Minor 2), Y(Major 1, Minor 0.5)
    custom_ticks(ax[0], 10, 1.0, 2.0, 0.5)

    # Subplot (b): Velocity
    ax[1].plot(time_data, x_dot_data_clean, 'k-', lw=1.5, label='True')
    ax[1].plot(time_data, x_dot_data, 'b--', lw=2, label='SG')
    ax[1].set_ylabel(r"$\dot{x}(t)$", fontsize=24) # LaTeX label
    # Label (b) inside plot
    ax[1].text(0.02, 0.86, '(b)', transform=ax[1].transAxes, fontsize=24 ) #, weight='bold')
    # Custom ticks: X(Major 10, Minor 2), Y(Major 1, Minor 0.5)
    custom_ticks(ax[1], 10, 1.0, 2.0, 0.5)

    # Subplot (c): Acceleration
    ax[2].plot(time_data, x_ddot_data_clean, 'k-', lw=1.5 ,label='True')
    ax[2].plot(time_data, x_ddot_data, 'b--', lw=2, label='SG')
    ax[2].set_ylabel(r"$\ddot{x}(t)$", fontsize=24) # LaTeX label
    ax[2].set_xlabel("t", fontsize=24)
    # Label (c) inside plot
    ax[2].text(0.02, 0.86, '(c)', transform=ax[2].transAxes, fontsize=24) #, weight='bold')
    # Custom ticks: X(Major 10, Minor 2), Y(Major 5, Minor 1)
    custom_ticks(ax[2], 10, 1.0, 2.0, 0.5)

    plt.tight_layout()
    plt.savefig("FigF14_SG_filter.pdf")
    plt.show()
    
    
############################################
    # Define ranges for plotting CCs later
    x_vals = np.linspace(np.min(x_data), np.max(x_data), NevalCC)
    xdot_vals = np.linspace(np.min(x_dot_data), np.max(x_dot_data), NevalCC)
    F1_th=F1(x_dot_data_clean)
    F2_th=F2(x_data_clean)
##############################################3

    
    # plot theoretical integrations
    plt.figure()
    plt.title("Theoretical ODE integration: Consistency Check")
    plt.plot(time_data, (x_ddot_data + F1_th + F2_th - F_ext(time_data))**2)
    plt.xlabel("t")
    plt.ylabel(r"MSE $(\ddot{x} + F_1(\dot{x}) + F_2(x) - F_{ext})^2$")#$(\ddot{x} - \ddot{x}_{model})^2$")
    plt.grid(True, alpha=0.3)
#    plt.show()
    #plt.plot(time_data, x_data)
    #plt.xlabel("Time")
    #plt.ylabel("x(t)")
    #plt.title("Theoretical data")
    #plt.grid(True)
    #plt.show()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(time_data, x_data, color='black')
    ax1.set_ylabel("x(t)")
    ax1.set_title("Theoretical Data (without noise)")
    ax1.grid(True)
    ax2.plot(time_data, F_ext(time_data), color='black', linestyle='-')
    ax2.set_xlabel("t")
    ax2.set_ylabel(r"F$_{ext}(t)$")
    ax2.grid(True)
    plt.tight_layout()
    plt.show()    
    

    # range of training data values
    print("min(x) , max(x)=", np.min(x_data),",",np.max(x_data))
    print(f"min(ẋ) , max(ẋ)= {np.min(x_dot_data)} , {np.max(x_dot_data)}")

    ############################################
    ########### IDENTIFICATION #################
    ############################################


