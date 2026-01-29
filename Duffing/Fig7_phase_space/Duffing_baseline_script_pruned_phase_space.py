
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
from sklearn.linear_model import LinearRegression
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
epochs_max = 5000
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
SNR_dB_list = [np.inf] + list(np.linspace(40, 5, 36 ))  # ∞, 20, 17.5, ..., -5
#SNR_dB_list = list(np.linspace(-18, -20, 3 ))  # ∞, 20, 17.5, ..., -5

SNR_dB_list = list(np.linspace(5, -20, 26 ))  # ∞, 20, 17.5, ..., -5

SNR_dB_list = [20.0]

#repeat 3 times each value in the list
SNR_dB_list = np.repeat(SNR_dB_list, 1)


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
    # other integration methods:
    #, method='BDF', rtol=1e-6, atol=1e-8, dense_output=True)
    #, method='DOP853', rtol=1e-9, atol=1e-12)
    #, method='Radau', rtol=1e-6, atol=1e-8
    #verify that integration was succesful
    print(sol.status)   # 0 = success, 1 = reached event, -1 = failed
    print(sol.message)

    # extract variables 
    x_data = sol.y[0]      
    x_dot_data = sol.y[1]  
    time_data = sol.t       
    x_ddot_data = np.array([eq_2nd_ord_veloc(t, y)[1] for t, y in zip(sol.t, sol.y.T)])
    F1_th=F1(x_dot_data)
    F2_th=F2(x_data)
    # linear range of data to plot identified CCs 
    x_vals = np.linspace(np.min(x_data), np.max(x_data), NevalCC)
    xdot_vals = np.linspace(np.min(x_dot_data), np.max(x_dot_data), NevalCC)


# Calculate limits for the gray box (Training Domain)
    x_min_train, x_max_train = np.min(x_data), np.max(x_data)
    v_min_train, v_max_train = np.min(x_dot_data), np.max(x_dot_data)
    
    width_train = x_max_train - x_min_train
    height_train = v_max_train - v_min_train

    # plot phase space
    

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    # Draw the gray box representing the training domain
    # xy is the bottom-left corner
    rect_train = Rectangle((x_min_train, v_min_train), width_train, height_train, 
                           linewidth=1, edgecolor='none', facecolor='gray', alpha=0.3, label='Training data range')
    ax.add_patch(rect_train)
    # X: Major every 1.0, Minor every 0.2
    # Y: Major every 0.5, Minor every 0.1
    custom_ticks(ax, 1.0, 0.5, 0.5, 0.25)
    plt.plot(x_data, x_dot_data, 'k-', linewidth=1.5, label='Training trajectory')
    plt.xlabel("x",fontsize=24)
    plt.ylabel(r"$\dot{x}$",fontsize=24)
    #plt.title("Phase Space: Training Data (Theoretical)")
    plt.grid(True, alpha=0.3)
    plt.xlim(-1.7,1.7)
    plt.ylim(-1.1,1.5)
    ax.text(0.96, 0.04, "(a)", transform=ax.transAxes, fontsize=24, va='bottom', ha='right')
    plt.legend(loc='upper center',fontsize=18) # Add legend to see the 'Training Domain' label
    plt.tight_layout()
    plt.savefig("Fig7a_phase_space_train.pdf")
    plt.show()

    
    # plot theoretical integrations
    plt.figure()
    plt.title("Theoretical ODE integration: Consistency Check")
    plt.plot(time_data, (x_ddot_data + F1_th + F2_th - F_ext(time_data))**2)
    plt.xlabel("t")
    plt.ylabel(r"MSE $(\ddot{x} + F_1(\dot{x}) + F_2(x) - F_{ext})^2$")#$(\ddot{x} - \ddot{x}_{model})^2$")
    plt.grid(True, alpha=0.3)
    plt.show()
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
    

    # Add noise to F_ext with a given SNR (in dB)
    if np.isinf(SNR_dB):
        print("Running with SNR = ∞ dB (no noise)")
        #print("noise=",noise)
        F_ext_noise_data = F_ext(time_data) 
        noise_percentage=0.0
        noise_percentage_th=0.0
    else:
        print(f"Running with SNR = {SNR_dB:.2f} dB")
        # Add noise based on a predefined SNR_dB
        Fext_signal_power = np.mean(F_ext(time_data)**2)
        noise_power = Fext_signal_power / (10**(SNR_dB / 10))
        noise_std = np.sqrt(noise_power)
        #add noise  
        F_ext_val_noisy = F_ext(time_data) + np.random.normal(0, noise_std, size=time_data.shape)
        #compute measured noise
        Fext_noise_substraction = F_ext_val_noisy - F_ext(time_data)
        signal_power = np.mean(F_ext(time_data)**2)
        noise_power = np.mean(Fext_noise_substraction**2)
        snr_measured = 10 * np.log10(signal_power / noise_power)
        # Compute noise percentage relative to RMS signal
        signal_rms = np.sqrt(signal_power)
        noise_rms= np.sqrt(noise_power)
        noise_percentage_th=100*10**(-SNR_dB / 20.0)
        noise_percentage = 100 * (noise_rms / signal_rms)
        print(f"Desired SNR in Fext: {SNR_dB} dB")
        print(f"Measured SNR in Fext: {snr_measured:.2f} dB")
        print(f"Desired noise percentage in Fext: {noise_percentage_th:.2f}%")
        print(f"Measured noise percentage in Fext: {noise_percentage:.2f}%")
        
        
        # --- now apply a Savitzky–Golay filter (not used, only for testing) ---
        # choose an odd window length and a small polynomial order
        window_length = 51    # must be odd, e.g. 5, 11, 51, …
        polyorder     = 3     # < window_length
        F_ext_filtered = savgol_filter(
            F_ext_val_noisy,
            window_length=window_length,
            polyorder=polyorder,
            mode='interp'       # avoids edge artifacts
        )
        # measure the SNR *after* filtering (optional)
        noise_after = F_ext_filtered - F_ext(time_data)
        snr_after   = 10 * np.log10(
            np.mean(F_ext(time_data)**2) / np.mean(noise_after**2)
        )
        print(f"SNR after SG filter: {snr_after:.1f} dB")
        plt.figure(figsize=(6, 4))
        plt.plot(time_data, F_ext(time_data),         label='Fext (true)')
        plt.plot(time_data, F_ext_val_noisy,          label='Fext + noise', alpha=0.7)
        plt.plot(time_data, F_ext_filtered,           label='SG-filtered', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel(r'F$_{ext}$(t)')
        plt.title('Original vs Noisy vs SG-Filtered Forcing')
        plt.legend()
        plt.tight_layout()
        plt.show()
        # Here we can select different options for the noisy Fext 
        #F_ext_noise_data = F_ext(time_data) 
        #F_ext_noise_data = F_ext_filtered
        F_ext_noise_data = F_ext_val_noisy

    
    # plot Training dataset
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(time_data, x_data, color='black')
    ax1.set_ylabel("x(t)")
    ax1.set_title("Theoretical Data (with noise) : Training dataset")
    ax1.grid(True)
    ax2.plot(time_data, F_ext_noise_data, color='black', linestyle='-')
    ax2.set_xlabel("t")
    ax2.set_ylabel(r"F$_{ext}$(t)")
    ax2.grid(True)
    plt.tight_layout()
    plt.show()   
    # plot 
    plt.figure()
    plt.title(r"MSE of F$_{ext}$ with and without noise")
    plt.plot(time_data, (F_ext(time_data) - F_ext_noise_data )**2) 
    plt.xlabel("t")
    plt.ylabel(r"Squared Error (F$_{ext}^{noiseless}$ - F$_{ext}^{noise}$)$^2$")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # range of training data values
    print("min(x) , max(x)=", np.min(x_data),",",np.max(x_data))
    print(f"min(ẋ) , max(ẋ)= {np.min(x_dot_data)} , {np.max(x_dot_data)}")

    ############################################
    ########### IDENTIFICATION #################
    ############################################




    ####################################################
    ############# Parametric-CC  #################   least squares
    # Right-hand side
    rhs = F_ext_noise_data -  x_ddot_data 
    # Design matrix: [x_dot, x, x^3]
    A = np.vstack([
        x_dot_data,
        x_data,
        x_data**3
    ]).T  # shape: (N, 3)
    # Solve least squares: A x [delta, alpha, beta] = rhs
    start = time.time()
    params, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    delta_ident_param, alpha_ident_param, beta_ident_param = params
    print("Parametric-CC model")
    end = time.time()  
    elapsed = end - start
    print(f"Training finished in {elapsed:.3f} seconds")
    print("Theoretical params:")
    print(f"delta = {delta:.6e}, alpha = {alpha:.6e}, beta = {beta:.6e}")
    print(" ")
    print("Identified params from Parametric-CC:")
    print(f"delta = {delta_ident_param:.6e}, alpha = {alpha_ident_param:.6e}, beta = {beta_ident_param:.6e}")

    #defining the Parametric-CC functions for forward simulations
    def ode_param(t, state):
        x, xdot = state
        xddot = (F_ext(t) - delta_ident_param*xdot - alpha_ident_param*x - beta_ident_param*x**3) # / m
        return [xdot, xddot]
    def f1_param(x_dot):
        return delta_ident_param * x_dot
    def f2_param(x):
        return alpha_ident_param * x + beta_ident_param * x**3
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
    ax1.plot(xdot_vals, f1_param(xdot_vals), label="Identified", linewidth=2)
    ax1.plot(xdot_vals, F1(xdot_vals), '--',color='black', label="Theor.", linewidth=2)
    ax1.set_title("Obtained CCs from Parametric-CC method")
    ax1.set_xlabel(r"$\dot{x}$")
    ax1.set_ylabel(r"f$_1(\dot{x})$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(x_vals, f2_param(x_vals), label="Identified", linewidth=2)
    ax2.plot(x_vals, F2(x_vals), '--',color='black', label="Theor.", linewidth=3)
    ax2.set_ylabel(r"f$_2$(x)")
    ax2.set_xlabel("x")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    # Integrate forward the obtained Parametric-CC model for verification
    # (using the same ICs and driven force as the training data)
    sol = solve_ivp(ode_param, t_span, y0, t_eval=t_simul,method='LSODA') 
    x_sim = sol.y[0]
    xdot_sim = sol.y[1]
    plt.figure(figsize=(8,5))
    plt.plot(time_data, x_sim, label="Parametric-CC", linewidth=2)
    plt.plot(time_data, x_data, "--", color='black', label="Training data", linewidth=2)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.title("Simulation with Parametric-CC model")
    plt.show()
    
    




    ################################################################################
    ################################################################################
    ################################   VALIDATION   ################################
    ################################################################################
    ################################################################################



    print('Validating the system')
    print('Integration of model EDOs')
    n_trials = 10 # number of random initial conditions
    # 1. Define bounds
    x_min_train, x_max_train = np.min(x_data), np.max(x_data)
    v_min_train, v_max_train = np.min(x_dot_data), np.max(x_dot_data)
    width_train = x_max_train - x_min_train
    height_train = v_max_train - v_min_train

    # 2. Lists to store ALL points from valid trajectories
    all_x_points = []
    all_v_points = []
    
    # --- SETUP phase space plot ---
    fig_phase, ax_phase = plt.subplots(figsize=(6, 6))
    rect_val = Rectangle((x_min_train, v_min_train), width_train, height_train, 
                         linewidth=1, linestyle='--', facecolor='gray', alpha=0.3, label='Training data range') #edgecolor='k'
    #rect_val = Rectangle((x_min_train, v_min_train), width_train, height_train, 
    #                     linewidth=1.5, edgecolor='#444444', linestyle='--', 
    #                     facecolor='none', zorder=5, label='Training Domain')
    ax_phase.add_patch(rect_val)
    ax_phase.set_xlabel("x",fontsize=24)
    ax_phase.set_ylabel(r"$\dot{x}$",fontsize=24)
    #ax_phase.set_title(f"Phase Space: {n_trials} Validation Simulations")
    ax_phase.grid(True, alpha=0.3)
    # Choose a colormap (e.g., 'viridis', 'plasma', 'jet', 'tab20','Greys')
    #colormap = plt.get_cmap('copper')
    # -------------------------------
  

#    for i in range(n_trials):
# Validation Loop Variables
    n_trials = 20        # Target number of valid simulations
    valid_trials = 0     # Counter for valid simulations
    attempts = 0         # Counter for total attempts (to prevent infinite loops if needed)
    max_attempts = 100000   # Safety break

    print(f"Starting search for {n_trials} valid simulations inside the training range...")

    while valid_trials < n_trials:
        attempts += 1
        if attempts > max_attempts:
            print("Max attempts reached. Stopping validation.")
            break


        # Random initial conditions uncomment
        x0_val = np.round(np.random.uniform(-0.5, 0.5),3)
        v0_val = np.round(np.random.uniform(-0.5, 0.5),3)
        y0_val = [x0_val, v0_val]
        Aext = np.round(np.random.uniform(0.45, 0.5),3)
        Omega = np.round(np.random.uniform(1.1, 1.3),3)
        
        
        
        #Aext = np.round(np.random.uniform(0.1, 0.5),3)
        #Omega = np.round(np.random.uniform(1.1, 1.3),3)
        
        #A = np.round(np.random.uniform(1.0, 1.5),3)
        #Omega = np.round(np.random.uniform(0.2, 0.4),3)
        #x0_val=x0-0.05
        #v0_val=v0+0.01
               
        #Aext=0.3
        #alpha=-1.0
        #beta=1.0
        #delta=0.3
        #Omega=1.2
        #x0=0.5
        #v0=-0.5
        #x0_val=x0
        #y0_val=y0
        #y0=[x0,v0]
        #y0_val=y0
        

        #x0_val=-0.8 #x0
        #v0_val=0.7 #v0
        #y0_val = [x0_val,v0_val]
        #y0_val = y0
        
        
        #kval = np.random.uniform(0.5, 1.)
        #cval = np.random.uniform(0.1, 0.5)
        #m = 1.0
        #mu_N = np.random.uniform(0.5, 1.0)
        #print(f"Trial {i+1} : x0={x0_val:.4f} ; v0={v0_val:.4f}")
        #print(r"$\A$="+f"{A:.4f} ; "+r"$\Omega$"+f"={Omega:.4f}")
        print(f"Trial {attempts} :")
        #print(f"alpha={alpha}, c={cval}")
        #print(f"$\Omega$={Omega}, $\mu$*N={mu_N}, $x_0$={x0}, $v_0$={v0}")
        #print(f"Omega={Omega}, A={A}, $x_0$={x0_val:.4f}, $v_0$={v0_val:.4f}")

        #kval = np.round(np.random.uniform(1, 1.5),3)
        #cval = np.round(np.random.uniform(0.1, 0.5),3)
        #m = 1.0
        #mu_N = np.round(np.random.uniform(0.5, 1.0),3)
        #Omega = np.round(np.random.uniform(0.2, 0.5),3)
        #x0 = np.round(np.random.uniform(-0.5, 0.5),3)
        #v0 = np.round(np.random.uniform(-0.5, 0.5),3)



        ################ Theoretical Eq ###### validation of the model
        print("Integrating Theor.")
        start = time.time()  
        sol_val = solve_ivp(eq_2nd_ord_veloc, t_span_val, y0_val, t_eval=t_val,method='LSODA')
        end = time.time()  
        elapsed = end - start
        print(f"Solve_ivp finished in {elapsed:.3f} seconds")
        t_simulated_th = sol_val.t
        x_simulated_th = sol_val.y[0]       
        x_dot_simulated_th = sol_val.y[1]   
        
        #plt.figure()
        #plt.plot(t_simulated_th, x_simulated_th, label="Validation: Theor. Integration")
        #plt.legend()
        #plt.xlabel("t")
        #plt.ylabel("x(t)")
        #plt.show()

        if np.max(x_simulated_th)>np.max(x_data):
            print("Extrapolation! max(x_sim)>max(x_train) :",np.max(x_simulated_th),">",np.max(x_data))
        if np.min(x_simulated_th)<np.min(x_data):
            print("Extrapolation! min(x_sim)<min(x_train) :",np.min(x_simulated_th),"<",np.min(x_data))

        # --- CHECK BOUNDS ---
        # Check if ANY point in the simulation goes outside the training box
        is_out_x = np.any(x_simulated_th < x_min_train) or np.any(x_simulated_th > x_max_train)
        is_out_v = np.any(x_dot_simulated_th < v_min_train) or np.any(x_dot_simulated_th > v_max_train)
        
        if is_out_x or is_out_v:
            # If outside, skip this trial (do not increment valid_trials)
            print(f"Attempt {attempts}: Out of bounds. Retrying...") 
            continue 
        
        # If we reach here, the curve is strictly inside the box
        valid_trials += 1
        print(f"Trial {valid_trials}/{n_trials} found (Attempt {attempts})")
        
        # Get a unique color based on the loop index
        #line_color = colormap(attempts / n_trials)
       
        # --- ADD LINE ---
        # Use a slightly lower alpha and thinner line for cleaner look
        #ax_phase.plot(x_simulated_th, x_dot_simulated_th, color=line_color, alpha=0.8, linewidth=1.5)
        
        # Plot a dot at the initial condition (zorder ensures it's on top of lines)
        #ax_phase.scatter(x0_val, v0_val, color=line_color, s=35, marker='o', edgecolors='k', zorder=10)

        # contour plot:
        #ax_phase.plot(x_simulated_th, x_dot_simulated_th, alpha=0.8, linewidth=1.5)
        #ax_phase.scatter(x0_val, v0_val,  s=35, marker='o', edgecolors='k', zorder=10)
        #spaghetti plot 
        ax_phase.plot(x_simulated_th, x_dot_simulated_th, color='k', alpha=0.5, linewidth=1.0)

        # Save data if valid
        all_x_points.append(x_simulated_th)
        all_v_points.append(x_dot_simulated_th)
        
        ################ parametric-CC ###### validation of the model
        sol_parametric = solve_ivp(ode_param, t_span_val, y0_val, t_eval=t_val,method='LSODA')
                        #rtol=1e-9, atol=1e-12, max_step=(t_eval[1] - t_eval[0]))   
        t_parametric= sol_parametric.t
        x_simulated_parametric = sol_parametric.y[0]
        x_dot_simulated_parametric = sol_parametric.y[1]
        #plt.figure(figsize=(8,5))
        #plt.plot(t_parametric, x_simulated_parametric, "--", label="Parametric-CC", linewidth=2)
        #plt.plot(t_simulated_th, x_simulated_th, label="Theor.", linewidth=2)
        #plt.xlabel("t")
        #plt.ylabel("x(t)")
        #plt.legend()
        #plt.title("Parametric model simulation")
        #plt.show()
        
# --- SHOW FINAL PLOT ---
    # Add a dummy colorbar just to show the range of trials (optional)
    #sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=1, vmax=n_trials))
    #cbar = plt.colorbar(sm, ax=ax_phase)
    #cbar.set_label('Trial Number')
    
    #sm = plt.cm.ScalarMappable( norm=plt.Normalize(vmin=1, vmax=n_trials))
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='k', lw=1.5, alpha=0.5),
                    rect_val]
    ax_phase.legend(custom_lines, ['Validation trajectories', 'Training data range'], loc='upper center', frameon=True, fontsize=18)
    ax_phase.grid(True, alpha=0.2)
    custom_ticks(ax_phase, 1.0, 0.5, 0.5, 0.25)
    ax_phase.set_xlim(-1.7,1.7)
    ax_phase.set_ylim(-1.1,1.5)
    ax_phase.text(0.96, 0.04, "(b)", transform=ax_phase.transAxes, fontsize=24, va='bottom', ha='right')
    plt.figure(fig_phase.number)
    plt.tight_layout()
    plt.savefig("Fig7b_phase_space_10_trajs.pdf")
    plt.show()
    
    
    
# 4. Concatenate all data
    #x_flat = np.concatenate(all_x_points)
    #v_flat = np.concatenate(all_v_points)
    #fig_phase, ax_phase = plt.subplots(figsize=(8, 6))
    #from matplotlib.patches import Rectangle
    #rect_val = Rectangle((x_min_train, v_min_train), width_train, height_train, 
    #                     linewidth=2, edgecolor='k', linestyle='--', facecolor='none', label='Training Domain')
    #ax_phase.add_patch(rect_val)
    #print("Calculating density map...")
    ## Calculate the point density
    #xy = np.vstack([x_flat, v_flat])
    #z = gaussian_kde(xy)(xy)
    ## Sort the points by density, so that the densest points are plotted last
    #idx = z.argsort()
    #x_flat, v_flat, z = x_flat[idx], v_flat[idx], z[idx]

    ## Option A: Contour Plot (Smooth Level Curves)
    ## Create a grid for contouring
    #x_grid = np.linspace(x_min_train, x_max_train, 100)
    #v_grid = np.linspace(v_min_train, v_max_train, 100)
    #Xgrid, Vgrid = np.meshgrid(x_grid, v_grid)
    #positions = np.vstack([Xgrid.ravel(), Vgrid.ravel()])
    #kernel = gaussian_kde(xy)
    #Zgrid = np.reshape(kernel(positions).T, Xgrid.shape)

    ## Plot filled contours (The "Map")
    ## cmap='Greys' makes it look like a B&W density map. Use 'viridis' for color.
    #cf = ax_phase.contourf(Xgrid, Vgrid, Zgrid, levels=15, cmap='Greys')
    ## Optional: Add contour lines on top for definition
    #ax_phase.contour(Xgrid, Vgrid, Zgrid, levels=15, colors='k', linewidths=0.5, alpha=0.5)
    ## Add colorbar for density
    #cbar = plt.colorbar(cf, ax=ax_phase)
    #cbar.set_label('Trajectory Density')
    #ax_phase.set_xlabel("x")
    #ax_phase.set_ylabel(r"$\dot{x}$")
    ##ax_phase.set_title(f"Density Map of {valid_trials} Valid Trajectories")
    #ax_phase.grid(True, alpha=0.3)
    #ax_phase.set_xlim(-1.7,1.7)
    #ax_phase.set_ylim(-1.1,1.1)
    #plt.tight_layout()
    #plt.show()    
