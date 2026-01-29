
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
#from sklearn.preprocessing import PolynomialFeatures
#from numpy.polynomial.legendre import legvander
from scipy.signal import savgol_filter
from scipy.special import comb  # For binomial coefficients
import os
import copy
import time
#from google.colab import drive
#drive.mount('/content/drive')
#output_path = "/content/drive/My Drive/Colab Notebooks/Second_order_noise/Python"
#output_path = "/content/drive/Shared with me/Federico2024_System_Identification/Python"
output_path = "./"
output_file_log = open("output_log.txt", "w")

#from pysr import PySRRegressor
#import sympy as sp
from sklearn.linear_model import Ridge 

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

time_chaos_x_SR_list=[]
time_chaos_x_parametric_list=[]
time_chaos_x_NN_list =[]
time_chaos_x_NN_nosym_list =[]
time_chaos_x_NN_SR_list =[]
time_chaos_x_NN_SR_nosym_list =[]
time_chaos_x_Sindy_list =[]
time_chaos_x_LS_list =[]
time_chaos_x_Sindy_ku0_list=[]

rmse_x_SR_list = []
rmse_x_dot_SR_list = []
rmse_x_parametric_list = []
rmse_x_dot_parametric_list = []
rmse_x_NN_list = []
rmse_x_dot_NN_list = []
rmse_x_NN_nosym_list = []
rmse_x_dot_NN_nosym_list = []
rmse_x_NN_nosym_SR_list = []
rmse_x_dot_NN_nosym_SR_list = []
rmse_x_NN_SR_list = []
rmse_x_dot_NN_SR_list = []
rmse_x_Sindy_list = []
rmse_x_dot_Sindy_list = []
rmse_x_LS_list = []
rmse_x_dot_LS_list = []
rmse_x_Sindy_k0_list = []
rmse_x_dot_Sindy_k0_list = []    
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
#Aext=0.5
#alpha=-1.0
#beta=1.0
#delta=0.3
#Omega=1.2
#x0=0.5
#v0=-0.5
#y0 = [x0, v0]  # [x(0), x'(0)]

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
neurons=100
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
SNR_dB_list = list(np.linspace(40, -20, 61 ))  # ∞, 20, 17.5, ..., -5

#SNR_dB_list = [np.inf] + list(np.linspace(40, 5, 36 ))  # ∞, 20, 17.5, ..., -5
#SNR_dB_list = list(np.linspace(-18, -20, 3 ))  # ∞, 20, 17.5, ..., -5

#SNR_dB_list = list(np.linspace(5, -20, 26 ))  # ∞, 20, 17.5, ..., -5

#SNR_dB_list = [20.0]

#repeat 3 times each value in the list
SNR_dB_list = np.repeat(SNR_dB_list, 10)


#SNR_dB_list = np.repeat(SNR_dB_list,B_list = list(np.linspace(5, -5, 3))  # ∞, 20, 17.5, ..., -5


for SNR_dB in SNR_dB_list:
    Aext=0.5
    alpha=-1.0
    beta=1.0
    delta=0.3
    Omega=1.2
    x0=0.5
    v0=-0.5
    y0 = [x0, v0]  # [x(0), x'(0)]
    plt.close()
    
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
#    plt.show()    
    

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
#        plt.show()
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
#    plt.show()   
    # plot 
    plt.figure()
    plt.title(r"MSE of F$_{ext}$ with and without noise")
    plt.plot(time_data, (F_ext(time_data) - F_ext_noise_data )**2) 
    plt.xlabel("t")
    plt.ylabel(r"Squared Error (F$_{ext}^{noiseless}$ - F$_{ext}^{noise}$)$^2$")
    plt.grid(True, alpha=0.3)
#    plt.show()
    
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
#    plt.show()
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
#    plt.show()



    ####################################################
    ############# Parametric-CC (Ridge) ################
    # Right-hand side
    rhs = F_ext_noise_data - x_ddot_data
    
    # Design matrix: [x_dot, x, x^3]
    A = np.vstack([
        x_dot_data,
        x_data,
        x_data**3
    ]).T  # shape: (N, 3)
    
    # --- NEW: Solve using Ridge Regression ---
    print("Training Parametric-CC model (Ridge)...")
    start = time.time()
    
    # ridge_alpha controls the penalty strength. 
    # Try 0.01, 0.1, or 1.0 depending on noise level.
    ridge_alpha = 1.0  
    
    # fit_intercept=False because physics models usually don't have a random bias term
    ridge_model = Ridge(alpha=ridge_alpha, fit_intercept=False)
    ridge_model.fit(A, rhs)
    
    # Extract parameters (Ridge stores them in .coef_)
    params = ridge_model.coef_
    delta_ident_param_ridge, alpha_ident_param_ridge, beta_ident_param_ridge = params
    
    end = time.time()
    elapsed = end - start
    print(f"Training finished in {elapsed:.3f} seconds")
    
    # --- Results Output ---
    print("Theoretical params:")
    print(f"delta = {delta:.6e}, alpha = {alpha:.6e}, beta = {beta:.6e}")
    print(" ")
    print(f"Identified params (Ridge alpha={ridge_alpha}):")
    print(f"delta = {delta_ident_param_ridge:.6e}, alpha = {alpha_ident_param_ridge:.6e}, beta = {beta_ident_param_ridge:.6e}")
    
    # defining the Parametric-CC functions for forward simulations
    def ode_param_ridge(t, state):
        x, xdot = state
        xddot = (F_ext(t) - delta_ident_param_ridge*xdot - alpha_ident_param_ridge*x - beta_ident_param_ridge*x**3) # / m
        return [xdot, xddot]
    def f1_param_ridge(x_dot):
        return delta_ident_param_ridge * x_dot
    def f2_param_ridge(x):
        return alpha_ident_param_ridge * x + beta_ident_param_ridge * x**3
    
    # --- Plotting (Unchanged) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
    # Note: Ensure xdot_vals, F1, x_vals, F2 are defined in your broader scope
    ax1.plot(xdot_vals, f1_param_ridge(xdot_vals), label="Identified", linewidth=2)
    ax1.plot(xdot_vals, F1(xdot_vals), '--',color='black', label="Theor.", linewidth=2)
    ax1.set_title("Obtained CCs from Parametric-CC Ridge method")
    ax1.set_xlabel(r"$\dot{x}$")
    ax1.set_ylabel(r"f$_1(\dot{x})$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(x_vals, f2_param_ridge(x_vals), label="Identified", linewidth=2)
    ax2.plot(x_vals, F2(x_vals), '--',color='black', label="Theor.", linewidth=3)
    ax2.set_ylabel(r"f$_2$(x)")
    ax2.set_xlabel("x")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
#    plt.show()
    
    # --- Simulation (Unchanged) ---
    # Note: Ensure t_span, y0, t_simul, time_data are defined
    sol = solve_ivp(ode_param_ridge, t_span, y0, t_eval=t_simul, method='LSODA')
    x_sim = sol.y[0]
    xdot_sim = sol.y[1]
    
    plt.figure(figsize=(8,5))
    plt.plot(time_data, x_sim, label="Parametric-CC Ridge", linewidth=2)
    plt.plot(time_data, x_data, "--", color='black', label="Training data", linewidth=2)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.title(f"Simulation with Parametric-CC (Ridge alpha={ridge_alpha})")
#    plt.show()



    ####################################################
    ############# Parametric-CC (STLSQ) ################
    
    # 1. Setup Data
    rhs = F_ext_noise_data - x_ddot_data
    A = np.vstack([x_dot_data, x_data, x_data**3]).T 
    n_features = A.shape[1]
    
    # 2. STLSQ Algorithm (Sequential Thresholded Least Squares)
    print("Training Parametric-CC model (STLSQ)...")
    start = time.time()
    
    # Hyperparameter: The cutoff for "noise"
    # Terms smaller than this will be deleted.
    # If your expected params are ~0.3, try 0.05 or 0.1
    threshold = 0.05  
    
    # Initial OLS fit
    params, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    
    # Iteration loop (usually converges in 1-2 steps for simple physics)
    for k in range(10):
        # Identify small coefficients
        small_inds = np.abs(params) < threshold
        
        # If nothing is small, stop (we are done)
        if not np.any(small_inds):
            break
            
        # Set small coefficients to zero (Hard Thresholding)
        params[small_inds] = 0
        
        # Identify large coefficients (the ones we keep)
        big_inds = ~small_inds
        
        # Re-solve OLS *only* on the big coefficients
        # This prevents the "shrinkage" error you saw with Ridge
        A_subset = A[:, big_inds]
        params_subset, _, _, _ = np.linalg.lstsq(A_subset, rhs, rcond=None)
        
        # Update the parameter vector
        params[big_inds] = params_subset
        
        # Check if the "zero structure" has stabilized
        # (If the same terms are zero as last time, we are done)
        # For this simple loop, we just run it; in complex cases, add a check.
    
    delta_ident_param_STLSQ, alpha_ident_param_STLSQ, beta_ident_param_STLSQ = params
    end = time.time()
    
    print(f"Training finished in {end - start:.3f} seconds")
    print(f"Identified params (STLSQ threshold={threshold}):")
    print(f"delta = {delta_ident_param:.6e}")
    print(f"alpha = {alpha_ident_param:.6e}")
    print(f"beta  = {beta_ident_param:.6e}")
    
    # --- Forward Simulation (Same as before) ---
    def ode_param(t, state):
        x, xdot = state
        # Note: We use the identified params which might be 0.0 if filtered out
        xddot = (F_ext(t) - delta_ident_param_STLSQ*xdot - alpha_ident_param_STLSQ*x - beta_ident_param_STLSQ*x**3) 
        return [xdot, xddot]
    def f1_param_STLSQ(x_dot):
        return delta_ident_param_STLSQ * x_dot
    def f2_param_STLSQ(x):
        return alpha_ident_param_STLSQ * x + beta_ident_param_STLSQ * x**3
        
    sol = solve_ivp(ode_param, t_span, y0, t_eval=t_simul, method='LSODA')
    x_sim = sol.y[0]
    
    plt.figure(figsize=(8,5))
    plt.plot(time_data, x_sim, label="STLSQ Fit", linewidth=2)
    plt.plot(time_data, x_data, "--", color='black', label="Training Data", linewidth=2)
    plt.legend()
    plt.title(f"Simulation with STLSQ (Threshold={threshold})")
#    plt.show()


    ####################################################
    ############# Robust Parametric-CC #################
    from sklearn.linear_model import HuberRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline 
    # 1. Setup Data
    rhs = F_ext_noise_data - x_ddot_data
    A = np.vstack([x_dot_data, x_data, x_data**3]).T 
    
    # 2. Train with Scaling + Robust Regression
    print("Training Parametric-CC model (Robust Huber)...")
    start = time.time()
    
    # We use a Pipeline: 
    # Step 1: Scale data (Crucial for x vs x^3)
    # Step 2: Huber Regressor (Robust to noise, better than OLS/Ridge)
    # epsilon=1.35 is standard; smaller values (1.1) make it more resistant to outliers
    model = make_pipeline(
        StandardScaler(with_mean=False), 
        HuberRegressor(fit_intercept=False, epsilon=1.35, max_iter=200)
    )
    
    model.fit(A, rhs)
    
    # Extract parameters
    # Note: The pipeline handles the un-scaling automatically for predictions,
    # but to get the raw physics coefficients, we access the named steps.
    # However, sklearn's pipeline makes extracting 'raw' coefficients tricky.
    # The easiest way to get the coefficients for your print statement 
    # is to let the model predict on unit vectors or just trust the pipeline for simulation.
    
    # Let's extract them manually for your print statement:
    regressor = model.named_steps['huberregressor']
    scaler = model.named_steps['standardscaler']
    raw_coefs = regressor.coef_ / scaler.scale_ # Un-scale coefficients
    delta_ident_param_huber, alpha_ident_param_huber, beta_ident_param_huber = raw_coefs
    
    end = time.time()
    
    print(f"Training finished in {end - start:.3f} seconds")
    print(f"Identified params (Robust):")
    print(f"delta = {delta_ident_param_huber:.6e} (Target: {delta})")
    print(f"alpha = {alpha_ident_param_huber:.6e} (Target: {alpha})")
    print(f"beta  = {beta_ident_param_huber:.6e} (Target: {beta})")
    
    # --- Forward Simulation ---
    def ode_param(t, state):
        x, xdot = state
        xddot = (F_ext(t) - delta_ident_param_huber*xdot - alpha_ident_param_huber*x - beta_ident_param_huber*x**3) 
        return [xdot, xddot]
    def f1_param_huber(x_dot):
        return delta_ident_param_huber * x_dot
    def f2_param_huber(x):
        return alpha_ident_param_huber * x + beta_ident_param_huber * x**3
    
    
    
    sol = solve_ivp(ode_param, t_span, y0, t_eval=t_simul, method='LSODA')
    x_sim = sol.y[0]
    
    plt.figure(figsize=(8,5))
    plt.plot(time_data, x_sim, label="Robust Fit", linewidth=2)
    plt.plot(time_data, x_data, "--", color='black', label="Training Data", alpha=0.7)
    plt.legend()
    plt.title("Simulation with Robust Huber Regression")
#    plt.show()



    ####################################################
    #############   Affine Scaled OLS   ################
    
    # 1. Define the Transformation Parameters
    # Maps x -> [-1, 1]
    A0 = (np.max(x_data) + np.min(x_data)) / 2
    A1 = (np.max(x_data) - np.min(x_data)) / 2
    
    # Maps x_dot -> [-1, 1]
    B0 = (np.max(x_dot_data) + np.min(x_dot_data)) / 2
    B1 = (np.max(x_dot_data) - np.min(x_dot_data)) / 2
    
    # 2. Transform the Data
    x_tilde = (x_data - A0) / A1
    v_tilde = (x_dot_data - B0) / B1
    
    # 3. Create Design Matrix
    # MUST include x^2 and Intercept because of the shift!
    # Model: RHS = c0 + c1*v_tilde + c2*x_tilde + c3*x_tilde^2 + c4*x_tilde^3
    A_scaled = np.vstack([
        np.ones_like(x_tilde), # Intercept (c0)
        v_tilde,               # Velocity term (c1)
        x_tilde,               # Linear term (c2)
        x_tilde**2,            # Quadratic ghost term (c3)
        x_tilde**3             # Cubic term (c4)
    ]).T 
    
    rhs = F_ext_noise_data - x_ddot_data
    
    # 4. Solve OLS in Scaled Space
    print("Training Affine Scaled OLS model...")
    start = time.time()
    params_scaled, _, _, _ = np.linalg.lstsq(A_scaled, rhs, rcond=None)
    c0, c1, c2, c3, c4 = params_scaled
    end = time.time()
    
    # 5. Recover Physical Parameters
    # We derive these by comparing coefficients of the expanded polynomial
    # beta is determined solely by the cubic term
    beta_ident_param_affineOLS = c4 / (A1**3)
    
    # alpha is coupled with beta and the shift A0
    # From expansion: Coeff of x_tilde is (alpha*A1 + 3*beta*A1*A0^2)
    # So: c2 = alpha*A1 + 3*beta*A1*A0^2
    alpha_ident_param_affineOLS = (c2 - 3 * beta_ident_param_affineOLS * A1 * (A0**2)) / A1
    
    # delta is simple scaling
    delta_ident_param_affineOLS = c1 / B1
    
    print(f"Training finished in {end - start:.3f} seconds")
    print("-" * 30)
    print(f"Theoretical: delta={delta:.4f}, alpha={alpha:.4f}, beta={beta:.4f}")
    print(f"Identified:  delta={delta_ident_param_affineOLS:.4f}, alpha={alpha_ident_param_affineOLS:.4f}, beta={beta_ident_param_affineOLS:.4f}")
    
    # 6. Sanity Check: The "Ghost" terms should match the theory
    # The quadratic coefficient c3 theoretically equals: 3 * beta * A1^2 * A0
    theoretical_c3 = 3 * beta_ident_param_affineOLS * (A1**2) * A0
    print(f"Ghost x^2 check: Found {c3:.4f}, Expected {theoretical_c3:.4f}")
    
    # --- Forward Simulation ---
    def ode_param(t, state):
        x, xdot = state
        xddot = (F_ext(t) - delta_ident_param_affineOLS*xdot - alpha_ident_param_affineOLS*x - beta_ident_param_affineOLS*x**3) 
        return [xdot, xddot]
    def f1_param_affineOLS(x_dot):
        return delta_ident_param_affineOLS * x_dot
    def f2_param_affineOLS(x):
        return alpha_ident_param_affineOLS * x + beta_ident_param_affineOLS * x**3
    





    ##########################################
    # Evaluation of the f1 and f2 functions
    ##########################################
    predicted_F1_theor = F1(xdot_vals).flatten()
    predicted_F1_param = f1_param(xdot_vals).flatten() # parametric-CC
    predicted_F1_param_ridge = f1_param_ridge(xdot_vals).flatten() # parametric-CC
    predicted_F1_param_STLSQ = f1_param_STLSQ(xdot_vals).flatten() # parametric-CC
    predicted_F1_param_huber = f1_param_huber(xdot_vals).flatten() # parametric-CC
    predicted_F1_param_affineOLS = f1_param_affineOLS(xdot_vals).flatten() # parametric-CC

    predicted_F2_theor = F2(x_vals).flatten()
    predicted_F2_param = f2_param(x_vals).flatten() # parametric-CC
    predicted_F2_param_ridge = f2_param_ridge(x_vals).flatten() # parametric-CC
    predicted_F2_param_STLSQ = f2_param_STLSQ(x_vals).flatten() # parametric-CC
    predicted_F2_param_huber = f2_param_huber(x_vals).flatten() # parametric-CC
    predicted_F2_param_affineOLS = f2_param_affineOLS(x_vals).flatten() # parametric-CC

    

    # 1. Define RMSE function
    def rmse(y_true, y_pred):
        # Flattening ensures shapes like (N,1) and (N,) don't cause broadcasting errors
        return np.sqrt(np.mean((y_true.flatten() - y_pred.flatten()) ** 2))

    # 2. Construct the Dictionary with new variables
    rmse_values = {
        "noise_perc_th": noise_percentage_th,
        "noise_perc": noise_percentage,
        "SNR_dB": SNR_dB,
        
        # Neural Network Variants
        "Param_f1": rmse(predicted_F1_theor, predicted_F1_param),
        "Param_f2": rmse(predicted_F2_theor, predicted_F2_param),
        "Param_f1_ridge": rmse(predicted_F1_theor, predicted_F1_param_ridge),
        "Param_f2_ridge": rmse(predicted_F2_theor, predicted_F2_param_ridge),
        "Param_f1_STLSQ": rmse(predicted_F1_theor, predicted_F1_param_STLSQ),
        "Param_f2_STLSQ": rmse(predicted_F2_theor, predicted_F2_param_STLSQ),
        "Param_f1_huber": rmse(predicted_F1_theor, predicted_F1_param_huber),
        "Param_f2_huber": rmse(predicted_F2_theor, predicted_F2_param_huber),
        "Param_f1_affineOLS": rmse(predicted_F1_theor, predicted_F1_param_affineOLS),
        "Param_f2_affineOLS": rmse(predicted_F2_theor, predicted_F2_param_affineOLS),      

    }

    # 3. Print specific console output
    print("-" * 40)
    print(f"param_f1       = {rmse_values['Param_f1']:.4e}")
    print(f"param_f2       = {rmse_values['Param_f2']:.4e}")


    # 4. Write to file
    fname = "rmse_results_for_f1_and_f2.txt"
    header = "# " + " ".join(rmse_values.keys())

    # Formats values to scientific notation. 
    # np.ravel(v)[0] ensures we extract a scalar float even if v is a numpy array.
    line = " ".join(f"{float(np.ravel(v)[0]):.4e}" for v in rmse_values.values())

    mode = "a" if os.path.exists(fname) else "w"
    with open(fname, mode) as f:
        if mode == "w":
            f.write(header + "\n")
            print("Created new file with header:")
            print(header)
        f.write(line + "\n")

    print(f"Appended results to {fname}")
    print("Line written:", line)
    
    




    ################################################################################
    ################################################################################
    ################################   VALIDATION   ################################
    ################################################################################
    ################################################################################



    print('Validating the system')
    print('Integration of model EDOs')

    n_trials = 10  # number of random initial conditions
    #rmse_x_NN_list = []
    #rmse_x_dot_NN_list = []
    #rmse_x_Sindy_list = []
    #rmse_x_dot_Sindy_list = []
    #rmse_x_LS_list = []
    #rmse_x_dot_LS_list = []


#    for i in range(n_trials):
# Validation Loop Variables
    n_trials = 10        # Target number of valid simulations
    valid_trials = 0     # Counter for valid simulations
    attempts = 0         # Counter for total attempts (to prevent infinite loops if needed)
    max_attempts = 1000   # Safety break

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
        Aext = np.round(np.random.uniform(0.1, 0.5),3)
        Omega = np.round(np.random.uniform(1.1, 1.3),3)
        
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
        is_out_x = np.any(x_simulated_th < np.min(x_data)) or np.any(x_simulated_th > np.max(x_data))
        is_out_v = np.any(x_dot_simulated_th < np.min(x_dot_data)) or np.any(x_dot_simulated_th > np.max(x_dot_data))
        
        if is_out_x or is_out_v:
            # If outside, skip this trial (do not increment valid_trials)
            print(f"Attempt {attempts}: Out of bounds. Retrying...") 
            continue 
        
        # If we reach here, the curve is strictly inside the box
        valid_trials += 1
        print(f"Trial {valid_trials}/{n_trials} found (Attempt {attempts})")
        




        ################ parametric-CC ###### validation of the model
        sol_parametric = solve_ivp(ode_param, t_span_val, y0_val, t_eval=t_val,method='LSODA')
                        #rtol=1e-9, atol=1e-12, max_step=(t_eval[1] - t_eval[0]))   
        t_parametric= sol_parametric.t
        x_simulated_parametric = sol_parametric.y[0]
        x_dot_simulated_parametric = sol_parametric.y[1]
        plt.figure(figsize=(8,5))
        plt.plot(t_parametric, x_simulated_parametric, "--", label="Parametric-CC", linewidth=2)
        plt.plot(t_simulated_th, x_simulated_th, label="Theor.", linewidth=2)
        plt.xlabel("t")
        plt.ylabel("x(t)")
        plt.legend()
        plt.title("Parametric model simulation")
#        plt.show()

        
        def custom_ticks(ax, major_x_interval, major_y_interval, minor_x_interval, minor_y_interval):
            # Set major ticks
            ax.xaxis.set_major_locator(ticker.MultipleLocator(major_x_interval))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(major_y_interval))
            # Set minor ticks
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_x_interval))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(minor_y_interval))
            # Customize tick appearance
        #    ax.tick_params(axis='both', direction='in', which='major', length=8, width=1.5, labelsize=24)
        #    ax.tick_params(axis='both', direction='in', which='minor', length=5, width=1)
            ax.tick_params(axis='x', direction='in', which='major', length=8, width=1.5, labelsize=24, top=True, bottom=True)
            ax.tick_params(axis='y', direction='in', which='major', length=8, width=1.5, labelsize=24, left=True, right=True)
            ax.tick_params(axis='x', direction='in', which='minor', length=5, width=1, top=True,bottom=True)
            ax.tick_params(axis='y', direction='in', which='minor', length=5, width=1, left=True, right=True)




        # Calculation of absolute value separation
        threshold_chaos = 0.2 # 4

        time_chaos_x_parametric=t_val[-1]
        for i in range(len(x_simulated_parametric)):
            diff = abs(x_simulated_parametric[i] - x_simulated_th[i])
            if diff > threshold_chaos:
#                time_chaos_x_parametric_list.append(t_val[i])
                time_chaos_x_parametric=t_val[i]
                break
        time_chaos_x_parametric_list.append(time_chaos_x_parametric)
        #time_chaos_x_NN_list.append(t_val[-1])


        time_matrix_append = np.column_stack([
            noise_percentage_th,
            noise_percentage,
            SNR_dB,
            time_chaos_x_parametric
        ])
        folder_path = output_path
        os.makedirs(folder_path, exist_ok=True)
        file_name = "times_noise_chaos_duffing_SR_and_param.txt"
        file_path = os.path.join(folder_path, file_name)
        # Save with header and space as delimiter
        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a') as f:
            np.savetxt(f, time_matrix_append,
                       #header="nois_th nois nois_db  NN  Sindy  LS NN_ret NN_SR Sindy_ku0 SR Param" if not file_exists else '',
                       header="nois_th nois nois_db  Parametric" if not file_exists else '',
                       #header="# noise_th noise noise_db t_SR t_parametric" if not file_exists else '',
                       fmt="%.2f", delimiter=" ", comments='')




#test_predictions = model(test_inputs).cpu().numpy()  # Move predictions to CPU for plotting



