 
##!pip install pysindy
# Van der Pol RN with Fext
# Importar bibliotecas necesarias
import matplotlib
matplotlib.use('Agg')


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
from pysindy.feature_library import PolynomialLibrary
from pysindy.feature_library import CustomLibrary
from pysindy.feature_library import ParameterizedLibrary
from pysindy.feature_library import IdentityLibrary
from pysindy.optimizers import ConstrainedSR3
from pysindy import AxesArray
from pysindy.optimizers import STLSQ
from pysindy import SINDy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy.polynomial.legendre import legvander
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
SR_crossed_terms=False



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
rmse_x_Sindy_ku0_list = []
rmse_x_dot_Sindy_ku0_list = []
rmse_x_LS_list = []
rmse_x_dot_LS_list = []
rmse_x_Sindy_k0_list = []
rmse_x_dot_Sindy_k0_list = []    
print("EDO: x'' + f1(x') + f2 (x) = F_ext(t)")
print("S1: stick-slip")
print("f1(x')= [c*x'+Ff(x')]/m")
print("f2(x)=[k x]/m")
print("F_ext(t)=F_ext_true(t)/m")

#parameters stick slip
#m=1.0 # kg
#cval=0.1 # Ns/m (viscous damping coefficient)
#kval=1.0 # N/m (stiffness)
#Aext=2 # N (forcing amplitude)
#Omega=0.3 # 0.3 and 0.15 rad/s (forcing frequency)
#x0=-0.076 #0.1 # m (initial displacement)
#v0=0.146 #0.1 # m/s (initial velocity)
#mu_N = 0.801 # 0.5 #0.5
#y0 = [x0, v0]  # [x(0), x'(0)]

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
Tval=Tsimul # for forward simulations with trained models
Nval=Nsimul
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
epochs_max = 10000
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
#SNR_dB_list = list(np.linspace(26, -20, 47 ))  # ∞, 20, 17.5, ..., -5

SNR_dB_list = list(np.linspace(5, -20,  26))  # ∞, 20, 17.5, ..., -5

#SNR_dB_list = [np.inf] + list(np.linspace(40, 5, 36 ))  # ∞, 20, 17.5, ..., -5
SNR_dB_list = list(np.linspace(-18, -20, 3 ))  # ∞, 20, 17.5, ..., -5

#SNR_dB_list = list(np.linspace(5, -20, 26 ))  # ∞, 20, 17.5, ..., -5

#SNR_dB_list = [20.0]

#repeat 3 times each value in the list
SNR_dB_list = np.repeat(SNR_dB_list, 10)


#SNR_dB_list = np.repeat(SNR_dB_list,B_list = list(np.linspace(5, -5, 3))  # ∞, 20, 17.5, ..., -5


for SNR_dB in SNR_dB_list:
    m=1.0 # kg
    cval=0.386 #0.1 # Ns/m (viscous damping coefficient)
    kval=1.274 # 1.0 # N/m (stiffness)
    Aext=2 # N (forcing amplitude)
    Omega=0.3 # 0.3 and 0.15 rad/s (forcing frequency)
    x0=-0.076 #0.1 # m (initial displacement)
    v0=0.146 #0.1 # m/s (initial velocity)
    mu_N = 0.801 # 0.5 #0.5
    y0 = [x0, v0]  # [x(0), x'(0)]

    print(f"SNR_dB={SNR_dB}")    
    #stick-slip
    print(f"Aext={Aext}, k={kval}, c={cval}")
    print(f"Omega={Omega}, mu_N={mu_N}, x₀={x0}, v₀={v0}")
    
    #Definition of the theoretical functions

    #stick-slip
    def smooth_sign(x, alpha=500):
        return np.tanh(alpha * x)
    def Ff_coul(x_dot):
     #   return mu_N * np.sign(x_dot)
        return mu_N * smooth_sign(x_dot)
    def F1(x_dot):
        return cval* x_dot + Ff_coul(x_dot) #delta * x_dot #+ Ff_coul(x_dot) #cval* x_dot + Ff_coul(x_dot) # + 0.0005 * x_dot**2 #+ Ff_coul(x_dot) #r(x_dot) Ff_coul Ff_dr
    def F2(x):
        return kval*x # alpha*x+beta*x**3 #kval*x
    def F_ext(t):
        return Aext*np.cos(Omega*t)
    def eq_2nd_ord_veloc(t,y):
        x, x_dot = y  # y=[x, x']
        x_ddot = (F_ext(t) - F1(x_dot) - F2(x))/m
        return [x_dot, x_ddot]

    # ODE: x'' + F1(x_dot) + F2(x) = F_ext(t) 
    # ODE: x'' + c x_dot + mu N sign(x_dot) + kval x = F_ext(t) 
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
    #plt.tight_layout()
    #plt.show()    
    

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
        #plt.tight_layout()
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
    #plt.tight_layout()
    plt.show()   
    # plot 
    plt.figure()
    plt.title(r"MSE of F$_{ext}$ with and without noise")
    plt.plot(time_data, (F_ext(time_data) - F_ext_noise_data )**2) 
    plt.xlabel("t")
    plt.ylabel(r"Squared Error (F$_{ext}^{noiseless}$ - F$_{ext}^{noise}$)$^2$")
    plt.grid(True, alpha=0.3)
    #plt.show()
    
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
        np.tanh(500 * x_dot_data)  # smooth sign for Coulomb
        #x_data**3
    ]).T  # shape: (N, 3)
    # Solve least squares: A x [delta, alpha, beta] = rhs
    start = time.time()
    params, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    c_ident_param, k_ident_param, mu_ident_param = params
    print("Parametric model")
    end = time.time()  
    elapsed = end - start
    print(f"Training finished in {elapsed:.3f} seconds")
    print("Theoretical params:")
    print(f"c = {cval:.6e}, k = {kval:.6e}, mu*N = {mu_N:.6e}")
    print("Identified params from Parametric model:")
    print(f"c = {c_ident_param:.6e}, k = {k_ident_param:.6e}, mu*N = {mu_ident_param:.6e}")
    

    #stick-slip
    def ode_param(t, state):
        x, xdot = state
        xddot = (F_ext(t) - c_ident_param*xdot - k_ident_param*x - mu_ident_param*np.tanh(500 * xdot) ) / m
        return [xdot, xddot]
    def f1_param(x_dot):
        return c_ident_param * x_dot + mu_ident_param * np.tanh(500 * x_dot)
    def f2_param(x):
        return k_ident_param * x  
        
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
    ax1.plot(xdot_vals, f1_param(xdot_vals), label="Identified", linewidth=2)
    ax1.plot(xdot_vals, F1(xdot_vals), '--',color='black', label="Theor.", linewidth=2)
    ax1.set_title("Obtained CCs from Parametric method")
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
    #plt.tight_layout()
    plt.show()
    # Integrate forward the obtained Parametric-CC model for verification
    # (using the same ICs and driven force as the training data)
    sol = solve_ivp(ode_param, t_span, y0, t_eval=t_simul,method='LSODA') 
    x_sim = sol.y[0]
    xdot_sim = sol.y[1]
    plt.figure(figsize=(8,5))
    plt.plot(time_data[0:len(x_sim)], x_sim, label="Parametric", linewidth=2)
    plt.plot(time_data, x_data, "--", color='black', label="Training data", linewidth=2)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.title("Simulation with Parametric-CC model")
    #plt.show()



    ####################################################
    ############# Poly-CC  #################   least squares
    # This method uses the Polynomial expansions for f1 and f2 
    # but normalizing the data ranges to [-1,1] interval,
    # according to the approach discussed in Refs.
    # Gonzalez, F.J. Nonlinear Dyn 112, 16167–16197 (2024).  
    # Gonzalez, F.J., Lara, L.P. Nonlinear Dyn 113, 33063–33086 (2025).  
    N_order=10
    min_x = np.min(np.abs(x_data))
    max_x = np.max(np.abs(x_data))
    min_xd = np.min(np.abs(x_dot_data))
    max_xd = np.max(np.abs(x_dot_data))
    A0_x=(max_x+min_x)/2.0
    A1_x=(max_x-min_x)/2.0
    A0_xd=(max_xd+min_xd)/2.0
    A1_xd=(max_xd-min_xd)/2.0
    X_poly = np.vstack([((x_data-A0_x)/A1_x)**i for i in range(N_order + 1)]).T
    X_dot_poly = np.vstack([((x_dot_data-A0_xd)/A1_xd)**i for i in range(N_order + 1)]).T  # [N x (N_order+1)]
    # Combine both: A @ coeffs = b
    Amat = np.hstack([-X_dot_poly, -X_poly])  # Minus signs because x'' = F_ext - f1 - f2
    bmat = x_ddot_data -  F_ext_noise_data #F_ext(t_simul)       # Leftover = f1(x') + f2(x)
    # ---- Least squares fit ---- #
    start=time.time()
    coeffs, _, _, _ = np.linalg.lstsq(Amat, bmat, rcond=None)
    end = time.time()  
    elapsed = end - start
    print('End Training Poly-CC')
    print(f"Training finished in {elapsed:.3f} seconds")
    # Separate coefficients
    c_f1 = coeffs[:N_order+1]
    c_f2 = coeffs[N_order+1:]
    print(f"Coefficients of f1(x') (order {N_order}): {c_f1}")
    print(f"Coefficients of f2(x)  (order {N_order}): {c_f2}")
    # Print coefficients for f1_fit and f2_fit
    print("Coefficients for f1_fit($\\dot{x}$):")
    for i, c in enumerate(c_f1):
        print(f"  c[{i}] = {c:.6e}")
    print("\nCoefficients for f2_fit(x):")
    for i, c in enumerate(c_f2):
        print(f"  c[{i}] = {c:.6e}")
    # Transform scaled polynomial back to real x space
    def transform_coefficients(c_scaled, A0, A1):
        N = len(c_scaled)
        c_real = np.zeros(N)
        for j in range(N):
            for k in range(j, N):
                c_real[j] += c_scaled[k] * comb(k, j) * ((-A0) ** (k - j)) / (A1 ** k)
        return c_real
    c_f1_transformed = transform_coefficients(c_f1, A0_xd, A1_xd)
    c_f2_transformed = transform_coefficients(c_f2, A0_x, A1_x)
    print("\nTransformed (real x) coefficients for f1:")
    for i, coef in enumerate(c_f1_transformed):
        print(f"  coef[{i}] = {coef:.6e}")
    print("\nTransformed (real x) coefficients for f2:")
    for i, coef in enumerate(c_f2_transformed):
        print(f"  coef[{i}] = {coef:.6e}")

    def f1_fit(x_dot):
        return sum(c * ((x_dot-A0_xd)/A1_xd)**i for i, c in enumerate(c_f1))
    def f2_fit(x):
        return sum(c * ((x-A0_x)/A1_x)**i for i, c in enumerate(c_f2))
    def fitted_model_LS(t, y):
        x, x_dot = y
        x_ddot = F_ext(t) - f1_fit(x_dot) - f2_fit(x)
        return [x_dot, x_ddot]

    shift_poly=f1_fit(0)
    f1_fit_shifted=f1_fit(xdot_vals)-shift_poly
    f2_fit_shifted=f2_fit(x_vals)+shift_poly
    # plotting obtained CCs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
    ax1.plot(xdot_vals, f1_fit_shifted, label="Identified", linewidth=2)
    ax1.plot(xdot_vals, F1(xdot_vals), '--',color='black', label="Theor.", linewidth=2)
    ax1.set_title("Obtained CCs from Poly-CC method")
    ax1.set_xlabel(r"$\dot{x}$")
    ax1.set_ylabel(r"f$_1(\dot{x})$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(x_vals, f2_fit_shifted, label="Identified", linewidth=2)
    ax2.plot(x_vals, F2(x_vals), '--',color='black', label="Theor.", linewidth=3)
    ax2.set_ylabel(r"f$_2$(x)")
    ax2.set_xlabel("x")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    #plt.tight_layout()
    #plt.show()

#    try:
#        sol = solve_ivp(fitted_model_LS, t_span, y0, t_eval=t_simul,method='LSODA', rtol=1e-9, atol=1e-12)
#        #x_simulated_LS = sol_fit.y[0]       # Posición
#        #x_dot_simulated_LS = sol_fit.y[1]   # Velocidad
#        # check for NaNs or infs just in case
#        if np.any(np.isnan(x_simulated_LS)) or np.any(np.isinf(x_simulated_LS)):
#            print(f"Skipping trial {i+1}: LS simulation returned NaNs or infs.")
#            # continue your RMSE calculations and plots here...
#    except Exception as e:
#        print(f"Skipping trial {i+1}: Exception during LS simulation -> {e}")              
#    x_sim = sol.y[0]
#    xdot_sim = sol.y[1]
#    plt.figure(figsize=(8,5))
#    plt.plot(time_data[0:len(x_sim)], x_sim, label="Poly-CC", linewidth=2)
#    plt.plot(time_data, x_data,"--",color='black', label="Training data", linewidth=2)
#    plt.xlabel("t")
#    plt.ylabel("x(t)")
#    plt.legend()
#    plt.title("Simulation with Poly-CC model")
#    plt.show()
    
    
    ##############################################
    #############  SINDY variants ################   
    ##############################################

    ##############################################
    #############  SINDY with u0 #################   
    #data = np.stack((x_data, x_dot_data), axis=-1)
    # define dataset
    data = np.stack((x_data, x_dot_data,time_data), axis=-1)
    #X_dot_sindy=np.array([x_dot_data,x_ddot_data for i, t in enumerate(t_simul)])
    X_sindy = np.stack((x_data,x_dot_data),axis=1) #time_data # sol.y.T  # Transpose to get shape (N, 2)
    X_dot_sindy = np.stack((x_dot_data, x_ddot_data), axis=1)
    u_sindy = F_ext_noise_data # np.stack((F_ext_noise_data),axis=1)
    u_sindy_resh = F_ext_noise_data.reshape(-1,1) #flatten()
    X_combined = np.hstack((X_sindy, u_sindy_resh))
    #polynomials and not crossed terms
    # fourier_library = ps.FourierLibrary(n_frequencies=1)
    #combined_library = ps.GeneralizedLibrary([poly_library, fourier_library])
    print("Sindy polynomial basis with external force u0")
    poly_library = ps.PolynomialLibrary(degree=10, include_interaction=False,order='c',include_bias=False) #, include_input=False) #, include_input=False)#,interaction_only=True
    optimizer_sindy = ps.STLSQ(threshold=1e-4, max_iter=10000)
    model = ps.SINDy(feature_library=poly_library,optimizer=optimizer_sindy)
    start = time.time()
    model.fit(X_sindy, t=time_data, x_dot=X_dot_sindy, u=u_sindy)
    end = time.time()  
    elapsed = end - start
    print('End Training SINDy with u0')
    print(f"Training finished in {elapsed:.3f} seconds")
    #model = ps.SINDy(feature_library=combined_library)
    #model.fit(X_sindy, t=sol.t, u=u_sindy, x_dot=X_dot_sindy)  # ¡Aquí pasamos u!
    print("\nIdentified SINDy model (exponential format, no labels):")
    model.print()
    print(" ")
    print("Sindy with term k*u0")
    poly_library = ps.PolynomialLibrary(degree=10, include_interaction=False,include_bias=False) #, include_input=False) #, include_input=False)#,interaction_only=True
    input_library = ps.IdentityLibrary()
    inputs_idx = np.array([[0, 1], [2, 2]], dtype=int)
    combined_library = ps.GeneralizedLibrary([poly_library, input_library],inputs_per_library=inputs_idx)
    optimizer = ps.STLSQ(threshold=1e-4, max_iter=10000)
    model_sindy_ku0 = ps.SINDy(feature_library=combined_library, optimizer=optimizer)
    start=time.time()
    model_sindy_ku0.fit(
        X_sindy, t=time_data, x_dot=X_dot_sindy,
        u=u_sindy
    )
    model_sindy_ku0.print()
    end = time.time()  
    elapsed = end - start
    print('End Training SINDy with k*u0')
    print(f"Training finished in {elapsed:.3f} seconds")

    ################ SINDY without restrictions ###### validation of the model
    print("Simulating Sindy without restrictions")
    x_simulated_Sindy_ku0 = []
    x_dot_simulated_Sindy_ku0 = []
    start = time.time()
    try:
        u_val_sindy = F_ext(time_data)
        x_val_sindy_ku0 = model_sindy_ku0.simulate(y0, t_simul, u=u_val_sindy)
        x_simulated_Sindy_ku0 = x_val_sindy_ku0[:, 0]
        x_dot_simulated_Sindy_ku0 = x_val_sindy_ku0[:, 1]
        # check for NaNs or infs just in case
        if np.any(np.isnan(x_val_sindy_ku0)) or np.any(np.isinf(x_val_sindy_ku0)):
            print(f"Skipping trial {SNR_dB}: SINDy simulation returned NaNs or infs.")
    except Exception as e:
        print(f"Skipping trial {SNR_dB}: Exception during SINDy simulation -> {e}")
    end = time.time()  
    elapsed = end - start
    print(f"Solve_ivp finished in {elapsed:.3f} seconds")
    plt.figure(figsize=(8,5))
    plt.plot(time_data[0:len(x_simulated_Sindy_ku0)], x_simulated_Sindy_ku0, label="SINDy", linewidth=2)
    plt.plot(time_data, x_data,"--",color='black', label="Training data", linewidth=2)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.title("Simulation with baseline SINDy model")
    plt.show()

    #######################################################################
    #############  SINDY with u0 + SR3constraint (+1.0*u0) #################   
    print(" ")
    print("Sindy restricted to +1.0*u0")
    F_p = poly_library.n_output_features_         # number of state‐only features
    F_u = input_library.n_output_features_        # should be 1 (just u₀)
    F   = F_p + F_u                                # total features per target
    T   = X_sindy.shape[1]                        # number of state equations, here 2
    C = np.zeros((1, F * T), dtype=float)
    d = np.array([1.0])  # RHS: force coefficient == 1
    idx = F + F_p
    C[0, idx] = 1.0
    # Instantiate optimizer with correct keywords
    optimizer = ConstrainedSR3(
        constraint_lhs=C,
        constraint_rhs=d,
        constraint_order="target",    # default, but explicit is clearer
        #reg_weight_lam=1e-3,
        #reg_weight_lam=1e-3,
        #relax_coeff_nu=1e-6,
        threshold=1e-5,
        max_iter=10000,
        equality_constraints=True     # enforce Cw = d strictly
    )
    # Fit exactly as before
    start= time.time()
    model = SINDy(feature_library=combined_library, optimizer=optimizer, feature_names=['x0', 'x1', 'u0'])
    model.fit(X_sindy, t=time_data, x_dot=X_dot_sindy, u=u_sindy)
    model.print()
    end = time.time()  
    elapsed = end - start
    print('End Training SINDy with 1*u0')
    print(f"Training finished in {elapsed:.3f} seconds")

    # === 2) Extract coefficients and feature names ===
    # model.coefficients() → shape (n_states, n_features)
    # model.get_feature_names() → list of length n_features
    # 1) Retrieve the feature‐name list
    features = model.get_feature_names()
    # 2) Retrieve the coefficient matrix
    #    shape = (n_states, n_features)
    coefs = np.array(model.coefficients())
    # Check shapes
    n_states, n_features = coefs.shape
    assert n_states >= 2, "Need at least two state equations"
    # === 3) Build f1(x0) and f2(x1) from eqn 1 (second row) ===
    row = coefs[1]  # coefficients for x1'
    f1_terms = []   # collect (power, coef) for x0
    f2_terms = []   # collect (power, coef) for x1
    for feat, c in zip(features, row):
        if feat.startswith("x0"):
            # feature names look like 'x0', 'x0^2', 'x0^3', etc.
            # extract power:
            if feat == "x0":
                p = 1
            else:
                p = int(feat.split("^")[1])
            f2_terms.append((p, c))
        elif feat.startswith("x1"):
            if feat == "x1":
                p = 1
            else:
                p = int(feat.split("^")[1])
            f1_terms.append((p, c))
        # ignore u0 (you’ve already fixed its coef = 1)
    print("\nModel coefficients with more precision:")
    print(model.coefficients())  # This should show the coefficients with more decimals
    # Extract coefficients and feature names
    coeffs = model.coefficients()
    features = model.get_feature_names()
    # Define a small threshold to exclude near-zero values
    threshold = 1e-20  # You can adjust this as needed
    print("\nIdentified SINDy model (nonzero terms only, exponential format):")
    for eq_idx in range(coeffs.shape[1]):
        terms = []
        for coef, name in zip(coeffs[:, eq_idx], features):
     #       terms.append(f"({coef:.12e}) * {name}")
            if np.abs(coef) > threshold:
                terms.append(f"({coef:.12e}) * {name}")
        eq_str = " + ".join(terms) if terms else ""
        print(eq_str)   
    # Define functions f1 and f2
    def f1_sindy(x):
        y = np.zeros_like(x)
        for p, c in f1_terms:
            y += -c * x**p # the
        return y
    def f2_sindy(x):
        y = np.zeros_like(x)
        for p, c in f2_terms:
            y += -c * x**p
        return y
    def fitted_model_sindy(t, y):
        x, x_dot = y
        x_ddot = F_ext(t) - f1_sindy(x_dot) - f2_sindy(x)
        return [x_dot, x_ddot]
        
    # plotting obtained CCs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
    ax1.plot(xdot_vals, f1_sindy(xdot_vals), label="Identified", linewidth=2)
    ax1.plot(xdot_vals, F1(xdot_vals), '--',color='black', label="Theor.", linewidth=2)
    ax1.set_title("Obtained CCs from SINDy-CC method")
    ax1.set_xlabel(r"$\dot{x}$")
    ax1.set_ylabel(r"f$_1(\dot{x})$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(x_vals, f2_sindy(x_vals), label="Identified", linewidth=2)
    ax2.plot(x_vals, F2(x_vals), '--',color='black', label="Theor.", linewidth=3)
    ax2.set_ylabel(r"f$_2$(x)")
    ax2.set_xlabel("x")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    #plt.tight_layout()
    #plt.show()

#    try:
#        sol = solve_ivp(fitted_model_sindy, t_span, y0, t_eval=t_simul,method='LSODA') #, rtol=1e-9, atol=1e-12)
#        #x_simulated_LS = sol_fit.y[0]       # Posición
#        #x_dot_simulated_LS = sol_fit.y[1]   # Velocidad
#        # check for NaNs or infs just in case
#        if np.any(np.isnan(x_simulated_LS)) or np.any(np.isinf(x_simulated_LS)):
#            print(f"Skipping trial {SNR_dB}: LS simulation returned NaNs or infs.")
#            # continue your RMSE calculations and plots here...
#    except Exception as e:
#        print(f"Skipping trial {SNR_dB}: Exception during LS simulation -> {e}")  
#
#    x_sim = sol.y[0]
#    xdot_sim = sol.y[1]
#    plt.figure(figsize=(8,5))
#    plt.plot(time_data[0:len(x_sim)], x_sim, label="SINDy-CC", linewidth=2)
#    plt.plot(time_data, x_data,"--",color='black', label="Training data", linewidth=2)
#    plt.ylim(-1.5,1.5)
#    plt.xlabel("t")
#    plt.ylabel("x(t)")
#    plt.legend()
#    plt.title("Simulation with SINDy-CC model")
#    plt.show()
    




    ##########################################
    # Evaluation of the f1 and f2 functions
    ##########################################
    predicted_F1_param = f1_param(xdot_vals).flatten() # parametric-CC
    predicted_F1_theor = F1(xdot_vals).flatten() # theor
    predicted_F1_poly = f1_fit_shifted.flatten() # Poly-CC
    predicted_F1_syndycc = f1_sindy(xdot_vals).flatten() #  SINDy-CC
    #predicted_F1_SR = f1_fun_sr(xdot_vals).flatten() # SR
    #predicted_F1_NNCC = predicted_F1_nosym_shifted.flatten()  # NN-CC
    #predicted_F1_NNCC_sym = predicted_F1_sym_shifted.flatten()   # NN-CC(+sym)
    #predicted_F1_NNCC_sym_SR = model_f1SR.predict(xdot_vals.reshape(-1,1)) # NN-CC(+sym+post-SR)

    predicted_F2_param = f2_param(x_vals).flatten() # parametric-CC
    predicted_F2_theor = F2(x_vals).flatten() # theor
    predicted_F2_poly = f2_fit_shifted.flatten() # Poly-CC
    predicted_F2_syndycc = f2_sindy(x_vals).flatten() #  SINDy-CC
    #predicted_F2_SR = f2_fun_sr(x_vals).flatten() # SR
    #predicted_F2_NNCC = predicted_F2_nosym_shifted.flatten()  # NN-CC
    #predicted_F2_NNCC_sym = predicted_F2_sym_shifted.flatten()   # NN-CC(+sym)
    #predicted_F2_NNCC_sym_SR = model_f2SR.predict(x_vals.reshape(-1,1)) # NN-CC(+sym+post-SR)
    
    

    # 1. Define RMSE function
    def rmse(y_true, y_pred):
        # Flattening ensures shapes like (N,1) and (N,) don't cause broadcasting errors
        return np.sqrt(np.mean((y_true.flatten() - y_pred.flatten()) ** 2))

    # 2. Construct the Dictionary with new variables
    rmse_values = {
        "noise_perc_th": noise_percentage_th,
        "noise_perc": noise_percentage,
        "SNR_dB": SNR_dB,
        
        # Other Methods
        "SINDy_f1": rmse(predicted_F1_theor, predicted_F1_syndycc),
        "SINDy_f2": rmse(predicted_F2_theor, predicted_F2_syndycc),
        
        "Poly_f1": rmse(predicted_F1_theor, predicted_F1_poly),
        "Poly_f2": rmse(predicted_F2_theor, predicted_F2_poly),
        
        "Param_f1": rmse(predicted_F1_theor, predicted_F1_param),
        "Param_f2": rmse(predicted_F2_theor, predicted_F2_param),
        
    }

 

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

        plt.close()
        plt.close(fig)
        
        # Random initial conditions uncomment
        x0_val = np.round(np.random.uniform(-0.5, 0.5),3)
        v0_val = np.round(np.random.uniform(-0.5, 0.5),3)
        y0_val = [x0_val, v0_val]
        Aext = np.round(np.random.uniform(1, 1.5),3)
        Omega = np.round(np.random.uniform(0.2, 1.0),3)
        
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
        plt.plot(t_parametric, x_simulated_parametric, "--", label="Parametric", linewidth=2)
        plt.plot(t_simulated_th, x_simulated_th, label="Theor.", linewidth=2)
        plt.xlabel("t")
        plt.ylabel("x(t)")
        plt.legend()
        plt.title("Parametric model simulation")
        plt.show()




        #############################################################        
        ################ SINDY ###### validation of the model
        print("Simulating Sindy without restrictions")
        x_simulated_Sindy_ku0 = []
        x_dot_simulated_Sindy_ku0 = []
        start = time.time()  
        start = time.time()
        x_simulated_NN=[]
        x_dot_simulated_NN=[]
        try:
            u_val_sindy = F_ext(t_val)
            x_val_sindy_ku0 = model_sindy_ku0.simulate(y0_val, t_val, u=u_val_sindy)
            x_simulated_Sindy_ku0 = x_val_sindy_ku0[:, 0]
            x_dot_simulated_Sindy_ku0 = x_val_sindy_ku0[:, 1]

 
            # check for NaNs or infs just in case
            if np.any(np.isnan(x_val_sindy_ku0)) or np.any(np.isinf(x_val_sindy_ku0)):
                print(f"Skipping trial {valid_trials}: SINDy simulation returned NaNs or infs.")
        except Exception as e:
            print(f"Skipping trial {valid_trials}: Exception during SINDy simulation -> {e}")
        end = time.time()  
        elapsed = end - start
        print(f"Solve_ivp finished in {elapsed:.3f} seconds")
                     
                     
        ################ SINDy-CC ###### validation of the model
        print("Simulating Sindy-CC restrictions")
        x_simulated_Sindy = []
        x_dot_simulated_Sindy = []
        start = time.time()
 
        try:
            sol = solve_ivp(fitted_model_sindy, t_span_val, y0_val, t_eval=t_val,method='LSODA')
            x_simulated_Sindy = sol.y[0]       # Posición
            x_dot_simulated_Sindy = sol.y[1]   # Velocidad
            # check for NaNs or infs just in case
            if np.any(np.isnan(x_simulated_Sindy)) or np.any(np.isinf(x_simulated_Sindy)):
                print(f"Skipping trial {valid_trials}: SINDy simulation returned NaNs or infs.")
        except Exception as e:
            print(f"Skipping trial {valid_trials}: Exception during SINDy simulation -> {e}")
        end = time.time()  
        elapsed = end - start
        print(f"Solve_ivp finished in {elapsed:.3f} seconds")


        ################ LS ###### validation of the model
        print("Integrating LS-CC")
        start = time.time()  
        x_simulated_LS = [] #np.zeros(len(t_val))
        x_dot_simulated_LS = [] # np.zeros(len(t_val))
        try:
            sol_fit = solve_ivp(fitted_model_LS, t_span_val, y0_val, t_eval=t_val,method='LSODA')
            x_simulated_LS = sol_fit.y[0]       # Posición
            x_dot_simulated_LS = sol_fit.y[1]   # Velocidad
            # check for NaNs or infs just in case
            if np.any(np.isnan(x_simulated_LS)) or np.any(np.isinf(x_simulated_LS)):
                print(f"Skipping trial {valid_trials}: LS simulation returned NaNs or infs.")
            # continue your RMSE calculations and plots here...
        except Exception as e:
            print(f"Skipping trial {valid_trials}: Exception during LS simulation -> {e}")
        end = time.time()  
        elapsed = end - start
        print(f"Solve_ivp finished in {elapsed:.3f} seconds")


        # CHECK if some integration failed
 
        if len(x_simulated_LS) != len(t_val):
            print("Warning:")
            print("LS-CC finished integration before maximum simulation time")
            print("time:",t_val[len(x_simulated_LS)-1])
        if len(x_simulated_Sindy) != len(t_val):
            print("Warning:")
            print("Sindy-CC finished integration before maximum simulation time")
            print("time:",t_val[len(x_simulated_Sindy)-1])
        if len(x_simulated_Sindy_ku0) != len(t_val):
            print("Warning:")
            print("Sindy finished integration before maximum simulation time")
            print("time:",t_val[len(x_simulated_Sindy_ku0)-1])
 
        
        # Plot the simulation results
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(t_val, x_simulated_th,'-',color='blue', label="$x_{th}$",linewidth='2')
 
        plt.plot(t_val[0:len(x_simulated_LS)], x_simulated_LS,"--",color='red', label="$x_{val} LS$",linewidth='3')
        plt.plot(t_val[0:len(x_simulated_Sindy)], x_simulated_Sindy,"--",color='violet', label="$x_{val} Sindy-CC$",linewidth='3')
        
        plt.plot(t_val[0:len(x_simulated_Sindy_ku0)], x_simulated_Sindy_ku0,"--",color='darkgreen', label="$x_{val} Sindy$",linewidth='3')
        plt.ylim(np.min(x_simulated_th)-0.2,np.max(x_simulated_th)+0.2)
        plt.xlabel("Time $t$")
        plt.ylabel("Position")
        plt.title("Validation Test of Position over Time")
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(t_val, x_dot_simulated_th,color='blue', label="$\\dot{x}_{th}$")
 
        plt.plot(t_val[0:len(x_dot_simulated_LS)], x_dot_simulated_LS,"--",color='red', label="$\\dot{x}_{val} LS$", linestyle="dashed",linewidth='3')
        plt.plot(t_val[0:len(x_dot_simulated_Sindy)], x_dot_simulated_Sindy,"--",color='violet', label="$\\dot{x}_{val} Sindy$",linewidth='3')
        plt.plot(t_val[0:len(x_dot_simulated_Sindy_ku0)], x_dot_simulated_Sindy_ku0,"--",color='darkgreen', label="$\\dot{x}_{val} Sindy$",linewidth='3')#        plt.plot(t_val[0:-1], x_val_sindy[:,1],"--",color='darkgreen', label="$\\dot{x}_{val} Sindy$",linewidth='3')
        plt.ylim(np.min(x_dot_simulated_th)-0.2,np.max(x_dot_simulated_th)+0.2)
        #plt.ylim(-0.75,0.75)
        plt.xlabel("Time $t$")
        plt.ylabel("Velocity")
        plt.title("Neural Network Simulation of Velocity over Time")
        plt.legend()
        plt.grid(True)
        plt.show()



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



#        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
#
#        ## ---- (a) x(t) values ----
#        axs[0, 0].plot(t_val, x_simulated_th, '-', color='blue', label="Theor.", linewidth=3)
#        #axs[0, 0].plot(t_val, x_simulated_NN, "--", color='orange', label=r"NN-CC$_{+sym}$", linewidth=3)
#        #axs[0, 0].plot(t_val, x_simulated_NN_nosym, "--", color='violet', label="NN-CC nosym", linewidth=3)
#        #axs[0, 0].plot(t_val, x_simulated_NN_SR, "--", color='magenta', label=r"NN-CC$_{+sym+post\!\!-\!\!SR}$", linewidth=3)
#        axs[0, 0].plot(t_val[:len(x_simulated_LS)], x_simulated_LS, "--", color='red', label="Poly-CC", linewidth=3)
#        axs[0, 0].plot(t_val[0:len(x_simulated_Sindy)], x_simulated_Sindy, "--", color='darkgreen', label="SINDy-CC", linewidth=3)
#        axs[0, 0].set_ylim(np.min(x_simulated_LS)-0.2, np.max(x_simulated_LS)+0.2)
#        axs[0, 0].set_ylabel("$x$", fontsize=24)
#        axs[0, 0].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1))
#        custom_ticks(axs[0, 0], 20, 0.5 , 5 , 0.25)
#        axs[0, 0].text(0.98, 0.04, "(a)", transform=axs[0, 0].transAxes, fontsize=24, va='bottom', ha='right')
#
#        # ---- (b) x_dot(t) values ----
#        axs[0, 1].plot(t_val, x_dot_simulated_th, '-', color='blue', label="Theor.", linewidth=3)
#        #axs[0, 1].plot(t_val, x_dot_simulated_NN, "--", color='orange', label=r"NN-CC$_{+sym}$", linewidth=3)
#        #axs[0, 1].plot(t_val, x_dot_simulated_NN_nosym, "--", color='violet', label="NN-CC nosym", linewidth=3)
#        #axs[0, 1].plot(t_val, x_dot_simulated_NN_SR, "--", color='magenta', label=r"NN-CC$_{+sym+post\!\!-\!\!SR}$", linewidth=3)
#        axs[0, 1].plot(t_val[:len(x_dot_simulated_LS)], x_dot_simulated_LS, "--", color='red', label="Poly-CC", linewidth=3)
#        axs[0, 1].plot(t_val[0:len(x_dot_simulated_Sindy)], x_dot_simulated_Sindy, "--", color='darkgreen', label="SINDy-CC", linewidth=3)
#        axs[0, 1].set_ylim(np.min(x_dot_simulated_LS)-0.2, np.max(x_dot_simulated_LS)+0.2)
#        axs[0, 1].set_ylabel("$\dot{x}$", fontsize=24)
#        axs[0, 1].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1))
#        custom_ticks(axs[0, 1], 20, 0.5, 5, 0.25)
#        axs[0, 1].text(0.98, 0.04, "(b)", transform=axs[0, 1].transAxes, fontsize=24, va='bottom', ha='right')
#
#        # ---- (c) Residuals: x_model - x_theor ----
#        #axs[1, 0].plot(t_val, x_simulated_NN - x_simulated_th, '--', color='orange', label=r"NN-CC$_{+sym}$", linewidth=3)
#        axs[1, 0].plot(t_val[:len(x_simulated_LS)], x_simulated_LS - x_simulated_th[:len(x_simulated_LS)], '--', color='red', label="Poly-CC", linewidth=3)
#        axs[1, 0].plot(t_val[0:len(x_simulated_Sindy)], x_simulated_Sindy - x_simulated_th[0:len(x_simulated_Sindy)], '--', color='darkgreen', label="SINDy-CC", linewidth=3)
#        axs[1, 0].axhline(0, color='black', linewidth=1)
#        axs[1, 0].set_ylabel("$x-x_{th.}$", fontsize=22)
#        axs[1, 0].set_xlabel("$t$", fontsize=24)
#        axs[1, 0].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1))
#        custom_ticks(axs[1, 0], 20, 0.2, 5, 0.1)
#        axs[1, 0].text(0.9, 0.04, "(c)", transform=axs[1, 0].transAxes, fontsize=24, va='bottom', ha='right')
#
#        # ---- (d) Residuals: x_dot_model - x_dot_theor ----
#        #axs[1, 1].plot(t_val, x_dot_simulated_NN - x_dot_simulated_th, '--', color='orange', label=r"NN-CC$_{+sym}$", linewidth=3)
#        axs[1, 1].plot(t_val[:len(x_dot_simulated_LS)], x_dot_simulated_LS - x_dot_simulated_th[:len(x_dot_simulated_LS)], '--', color='red', label="Poly-CC", linewidth=3)
#        axs[1, 1].plot(t_val[0:len(x_dot_simulated_Sindy)], x_dot_simulated_Sindy - x_dot_simulated_th[0:len(x_simulated_Sindy)], '--', color='darkgreen', label="SINDy-CC", linewidth=3)
#        axs[1, 1].axhline(0, color='black', linewidth=1)
#        axs[1, 1].set_ylabel("$\dot{x}-\dot{x}_{th.}$", fontsize=22)
#        axs[1, 1].set_xlabel("$t$", fontsize=24)
#        axs[1, 1].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1))
#        custom_ticks(axs[1, 1], 20, 0.1, 5, 0.05)
#        axs[1, 1].text(0.9, 0.04, "(d)", transform=axs[1, 1].transAxes, fontsize=24, va='bottom', ha='right')
#
#        #plt.tight_layout()
#
#        # Save
#        folder_path = output_path
#        # "/content/drive/My Drive/Colab Notebooks/Plots"
#        os.makedirs(folder_path, exist_ok=True)
#        file_name = "valid_duffing.pdf"
#        file_path = os.path.join(folder_path, file_name)
##        plt.savefig(file_path, format='pdf', bbox_inches='tight')
##       plt.show()
#        print(f"Saved to: {file_path}")
#        plt.close()



        # Calculation of absolute value separation
        threshold_chaos = 0.2 # 4
        
 
 
        time_chaos_x_parametric=t_val[len(x_simulated_th)-1]
        for i in range(len(x_simulated_parametric)):
            diff = abs(x_simulated_parametric[i] - x_simulated_th[i])
            if diff > threshold_chaos:
#                time_chaos_x_parametric_list.append(t_val[i])
                time_chaos_x_parametric=t_val[i]
                break
        time_chaos_x_parametric_list.append(time_chaos_x_parametric)
        #time_chaos_x_NN_list.append(t_val[-1])
        
 
        time_chaos_x_LS=t_val[len(x_simulated_LS)-1]
        for i in range(len(x_simulated_LS)):
            diff = abs(x_simulated_LS[i] - x_simulated_th[i])
            if diff > threshold_chaos:
                #time_chaos_x_LS_list.append(t_val[i])
                time_chaos_x_LS=t_val[i]
                break
        time_chaos_x_LS_list.append(time_chaos_x_LS)
        #time_chaos_x_Sindy_list.append(t_val[-1])

        time_chaos_x_Sindy=t_val[len(x_simulated_Sindy)-1]
        for i in range(len(x_simulated_Sindy)):
            diff = abs(x_simulated_Sindy[i] - x_simulated_th[i])
            if diff > threshold_chaos:
                #time_chaos_x_Sindy_list.append(t_val[i])
                time_chaos_x_Sindy=t_val[i]
                break
        time_chaos_x_Sindy_list.append(time_chaos_x_Sindy)
        #time_chaos_x_Sindy_ku0_list.append(t_val[-1])
        
        time_chaos_x_Sindy_ku0=t_val[-1]
        for i in range(len(x_simulated_Sindy_ku0)):
            diff = abs(x_simulated_Sindy_ku0[i] - x_simulated_th[i])
            if diff > threshold_chaos:
                #time_chaos_x_Sindy_ku0_list.append(t_val[i])
                time_chaos_x_Sindy_ku0=t_val[i]
                break
        time_chaos_x_Sindy_ku0_list.append(time_chaos_x_Sindy_ku0)

        time_matrix_append = np.column_stack([
            noise_percentage_th,
            noise_percentage,
            SNR_dB,
            #time_chaos_x_NN_nosym,
            #time_chaos_x_NN_SR_nosym,
            #time_chaos_x_NN,
            #time_chaos_x_NN_SR,
            time_chaos_x_Sindy,
            time_chaos_x_Sindy_ku0,
            time_chaos_x_LS,
            #time_chaos_x_SR,
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
                       header="nois_th nois nois_db  Sindy Sindy_ku0  Poly  Parametric" if not file_exists else '',
                       #header="# noise_th noise noise_db t_SR t_parametric" if not file_exists else '',
                       fmt="%.2f", delimiter=" ", comments='')



        # Calculation of RMSE values
        #  print(f"Trial {i+1} : x0={x0_val:.4f} ; v0={v0_val:.4f}")

    #    print("parametric results")
        rmse_x_parametric = np.sqrt(np.mean((x_simulated_th[0:len(x_simulated_parametric)] - x_simulated_parametric) ** 2))
        rmse_x_dot_parametric = np.sqrt(np.mean((x_dot_simulated_th[0:len(x_dot_simulated_parametric)] - x_dot_simulated_parametric) ** 2))
        print(f"parametric results\t - x: {rmse_x_parametric:.6f}, x': {rmse_x_dot_parametric:.6f}")
        #print(f"RMSE for Position (x): {rmse_x_parametric:.6f}")
        #print(f"RMSE for Velocity (x'): {rmse_x_dot_parametric:.6f}")
        rmse_x_parametric_list.append(rmse_x_parametric)
        rmse_x_dot_parametric_list.append(rmse_x_dot_parametric)


    #    print("Sindy results")
        rmse_x_Sindy = np.sqrt(np.mean((x_simulated_th[0:len(x_simulated_Sindy)] - x_simulated_Sindy) ** 2))
        rmse_x_dot_Sindy = np.sqrt(np.mean((x_dot_simulated_th[0:len(x_dot_simulated_Sindy)] - x_dot_simulated_Sindy) ** 2))
        print(f"Sindy results\t - x: {rmse_x_Sindy:.6f}, x': {rmse_x_dot_Sindy:.6f}")
        #print(f"RMSE for Position (x): {rmse_x_sindy:.6f}")
        #print(f"RMSE for Velocity (x'): {rmse_x_dot_sindy:.6f}")
        rmse_x_Sindy_list.append(rmse_x_Sindy)
        rmse_x_dot_Sindy_list.append(rmse_x_dot_Sindy)


    #    print("Sindy results")
        rmse_x_Sindy_ku0 = np.sqrt(np.mean((x_simulated_th[0:len(x_simulated_Sindy_ku0)] - x_simulated_Sindy_ku0) ** 2))
        rmse_x_dot_Sindy_ku0 = np.sqrt(np.mean((x_dot_simulated_th[0:len(x_dot_simulated_Sindy_ku0)] - x_dot_simulated_Sindy_ku0) ** 2))
        print(f"Sindy ku0 results\t - x: {rmse_x_Sindy_ku0:.6f}, x': {rmse_x_dot_Sindy_ku0:.6f}")
        #print(f"RMSE for Position (x): {rmse_x_sindy:.6f}")
        #print(f"RMSE for Velocity (x'): {rmse_x_dot_sindy:.6f}")
        rmse_x_Sindy_ku0_list.append(rmse_x_Sindy_ku0)
        rmse_x_dot_Sindy_ku0_list.append(rmse_x_dot_Sindy_ku0)

     #   print("LS results")
        rmse_x_LS = np.sqrt(np.mean((x_simulated_th[0:len(x_simulated_LS)] - x_simulated_LS) ** 2))
        rmse_x_dot_LS = np.sqrt(np.mean((x_dot_simulated_th[0:len(x_dot_simulated_LS)] - x_dot_simulated_LS) ** 2))
        print(f"LS results   \t - x: {rmse_x_LS:.6f}, x': {rmse_x_dot_LS:.6f}")
        #print(f"RMSE for Position (x): {rmse_x_LS:.6f}")
        #print(f"RMSE for Velocity (x'): {rmse_x_dot_LS:.6f}")
        rmse_x_LS_list.append(rmse_x_LS)
        rmse_x_dot_LS_list.append(rmse_x_dot_LS)

        rmse_matrix_append = np.column_stack([
            noise_percentage_th,
            noise_percentage,
            SNR_dB,
            rmse_x_Sindy,
            rmse_x_dot_Sindy,
            rmse_x_Sindy_ku0,
            rmse_x_dot_Sindy_ku0,
            rmse_x_LS,
            rmse_x_dot_LS,
            rmse_x_parametric,
            rmse_x_dot_parametric,
            
        ])
        folder_path = output_path
        os.makedirs(folder_path, exist_ok=True)
        file_name = "rmse_noise_x_xdot.txt"
        file_path = os.path.join(folder_path, file_name)
        # Save with header and space as delimiter
        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a') as f:
            np.savetxt(f, rmse_matrix_append,
                       header="noise_th noise noise_db x_Sindy xdot_Sindy x_Sindy_ku0 xdot_Sindy_ku0 x_LS xdot_LS x_param xdot_param" if not file_exists else '',
                       fmt="%.8f", delimiter=" ", comments='')


    # Compute overall RMSE and standard deviation
    total_rmse_x_Sindy = np.mean(rmse_x_Sindy_list)
    std_rmse_x_Sindy = np.std(rmse_x_Sindy_list)
    total_rmse_x_dot_Sindy = np.mean(rmse_x_dot_Sindy_list)
    std_rmse_x_dot_Sindy = np.std(rmse_x_dot_Sindy_list)
    total_rmse_x_Sindy_ku0 = np.mean(rmse_x_Sindy_ku0_list)
    std_rmse_x_Sindy_ku0 = np.std(rmse_x_Sindy_ku0_list)
    total_rmse_x_dot_Sindy_ku0 = np.mean(rmse_x_dot_Sindy_ku0_list)
    std_rmse_x_dot_Sindy_ku0 = np.std(rmse_x_dot_Sindy_ku0_list)
    total_rmse_x_LS = np.mean(rmse_x_LS_list)
    std_rmse_x_LS = np.std(rmse_x_LS_list)
    total_rmse_x_dot_LS = np.mean(rmse_x_dot_LS_list)
    std_rmse_x_dot_LS = np.std(rmse_x_dot_LS_list)
    
    # Print results
    print("\n======= Total RMSE over all trials (mean ± std, % std) =======")
    print("SINDy results")
    print(f"Position (x):     {total_rmse_x_Sindy:.6f} ± {std_rmse_x_Sindy:.6f}  ({std_rmse_x_Sindy/total_rmse_x_Sindy*100:.6f}%)")
    print(f"Velocity (x'):    {total_rmse_x_dot_Sindy:.6f} ± {std_rmse_x_dot_Sindy:.6f}  ({std_rmse_x_dot_Sindy/total_rmse_x_dot_Sindy*100:.6f}%)")
    print("SINDy ku0 results")
    print(f"Position (x):     {total_rmse_x_Sindy_ku0:.6f} ± {std_rmse_x_Sindy_ku0:.6f}  ({std_rmse_x_Sindy_ku0/total_rmse_x_Sindy_ku0*100:.6f}%)")
    print(f"Velocity (x'):    {total_rmse_x_dot_Sindy_ku0:.6f} ± {std_rmse_x_dot_Sindy_ku0:.6f}  ({std_rmse_x_dot_Sindy_ku0/total_rmse_x_dot_Sindy_ku0*100:.6f}%)")
    print("LS results")
    print(f"Position (x):     {total_rmse_x_LS:.6f} ± {std_rmse_x_LS:.6f}  ({std_rmse_x_LS/total_rmse_x_LS*100:.6f}%)")
    print(f"Velocity (x'):    {total_rmse_x_dot_LS:.6f} ± {std_rmse_x_dot_LS:.6f}  ({std_rmse_x_dot_LS/total_rmse_x_dot_LS*100:.6f}%)")


    # These lists must already be defined
    # Each one should contain RMSE values over multiple trials
    # e.g., rmse_x_NN_list = [rmse_trial1, rmse_trial2, ..., rmse_trialN]
    # same for other methods
    rmse_data = [
        rmse_x_Sindy_list,
        rmse_x_Sindy_ku0_list,
        rmse_x_LS_list,
    ]
    labels = ['SINDy-CC', 'SINDy_ku0', 'Poly-CC']
    plt.figure(figsize=(6, 6))
    bp = plt.boxplot(
        rmse_data, labels=labels, patch_artist=True,
        boxprops=dict(facecolor='skyblue', color='black', linewidth=1.5),
        medianprops=dict(color='darkred', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.2),
        capprops=dict(color='black', linewidth=1.2),
        flierprops=dict(marker='o', markerfacecolor='gray', markersize=5, linestyle='none')
    )
    # Log scale with minor ticks
    plt.yscale('log')
    plt.ylim(1e-3,1e-0)
    plt.minorticks_on()
    plt.tick_params(axis='y', which='both', direction='in', length=9, width=2,left=True,right=True)
    plt.tick_params(axis='y', which='minor', length=7, width=1.5)
    plt.tick_params(axis='x', which='both', direction='in', length=9, width=2)
    plt.tick_params(axis='x', which='minor', bottom=False, top=False)  # Disable minor ticks on x
    plt.ylabel("RMSE values for $x$", fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=20)
    plt.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)
    plt.grid(True, which='minor', axis='y', linestyle=':', alpha=0.4)
    # plt.title("Boxplot of RMSE for Position (x)")
    #plt.tight_layout()
    plt.show()
    rmse_dot_data = [
        rmse_x_dot_Sindy_list,
        rmse_x_dot_Sindy_ku0_list,
        rmse_x_dot_LS_list,
    ]
    labels = ['SINDy-CC', 'SINDy_ku0', 'Poly-CC']
    plt.figure(figsize=(6, 6))
    bp = plt.boxplot(
        rmse_dot_data, labels=labels, patch_artist=True,
        boxprops=dict(facecolor='skyblue', color='black', linewidth=1.5),
        medianprops=dict(color='darkred', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.2),
        capprops=dict(color='black', linewidth=1.2),
        flierprops=dict(marker='o', markerfacecolor='gray', markersize=5, linestyle='none')
    )
    # Log scale with minor ticks
    plt.yscale('log')
    plt.ylim(1e-3,1e-0)
    plt.minorticks_on()
    plt.tick_params(axis='y', which='both', direction='in', length=9, width=2,left=True,right=True)
    plt.tick_params(axis='y', which='minor', length=7, width=1.5)
    plt.tick_params(axis='x', which='both', direction='in', length=9, width=2)
    plt.tick_params(axis='x', which='minor', bottom=False, top=False)  # Disable minor ticks on x
    plt.ylabel("RMSE values for $\\dot{x}$", fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=20)
    plt.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)
    plt.grid(True, which='minor', axis='y', linestyle=':', alpha=0.4)
    # plt.title("Boxplot of RMSE for Position (x)")
    #plt.tight_layout()
    plt.show()
    # Stack the RMSE arrays column-wise
    rmse_matrix = np.column_stack([
        rmse_x_Sindy_list,
        rmse_x_dot_Sindy_list,
        rmse_x_Sindy_ku0_list,
        rmse_x_dot_Sindy_ku0_list,
        rmse_x_LS_list,
        rmse_x_dot_LS_list,
    ])
    folder_path = output_path
    os.makedirs(folder_path, exist_ok=True)
    file_name = "rmse_results_duffing_NN_without_symmetry.txt"
    file_path = os.path.join(folder_path, file_name)
    # Save with header and space as delimiter
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a') as f:
        np.savetxt(f, rmse_matrix,
                   header="noise_th noise_meas noise_db rmse_x_Sindy rmse_x_dot_Sindy rmse_x_Sindy_ku0 rmse_x_dot_Sindy_ku0 rmse_x_LS rmse_x_dot_LS" if not file_exists else '',
                   fmt="%.8f", delimiter=" ", comments='')
    #np.savetxt(file_path, rmse_matrix,
    #           header="rmse_x_NN rmse_x_dot_NN rmse_x_Sindy rmse_x_dot_Sindy rmse_x_LS rmse_x_dot_LS",
    #           fmt="%.8f", delimiter=" ")



#test_predictions = model(test_inputs).cpu().numpy()  # Move predictions to CPU for plotting



