  
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

from pysr import PySRRegressor
import sympy as sp
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
time_chaos_x_Sindy_sym_list =[]
time_chaos_x_Sindy_sym_SR_list =[]
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
rmse_x_Sindy_sym_list = []
rmse_x_dot_Sindy_sym_list = []
rmse_x_Sindy_sym_SR_list = []
rmse_x_dot_Sindy_sym_SR_list = []
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
#SNR_dB_list = list(np.linspace(20, -20, 41 ))  # ∞, 20, 17.5, ..., -5

#SNR_dB_list = [np.inf] + list(np.linspace(40, 5, 36 ))  # ∞, 20, 17.5, ..., -5
#SNR_dB_list = list(np.linspace(-18, -20, 3 ))  # ∞, 20, 17.5, ..., -5

#SNR_dB_list = list(np.linspace(5, -20, 26 ))  # ∞, 20, 17.5, ..., -5
 

#SNR_dB_list = [20.0]

#repeat 3 times each value in the list
SNR_dB_list = np.repeat(SNR_dB_list, 10)


#SNR_dB_list = np.repeat(SNR_dB_list,B_list = list(np.linspace(5, -5, 3))  # ∞, 20, 17.5, ..., -5


for SNR_dB in SNR_dB_list:
    Aext=0.5
    alpha=-1.05
    beta=1.02
    delta=0.3
    Omega=1.2
    x0=0.5
    v0=-0.5
    y0 = [x0, v0]  # [x(0), x'(0)]
    
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

#    # Plot Differentiation Results
#    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
#    ax[0].set_title(f"Differentiation Assessment (SNR={SNR_dB}dB)")
#    
#    ax[0].plot(time_data, x_data_clean, 'k-', lw=1, label='True')
#    ax[0].plot(time_data, x_noisy, 'r.', ms=2, alpha=0.3, label='Noisy Measure')
#    ax[0].plot(time_data, x_data, 'b--', lw=1.5, label='SG Smooth')
#    ax[0].set_ylabel("x")
#    ax[0].legend()
#    
#    ax[1].plot(time_data, x_dot_data_clean, 'k-', lw=1, label='True')
#    ax[1].plot(time_data, x_dot_data, 'b--', lw=1.5, label='SG Estimate')
#    ax[1].set_ylabel("v")
#    
#    ax[2].plot(time_data, x_ddot_data_clean, 'k-', lw=1, label='True')
#    ax[2].plot(time_data, x_ddot_data, 'b--', lw=1.5, label='SG Estimate')
#    ax[2].set_ylabel("a")
#    ax[2].set_xlabel("Time")
#    plt.tight_layout()
#    plt.show()


    # Define ranges for plotting CCs later
    x_vals = np.linspace(np.min(x_data), np.max(x_data), NevalCC)
    xdot_vals = np.linspace(np.min(x_dot_data), np.max(x_dot_data), NevalCC)
    F1_th=F1(x_dot_data_clean)
    F2_th=F2(x_data_clean)
##############################################3

#    # plot theoretical integrations
#    plt.figure()
#    plt.title("Theoretical ODE integration: Consistency Check")
#    plt.plot(time_data, (x_ddot_data + F1_th + F2_th - F_ext(time_data))**2)
#    plt.xlabel("t")
#    plt.ylabel(r"MSE $(\ddot{x} + F_1(\dot{x}) + F_2(x) - F_{ext})^2$")#$(\ddot{x} - \ddot{x}_{model})^2$")
#    plt.grid(True, alpha=0.3)
#    plt.show()
#    #plt.plot(time_data, x_data)
#    #plt.xlabel("Time")
#    #plt.ylabel("x(t)")
#    #plt.title("Theoretical data")
#    #plt.grid(True)
#    #plt.show()
#    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
#    ax1.plot(time_data, x_data, color='black')
#    ax1.set_ylabel("x(t)")
#    ax1.set_title("Theoretical Data (without noise)")
#    ax1.grid(True)
#    ax2.plot(time_data, F_ext(time_data), color='black', linestyle='-')
#    ax2.set_xlabel("t")
#    ax2.set_ylabel(r"F$_{ext}(t)$")
#    ax2.grid(True)
#    plt.tight_layout()
#   plt.show()    


   
 
     
 
 

    # range of training data values
    print("min(x) , max(x)=", np.min(x_data),",",np.max(x_data))
    print(f"min(ẋ) , max(ẋ)= {np.min(x_dot_data)} , {np.max(x_dot_data)}")

    ############################################
    ########### IDENTIFICATION #################
    ############################################




    ####################################################
    ############# Parametric-CC  #################   least squares
    # Right-hand side
    rhs = F_ext(time_data) -  x_ddot_data 
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
    #plt.tight_layout()
    plt.show()
    plt.close()
# Integrate forward the obtained Parametric-CC model for verification
    # (using the same ICs and driven force as the training data)
#    sol = solve_ivp(ode_param, t_span, y0, t_eval=t_simul,method='LSODA') 
#    x_sim = sol.y[0]
#    xdot_sim = sol.y[1]
#    plt.figure(figsize=(8,5))
#    plt.plot(time_data[0:len(x_sim)], x_sim, label="Parametric-CC", linewidth=2)
#    plt.plot(time_data, x_data, "--", color='black', label="Training data", linewidth=2)
#    plt.xlabel("t")
#    plt.ylabel("x(t)")
#    plt.legend()
#    plt.title("Simulation with Parametric-CC model")
#    plt.show()


    
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
    bmat = x_ddot_data -  F_ext(x_data) #F_ext(t_simul)       # Leftover = f1(x') + f2(x)
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
    plt.show()
    plt.close()
# 
#    sol = solve_ivp(fitted_model_LS, t_span, y0, t_eval=t_simul,method='LSODA') 
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
    u_sindy = F_ext(x_data) # np.stack((F_ext_noise_data),axis=1)
    u_sindy_resh = F_ext(x_data).reshape(-1,1) #flatten()
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

# ==========================================================
    # === EXTRACT CCs FOR SINDy_ku0 (variable force coeff) ===
    # ==========================================================
    
    # Get coefficients for the second equation (x_ddot)
    # The model predicts [x_dot, x_ddot], so we want index 1
    coefs_ku0 = model_sindy_ku0.coefficients()[1, :]
    features_ku0 = model_sindy_ku0.get_feature_names()
    
    f1_terms_ku0 = []
    f2_terms_ku0 = []
    
    # Parse the terms to separate f1(x_dot) and f2(x)
    # Theoretical Model: x'' = k*u - f1(x') - f2(x)
    # SINDy Model:       x'' = k*u + sum(c_i * theta_i)
    # Therefore:         f_term = -1 * sindy_term
    
    for feat, c in zip(features_ku0, coefs_ku0):
        # Default feature names are x0 (x), x1 (x_dot), u0 (force)
        
        if 'u' in feat: 
            continue # Skip forcing terms (u0)
            
        if 'x1' in feat: # Dependence on x_dot -> f1
            if feat == "x1": 
                p = 1
            elif "^" in feat: 
                p = int(feat.split("^")[1])
            else: 
                p = 0 
            f1_terms_ku0.append((p, c))
            
        elif 'x0' in feat: # Dependence on x -> f2
            if feat == "x0": 
                p = 1
            elif "^" in feat: 
                p = int(feat.split("^")[1])
            else: 
                p = 0
            f2_terms_ku0.append((p, c))

    # Define the reconstruction functions
    def f1_sindy_ku0_func(x_in):
        y = np.zeros_like(x_in)
        for p, c in f1_terms_ku0:
            y += -c * x_in**p # Minus sign because x'' = F - f1 - f2
        return y

    def f2_sindy_ku0_func(x_in):
        y = np.zeros_like(x_in)
        for p, c in f2_terms_ku0:
            y += -c * x_in**p
        return y

#    ################ SINDY without restrictions ###### validation of the model
#    print("Simulating Sindy without restrictions")
#    x_simulated_Sindy_ku0 = []
#    x_dot_simulated_Sindy_ku0 = []
#    start = time.time()
#    try:
#        u_val_sindy = F_ext(time_data)
#        x_val_sindy_ku0 = model_sindy_ku0.simulate(y0, t_simul, u=u_val_sindy)
#        x_simulated_Sindy_ku0 = x_val_sindy_ku0[:, 0]
#        x_dot_simulated_Sindy_ku0 = x_val_sindy_ku0[:, 1]
#        # check for NaNs or infs just in case
#        if np.any(np.isnan(x_val_sindy_ku0)) or np.any(np.isinf(x_val_sindy_ku0)):
#            print(f"Skipping trial {i+1}: SINDy simulation returned NaNs or infs.")
#    except Exception as e:
#        print(f"Skipping trial {i+1}: Exception during SINDy simulation -> {e}")
#    end = time.time()  
#    elapsed = end - start
#    print(f"Solve_ivp finished in {elapsed:.3f} seconds")
#    plt.figure(figsize=(8,5))
#    plt.plot(time_data[0:len(x_simulated_Sindy_ku0)], x_simulated_Sindy_ku0, label="SINDy", linewidth=2)
#    plt.plot(time_data, x_data,"--",color='black', label="Training data", linewidth=2)
#    plt.xlabel("t")
#    plt.ylabel("x(t)")
#    plt.legend()
#    plt.title("Simulation with baseline SINDy model")
#    plt.show()

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
    plt.show()
    plt.close()

#    sol = solve_ivp(fitted_model_sindy, t_span, y0, t_eval=t_simul,method='LSODA') 
#    x_sim = sol.y[0]
#    xdot_sim = sol.y[1]
#    plt.figure(figsize=(8,5))
#    plt.plot(time_data[0:len(x_sim)], x_sim, label="SINDy-CC", linewidth=2)
#    plt.plot(time_data, x_data,"--",color='black', label="Training data", linewidth=2)
#    plt.xlabel("t")
#    plt.ylabel("x(t)")
#    plt.legend()
#    plt.title("Simulation with SINDy-CC model")
#    plt.show()
    
    ##############################################
    #############  SINDY Symmetric (Odd only) ####    
    ##############################################

    print("\n--- SINDY WITH SYMMETRY CONSTRAINTS (ODD ONLY) ---")
    
    # 1. Define Odd Library for x and x_dot
    # -----------------------------------------------------------
    odd_degrees = [1, 3, 5, 7, 9]

    # Helper to ensure robust power calculation
    def safe_power(x, k):
        return np.sign(x) * (np.abs(x) ** k)

    # Factory functions to ensure lambda strictly takes 1 argument
    def make_odd_func(k):
        return lambda x: safe_power(x, k)
    
    def make_odd_name(k):
        if k == 1:
            return lambda x: f"{x}"
        else:
            return lambda x: f"{x}^{k}"

    odd_library = ps.CustomLibrary(
        library_functions=[make_odd_func(k) for k in odd_degrees], 
        function_names=[make_odd_name(k) for k in odd_degrees],
        include_bias=False
    )
    
    # 2. Define Linear Library for u
    # -----------------------------------------------------------
    identity_library = ps.IdentityLibrary() 
    
    # 3. Combine with GeneralizedLibrary
    # -----------------------------------------------------------
    # We map x(0) and x_dot(1) to the Odd Library.
    # We map u(2) to the Identity Library.
    # PADDING TRICK: We pass [2, 2] to make the array rectangular (2x2).
    # This creates two identical features for u. We will constrain the second one to 0.
    inputs_idx_sym = np.array([[0, 1], [2, 2]], dtype=int)
    
    combined_library_sym = ps.GeneralizedLibrary(
        [odd_library, identity_library],
        inputs_per_library=inputs_idx_sym
    )

    # 4. Train Unconstrained Model first (to get feature structure)
    # -----------------------------------------------------------
    print("Training SINDy Symmetric (Unconstrained step)...")
    
    # We use a simple optimizer just to generate the feature names list
    model_temp = ps.SINDy(feature_library=combined_library_sym, optimizer=ps.STLSQ(max_iter=5000))
    model_temp.fit(X_sindy, t=time_data, x_dot=X_dot_sindy, u=u_sindy)
    
    feats = model_temp.get_feature_names()
    n_features = len(feats)
    print(f"Features generated: {feats}")

    # 5. Define Strict Constraints
    # -----------------------------------------------------------
    # We want to constrain the SECOND equation (x_ddot).
    # We need to find the indices of the 'u' terms.
    # Because we passed [2, 2], there will be TWO 'u' features (e.g., 'u0' and 'u1' or 'u' and 'u').
    
    u_indices = [i for i, f in enumerate(feats) if 'u' in f]
    print(f"Indices of 'u' terms: {u_indices}")
    
    if len(u_indices) < 1:
        print("ERROR: No 'u' terms found. Check variable names.")
    else:
        # We need constraints for Eq 2. The coefficients for Eq 2 start at index n_features.
        
        # We assume we have at least one 'u'. 
        # If we have duplicate 'u's (due to padding), we constrain the rest to 0.
        n_constraints = len(u_indices)
        
        C_sym = np.zeros((n_constraints, n_features * 2))
        d_sym = np.zeros(n_constraints)
        
        # Constraint 1: First 'u' must have coefficient 1.0 (Physics: +1*u)
        # Location: Eq 2 offset + index of first u
        idx_u1 = n_features + u_indices[0] 
        C_sym[0, idx_u1] = 1.0
        d_sym[0] = 1.0
        
        # Constraint 2+: Any extra 'u' terms (padding artifacts) must be 0.0
        for i in range(1, n_constraints):
            idx_u_extra = n_features + u_indices[i]
            C_sym[i, idx_u_extra] = 1.0
            d_sym[i] = 0.0
            
        print("Applying strict 1*u constraints...")

        # 6. Train Final Constrained Model
        # -----------------------------------------------------------
        optimizer_sym_1u0 = ConstrainedSR3(
            constraint_lhs=C_sym,
            constraint_rhs=d_sym,
            constraint_order="target",    
            threshold=1e-5, 
            max_iter=10000,
            equality_constraints=True
        )
        
        model_sindy_sym = ps.SINDy(
            feature_library=combined_library_sym, 
            optimizer=optimizer_sym_1u0,
            feature_names=['x', 'x_dot', 'u'] # Explicit names
        )
        
        try:
            model_sindy_sym.fit(X_sindy, t=time_data, x_dot=X_dot_sindy, u=u_sindy)
            print("Constrained Model Coefficients:")
            model_sindy_sym.print()
            success_sym = True
        except Exception as e:
            print(f"FAILED to fit SINDy Symmetric. Error: {e}")
            success_sym = False

    
    # 7. Extract Functions f1 and f2 (Verified)
    # -----------------------------------------------------------
    # Clear lists explicitly to ensure no stale data from previous runs
    f1_terms_sym = []
    f2_terms_sym = []

    if success_sym:
        coefs = model_sindy_sym.coefficients()[1, :] # Eq for x_ddot
        feats = model_sindy_sym.get_feature_names()

        print("\n--- Parsing SINDy Terms for Reconstruction ---")
        for feat, c in zip(feats, coefs):
            # 1. Skip negligible terms
            if np.abs(c) < 1e-6: continue

            # 2. Skip forcing u
            if 'u' in feat: 
                print(f"  Skipping forcing: {c:.4f} * {feat}")
                continue 

            # 3. Parse Power
            p = 1
            if "^" in feat:
                p = int(feat.split("^")[1])
            
            # 4. Sort into f1 (velocity) or f2 (position)
            # Logic: "dot" means velocity. "x" without "dot" means position.
            if "dot" in feat:
                print(f"  f1 (Damping)   += {c:.4f} * x_dot^{p}")
                f1_terms_sym.append((p, c))
            elif "x" in feat:
                print(f"  f2 (Stiffness) += {c:.4f} * x^{p}")
                f2_terms_sym.append((p, c))
    
    print(f"  -> f1 terms list: {f1_terms_sym}")
    print(f"  -> f2 terms list: {f2_terms_sym}")

    # Define Reconstruction Functions
    # CRITICAL SIGN CONVENTION: 
    # ODE: x'' = F_ext - f1 - f2
    # SINDy: x'' = F_ext + (terms)
    # Therefore: -f1 - f2 = (terms)  =>  f1 + f2 = -(terms)
    
    def f1_sindy_sym(x_in):
        y = np.zeros_like(x_in)
        for p, c in f1_terms_sym:
            # We sum the terms found by SINDy: (c * x^p)
            y += c * (x_in**p)
        return -y  # Return NEGATIVE to match f1 definition

    def f2_sindy_sym(x_in):
        y = np.zeros_like(x_in)
        for p, c in f2_terms_sym:
            y += c * (x_in**p)
        return -y  # Return NEGATIVE to match f2 definition
    def fitted_model_sindy_sym(t, y):
        x, x_dot = y
        x_ddot = F_ext(t) - f1_sindy_sym(x_dot) - f2_sindy_sym(x)
        return [x_dot, x_ddot]
    # Quick Verification Plot
    # (Optional: run this to visually confirm the functions match theory)
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(xdot_vals, F1(xdot_vals), 'k-', label='Theoretical f1')
    plt.plot(xdot_vals, f1_sindy_sym(xdot_vals), 'r--', label='SINDy f1')
    plt.title(f"f1 Comparison (RMSE: {np.sqrt(np.mean((F1(xdot_vals)-f1_sindy_sym(xdot_vals))**2)):.2e})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(x_vals, F2(x_vals), 'k-', label='Theoretical f2')
    plt.plot(x_vals, f2_sindy_sym(x_vals), 'r--', label='SINDy f2')
    plt.title(f"f2 Comparison (RMSE: {np.sqrt(np.mean((F2(x_vals)-f2_sindy_sym(x_vals))**2)):.2e})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()
    
    # 8. Unconstrained Version (k*u) for comparison
    # -----------------------------------------------------------
    # We simply re-run with STLSQ but same library structure
    # This allows u to vary (e.g., 0.98*u or 1.02*u) but NO u^3, u^5 etc.
    print("\nTraining Unconstrained (k*u) version...")
    model_sindy_sym_ku0 = ps.SINDy(feature_library=combined_library_sym, optimizer=ps.STLSQ(threshold=1e-3))
    model_sindy_sym_ku0.fit(X_sindy, t=time_data, x_dot=X_dot_sindy, u=u_sindy)
    
    f1_terms_ku0 = []
    f2_terms_ku0 = []
    
    coefs_ku0 = model_sindy_sym_ku0.coefficients()[1, :]
    feats_ku0 = model_sindy_sym_ku0.get_feature_names()
    
    for feat, c in zip(feats_ku0, coefs_ku0):
        if 'u' in feat: continue
        p = 1
        if "^" in feat: p = int(feat.split("^")[1])
        if "x_dot" in feat: f1_terms_ku0.append((p, c))
        elif "x" in feat and "dot" not in feat: f2_terms_ku0.append((p, c))

    def f1_sindy_sym_ku0_func(x_in):
        y = np.zeros_like(x_in)
        for p, c in f1_terms_ku0: y += -c * x_in**p 
        return y
    def f2_sindy_sym_ku0_func(x_in):
        y = np.zeros_like(x_in)
        for p, c in f2_terms_ku0: y += -c * x_in**p
        return y

    # Plot Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    ax1.plot(xdot_vals, f1_sindy_sym(xdot_vals), 'r--', label="SINDy-Sym", linewidth=2)
    ax1.plot(xdot_vals, F1(xdot_vals), 'k-', label="Theor.", linewidth=1, alpha=0.6)
    ax1.set_title("Symmetric SINDy (Odd Functions)")
    ax1.set_ylabel(r"f$_1(\dot{x})$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(x_vals, f2_sindy_sym(x_vals), 'r--', label="SINDy-Sym", linewidth=2)
    ax2.plot(x_vals, F2(x_vals), 'k-', label="Theor.", linewidth=1, alpha=0.6)
    ax2.set_ylabel(r"f$_2$(x)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.show()
    
    

 

 

    #########################################################################
    #############  SINDy-CC+sym+post-SR (adding post-processing)    ######### 
    print(" ")
    print("SINDy-CC+sym+post-SR : now doing post-SR to SINDy-CC+sym")
    
    Xdot=xdot_vals.reshape(-1,1)
    Yobjective=f1_sindy_sym(xdot_vals)   
#    Xdot = x_dot_lin_data.reshape(-1, 1)
#    Xdotpred = x_dot_lin.reshape(-1, 1)
#    correction_extra=0.0
    start=time.time()
#    y = predicted_lin_F1_data + bias_correction + correction_extra
    model_f1SR = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*"],
        #unary_operators=["sin", "cos", "exp", "log", "abs", "sqrt","sign"],
        #unary_operators=["abs", "sqrt","sign"],
        loss="loss(x, y) = (x - y)^2",
        maxsize=20,
        populations=20,
        #procs=n_cores,
        #verbosity=1,
    )
    model_f1SR.fit(Xdot, Yobjective)
    print(model_f1SR)
    best_equation = model_f1SR.get_best()
    predicted_F1_sym_SR = model_f1SR.predict(Xdot)
    #pred_f1SR = model_f1SR.predict(Xdotpred)    


    X = x_vals.reshape(-1, 1)
    Yobjective = f2_sindy_sym(x_vals)
    model_f2SR = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*"],
        #unary_operators=["log", "abs", "sqrt","sin", "cos", "exp"],
        loss="loss(x, y) = (x - y)^2",
        maxsize=20,
        populations=20,
        #verbosity=1,
    )
    model_f2SR.fit(X, Yobjective)
    print(model_f2SR)
    best_equation = model_f2SR.get_best()
    predicted_F2_sym_SR = model_f2SR.predict(X)
    end = time.time()  
    elapsed = end - start
    print(r'End doing post-SR to NN-CC+sym . i.e. NN-CC$_{+sym+post\!\!-\!\!SR}$')
    print(f"Training finished in {elapsed:.3f} seconds")

    # plotting obtained CCs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
    ax1.plot(xdot_vals, f1_sindy_sym(xdot_vals), label=r"SINDy-CC$_{+sym}$", linewidth=2)
    ax1.plot(xdot_vals, predicted_F1_sym_SR, label=r"SINDy-CC$_{+sym+post\!\!-\!\!SR}$", linewidth=2)
    ax1.plot(xdot_vals, F1(xdot_vals), '--',color='black', label="Theor.", linewidth=2)
    ax1.set_title(r"Obtained CCs from SINDy-CC$_{+sym+post\!\!-\!\!SR}$ model")
    ax1.set_xlabel(r"$\dot{x}$")
    ax1.set_ylabel(r"f$_1(\dot{x})$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(x_vals, f2_sindy_sym(x_vals), label=r"SINDy-CC$_{+sym}$", linewidth=2)
    ax2.plot(x_vals, predicted_F2_sym_SR, label=r"SINDy-CC$_{+sym+post\!\!-\!\!SR}$", linewidth=2)
    ax2.plot(x_vals, F2(x_vals), '--',color='black', label="Theor.", linewidth=3)
    ax2.set_ylabel(r"f$_2$(x)")
    ax2.set_xlabel("x")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()
 

    # Lambdify f1 and f2 for forward simulations
    expr_f1 = model_f1SR.sympy()
    f1_lambda = sp.lambdify(sp.symbols("x0"), expr_f1, "numpy")
    expr_f1_smooth = expr_f1.replace(sp.sign, lambda arg: sp.tanh(500*arg))
    expr_f1_smooth = expr_f1_smooth.replace(sp.Abs, lambda arg: sp.sqrt(arg**2+1e-6))
    print(model_f1SR.sympy())
    print(expr_f1_smooth)
    f1_lambda = sp.lambdify(sp.symbols("x0"), expr_f1_smooth, "numpy")
    expr_f2 = model_f2SR.sympy()
    f2_lambda = sp.lambdify(sp.symbols("x0"), expr_f2, "numpy")
    def fitted_model_sindy_sym_SR(t, y):
        x_val = y[0]
        x_dot_val = y[1]
        F_ext_val = F_ext(t)
        f1_val = f1_lambda(x_dot_val)
        f2_val = f2_lambda(x_val)
        x_ddot = F_ext_val - f1_val - f2_val
        return [x_dot_val, x_ddot]  
        
#    sol = solve_ivp(SINDy_sym_SR_model, t_span, y0, t_eval=t_simul,method='LSODA')     
#    x_sim = sol.y[0]
#    xdot_sim = sol.y[1]
#    plt.figure(figsize=(8,5))
#    plt.plot(time_data[0:len(x_sim)], x_sim, label=r"SINDy-CC$_{+sym+post\!\!-\!\!SR}$", linewidth=2)
#    plt.plot(time_data, x_data,"--",color='black', label="Training data", linewidth=2)
#    plt.xlabel("t")
#    plt.ylabel("x(t)")
#    plt.legend()
#    plt.title(r"Simulation with SINDy-CC$_{+sym+post\!\!-\!\!SR}$ model")
#    plt.show()        
     
    
    
 
    

    ##########################################
    # Evaluation of the f1 and f2 functions
    ##########################################
    predicted_F1_param = f1_param(xdot_vals).flatten() # parametric-CC
    predicted_F1_theor = F1(xdot_vals).flatten() # theor
    predicted_F1_poly = f1_fit_shifted.flatten() # Poly-CC
    predicted_F1_syndycc = f1_sindy(xdot_vals).flatten() #  SINDy-CC
    predicted_F1_sindy_ku0 = f1_sindy_ku0_func(xdot_vals).flatten()
    predicted_F1_sindy_sym = f1_sindy_sym(xdot_vals).flatten()
    predicted_F1_sindy_sym_SR = model_f1SR.predict(xdot_vals.reshape(-1,1)) # SINDy-CC(+sym+post-SR)
    
    predicted_F2_param = f2_param(x_vals).flatten() # parametric-CC
    predicted_F2_theor = F2(x_vals).flatten() # theor
    predicted_F2_poly = f2_fit_shifted.flatten() # Poly-CC
    predicted_F2_syndycc = f2_sindy(x_vals).flatten() #  SINDy-CC
    predicted_F2_sindy_ku0 = f2_sindy_ku0_func(x_vals).flatten()
    predicted_F2_sindy_sym = f2_sindy_sym(x_vals).flatten()
    predicted_F2_sindy_sym_SR = model_f2SR.predict(x_vals.reshape(-1,1)) # SINDy-CC(+sym+post-SR)

    # 1. Define RMSE function
    def rmse(y_true, y_pred):
        # Flattening ensures shapes like (N,1) and (N,) don't cause broadcasting errors
        return np.sqrt(np.mean((y_true.flatten() - y_pred.flatten()) ** 2))

    # 2. Construct the Dictionary with new variables
    rmse_values = {
        "noise_perc_th": noise_percentage_th,
        "noise_perc": noise_percentage,
        "SNR_dB": SNR_dB,

        # NEW: SINDy Unconstrained (k*u0)
        "SINDy_ku0_f1": rmse(predicted_F1_theor, predicted_F1_sindy_ku0),
        "SINDy_ku0_f2": rmse(predicted_F2_theor, predicted_F2_sindy_ku0),
        
        # SINDy Constrained (1*u0)
        "SINDy_f1": rmse(predicted_F1_theor, predicted_F1_syndycc),
        "SINDy_f2": rmse(predicted_F2_theor, predicted_F2_syndycc),
        
        # SINDy Symmetric Constrained (1*u0)
        "SINDy_Sym_f1": rmse(predicted_F1_theor, predicted_F1_sindy_sym),
        "SINDy_Sym_f2": rmse(predicted_F2_theor, predicted_F2_sindy_sym),

        # SINDy Symmetric Constrained (1*u0) + post-SR
        "SINDy_Sym_f1_SR": rmse(predicted_F1_theor, predicted_F1_sindy_sym_SR),
        "SINDy_Sym_f2_SR": rmse(predicted_F2_theor, predicted_F2_sindy_sym_SR),
                
        # Poly-CC
        "Poly_f1": rmse(predicted_F1_theor, predicted_F1_poly),
        "Poly_f2": rmse(predicted_F2_theor, predicted_F2_poly),
        
        # Parametric-CC
        "Param_f1": rmse(predicted_F1_theor, predicted_F1_param),
        "Param_f2": rmse(predicted_F2_theor, predicted_F2_param),
        
        #"SR_f1": rmse(predicted_F1_theor, predicted_F1_SR),
        #"SR_f2": rmse(predicted_F2_theor, predicted_F2_SR),
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
    
    



 


