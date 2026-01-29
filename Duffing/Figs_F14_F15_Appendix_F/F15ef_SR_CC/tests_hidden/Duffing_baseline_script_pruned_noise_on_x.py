
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
alpha=-1.0
beta=1.0
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

#SNR_dB_list =  list(np.linspace(0, -20, 21 ))  # ∞, 20, 17.5, ..., -5
#SNR_dB_list = list(np.linspace(5, -20, 26 ))  # ∞, 20, 17.5, ..., -5
#SNR_dB_list = [20.0]

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
    if SNR_dB < 30:
        sg_window = 51 
    else:
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
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax[0].set_title(f"Differentiation Assessment (SNR={SNR_dB}dB)")
    
    ax[0].plot(time_data, x_data_clean, 'k-', lw=1, label='True')
    ax[0].plot(time_data, x_noisy, 'r.', ms=2, alpha=0.3, label='Noisy Measure')
    ax[0].plot(time_data, x_data, 'b--', lw=1.5, label='SG Smooth')
    ax[0].set_ylabel("x")
    ax[0].legend()
    
    ax[1].plot(time_data, x_dot_data_clean, 'k-', lw=1, label='True')
    ax[1].plot(time_data, x_dot_data, 'b--', lw=1.5, label='SG Estimate')
    ax[1].set_ylabel("v")
    
    ax[2].plot(time_data, x_ddot_data_clean, 'k-', lw=1, label='True')
    ax[2].plot(time_data, x_ddot_data, 'b--', lw=1.5, label='SG Estimate')
    ax[2].set_ylabel("a")
    ax[2].set_xlabel("Time")
    plt.tight_layout()
#    plt.show()

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
    bmat = x_ddot_data -  F_ext(time_data) #_noise_data #F_ext(t_simul)       # Leftover = f1(x') + f2(x)
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
    plt.tight_layout()
#    plt.show()

    sol = solve_ivp(fitted_model_LS, t_span, y0, t_eval=t_simul,method='LSODA') 
    x_sim = sol.y[0]
    xdot_sim = sol.y[1]
    plt.figure(figsize=(8,5))
    plt.plot(time_data, x_sim, label="Poly-CC", linewidth=2)
    plt.plot(time_data, x_data,"--",color='black', label="Training data", linewidth=2)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.title("Simulation with Poly-CC model")
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
    u_sindy = F_ext(time_data) # np.stack((F_ext_noise_data),axis=1)
    u_sindy_resh = F_ext(time_data).reshape(-1,1) #flatten()
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
            print(f"Skipping trial {i+1}: SINDy simulation returned NaNs or infs.")
    except Exception as e:
        print(f"Skipping trial {i+1}: Exception during SINDy simulation -> {e}")
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
    plt.tight_layout()
#    plt.show()

    sol = solve_ivp(fitted_model_sindy, t_span, y0, t_eval=t_simul,method='LSODA') 
    x_sim = sol.y[0]
    xdot_sim = sol.y[1]
    plt.figure(figsize=(8,5))
    plt.plot(time_data, x_sim, label="SINDy-CC", linewidth=2)
    plt.plot(time_data, x_data,"--",color='black', label="Training data", linewidth=2)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.title("Simulation with SINDy-CC model")
#    plt.show()
    

    ############# SR black box #################
    target_SR=F_ext(time_data)-x_ddot_data 
    X_SR = np.column_stack([x_data, x_dot_data])
    model = PySRRegressor(
        niterations=200,
        binary_operators=["+", "-", "*", "pow"],  #"/"
        #unary_operators=["log", "abs","sign","sin", "cos", "exp"],
        loss="loss(x, y) = (x - y)^2",   # MSE
        populations=10,
        population_size=100,
        maxsize=20,
        progress=True
    )

    print("Start Training SR")
    start = time.time()
    model.fit(X_SR, target_SR, variable_names=["x", "xdot"])
    end = time.time()  
    elapsed = end - start
    print('End Training SR')
    print(f"Training finished in {elapsed:.3f} seconds")


    print(model)
    print(model.get_best())
    y_pred_SR = model.predict(X_SR)
    best_expr = model.sympy()
    print("\nSymbolic expression")
    print("f(x, xdot):", best_expr)
    x_sym, xdot_sym = sp.symbols("x xdot")

    # comment the following for high noise values where SR fails
    #separating f1 and f2 function terms
    expr_sr = sp.simplify(best_expr.expand())    
    f1_terms_sr = []  # only xdot
    f2_terms_sr = []  # only x
    mixed_terms_sr = []  # contain both
    for term in expr_sr.as_ordered_terms():
        free_syms = term.free_symbols
        if free_syms == {xdot_sym}:
            f1_terms_sr.append(term)
        elif free_syms == {x_sym}:
            f2_terms_sr.append(term)
        else:
            mixed_terms_sr.append(term)
            print("mixed terms in SR approach")
            print(mixed_terms_sr)
    f1_expr_sr = sum(f1_terms_sr)
    f2_expr_sr = sum(f2_terms_sr)
    f1_fun_sr = sp.lambdify(xdot_sym, f1_expr_sr, "numpy")
    f2_fun_sr = sp.lambdify(x_sym, f2_expr_sr, "numpy")

    # plotting obtained CCs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
    ax1.plot(xdot_vals, f1_fun_sr(xdot_vals), label="Identified", linewidth=2)
    ax1.plot(xdot_vals, F1(xdot_vals), '--',color='black', label="Theor.", linewidth=2)
    ax1.set_title("Obtained CCs from SR method")
    ax1.set_xlabel(r"$\dot{x}$")
    ax1.set_ylabel(r"f$_1(\dot{x})$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(x_vals, f2_fun_sr(x_vals), label="Identified", linewidth=2)
    ax2.plot(x_vals, F2(x_vals), '--',color='black', label="Theor.", linewidth=3)
    ax2.set_ylabel(r"f$_2$(x)")
    ax2.set_xlabel("x")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
#    plt.show()


    f_SR = sp.lambdify((x_sym, xdot_sym), best_expr, "numpy")
    def ode_sr(t, state):
        x, xdot = state
        xddot = F_ext(t) - f_SR(x, xdot)
        return [xdot, xddot]

    sol = solve_ivp(ode_sr, t_span, y0, t_eval=t_simul,method='LSODA') 
    x_sim = sol.y[0]
    xdot_sim = sol.y[1]
    plt.figure(figsize=(8,5))
    plt.plot(time_data, x_sim, label="SR", linewidth=2)
    plt.plot(time_data, x_data,"--",color='black', label="Training data", linewidth=2)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.title("Simulation with SR model")
#    plt.show()

    
    ##############################################
    #############  NN-CC WITHOUT SYMMETRIES  ######### this next part is for training NNS
    # Redefining Hyperparameters (for testing proposes only)
    #for neurons in [150,20,50,100,200]:
    #Nlearning_rate = 1e-4
    #epochs_max = 20000
    #N_constraint = 1000
    
    # Convert data to tensors
    t_max =  np.max(t_simul)
    t_tensor = torch.tensor(t_simul, dtype=torch.float32).unsqueeze(1).to(device)
    x_tensor = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1).to(device)
    x_dot_tensor = torch.tensor(x_dot_data, dtype=torch.float32).unsqueeze(1).to(device)
    x_ddot_tensor = torch.tensor(x_ddot_data, dtype=torch.float32).unsqueeze(1).to(device)
    F_ext_tensor = torch.tensor(F_ext(time_data), dtype=torch.float32).unsqueeze(1).to(device)
    # define tensors from linear space to then evaluate CCs
    x_vals_tensor = torch.tensor(x_vals, dtype=torch.float32).unsqueeze(1).to('cpu')
    xdot_vals_tensor = torch.tensor(xdot_vals, dtype=torch.float32).unsqueeze(1).to('cpu')

    #x_dot_constraint = torch.linspace(min(x_dot_data), max(x_dot_data), N_constraint).unsqueeze(1).to(device)
    #x_constraint     = torch.linspace(min(x_data),  max(x_data),     N_constraint).unsqueeze(1).to(device)

    # Define the Neural Network architectures
    class NN1(nn.Module):
        def __init__(self):
            super(NN1, self).__init__()
            self.fc1 = nn.Linear(1, neurons)
            self.fc2 = nn.Linear(neurons, neurons)
            self.fc3 = nn.Linear(neurons, neurons)
            self.fc4 = nn.Linear(neurons, 1)
        #    self.fc5 = nn.Linear(neurons, 1)
        def forward(self, x):
            #x = torch.relu(self.fc1(x))
            #x = torch.relu(self.fc2(x))
            #x = torch.nn.functional.leaky_relu(self.fc1(x),0.01)
            #x = torch.nn.functional.leaky_relu(self.fc2(x),0.01)
            #x = torch.nn.functional.leaky_relu(self.fc3(x),0.01)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            #x = torch.relu(self.fc3(x))
            #x = torch.relu(self.fc4(x))
            return self.fc4(x)
    # relu tanh sigmoid(good) rrelu nn.functional.softplus
    # rrelu nn.functional.silu rrelu nn.functional.selu
    # rrelu nn.functional.gelu
    class NN2(nn.Module):
        def __init__(self):
            super(NN2, self).__init__()
            self.fc1 = nn.Linear(1, neurons)
            self.fc2 = nn.Linear(neurons, neurons)
            self.fc3 = nn.Linear(neurons, neurons)
            self.fc4 = nn.Linear(neurons, 1)
            #self.fc4 = nn.Linear(neurons, neurons)
            #self.fc5 = nn.Linear(neurons, 1)
        def forward(self, x):
            #x = torch.nn.functional.leaky_relu(self.fc1(x),0.01)
            #x = torch.nn.functional.leaky_relu(self.fc2(x),0.01)
            #x = torch.nn.functional.leaky_relu(self.fc3(x),0.01)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            #x = torch.relu(self.fc3(x))
            #x = torch.relu(self.fc4(x))
            return self.fc4(x)

    # Instantiate the model
    model1_nosym = NN1().to(device)
    model2_nosym = NN2().to(device)
    criterion = nn.MSELoss()
    optimizer1 = optim.Adam(model1_nosym.parameters(), lr=learning_rate ) 
    # , weight_decay=weight_decay) # working well with lr=1e-4
    optimizer2 = optim.Adam(model2_nosym.parameters(), lr=learning_rate ) 
    #other optimizers (for testing purposes)
    #optimizer1 = optim.AdamW(model1_nosym.parameters(), lr=learning_rate , weight_decay=weight_decay)
    #optimizer2 = optim.AdamW(model2_nosym.parameters(), lr=learning_rate , weight_decay=weight_decay)    
    #optimizer1 = optim.SGD(model1_nosym.parameters(), lr=learning_rate , momentum=momentum) # working well with lr=1e-1
    #optimizer2 = optim.SGD(model2_nosym.parameters(), lr=learning_rate , momentum=momentum)
    
    # Training loop
    zero_input = torch.tensor([[0.0]], dtype=torch.float32).to(device)
    time_start=time.time()
    for epoch in range(epochs_max):
        model1_nosym.train()
        model2_nosym.train()
        predictions = x_ddot_tensor + model1_nosym(x_dot_tensor) + model2_nosym(x_tensor)
        loss = criterion(predictions, F_ext_tensor)
        # Add constraint: model2(0.0) ≈ 0
        restriction_loss=0.0*model2_nosym(zero_input)
        if(apply_restriction):
            model2_at_zero = model2_nosym(zero_input)
            model1_at_zero = model1_nosym(zero_input)
            restriction_loss = lambda_penalty * ((model2_at_zero ** 2).mean() + (model1_at_zero ** 2).mean())  # squared penalty
            #constraint_loss = lambda_penalty * (model2_at_zero ** 2).mean()  # squared penalty
            total_loss = loss + restriction_loss
        else:
            total_loss = loss      
        constraint_loss = restriction_loss
        # Backward pass and optimization
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        total_loss.backward()
        optimizer1.step()
        optimizer2.step()
        # Print the loss
        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}], Loss: {loss.item():.4e}, Constraints: {constraint_loss.item():.4e}")
        if total_loss.item() < error_threshold:
            print(f"Training stopped at epoch {epoch}, Total Loss: {total_loss.item()}")
            break
    time_end=time.time()
    print(" ")
    print("End training baseline NN-CC")
    print("Neurons :",neurons)
    print(f"Training time: {time_end-time_start} seconds")

    # Move to cpu after training 
    model1_nosym=model1_nosym.to('cpu')
    model2_nosym=model2_nosym.to('cpu')
    t_tensor = t_tensor.to('cpu')
    x_tensor = x_tensor.to('cpu')
    x_dot_tensor = x_dot_tensor.to('cpu')
    x_ddot_tensor = x_ddot_tensor.to('cpu')
    F_ext_tensor = F_ext_tensor.to('cpu')

    
    model1_nosym.eval()
    model2_nosym.eval()
    with torch.no_grad():
        predicted_F1_nosym = model1_nosym(xdot_vals_tensor).numpy()
        predicted_F2_nosym = model2_nosym(x_vals_tensor).numpy()
        shift_NN=model2_nosym(zero_input).numpy()
        predicted_F1_nosym_shifted = predicted_F1_nosym+shift_NN
        predicted_F2_nosym_shifted = predicted_F2_nosym-shift_NN
    def NN_nosym_model(t, y):
        x = torch.tensor([[y[0]]], dtype=torch.float32)
        x_dot = torch.tensor([[y[1]]], dtype=torch.float32)
        t_tensor = torch.tensor([[t]], dtype=torch.float32)
        F_ext_tensor = torch.tensor([[F_ext(t)]], dtype=torch.float32)
        model1_nosym.eval()
        model2_nosym.eval()
        with torch.no_grad(): # Neural net-based force computation
            force = F_ext_tensor - model1_nosym(x_dot) - model2_nosym(x)
        x_ddot = force.item()
        return [y[1], x_ddot] 
        
    # plotting obtained CCs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
    ax1.plot(xdot_vals, predicted_F1_nosym_shifted , label="Identified", linewidth=2)
    ax1.plot(xdot_vals, F1(xdot_vals), '--',color='black', label="Theor.", linewidth=2)
    ax1.set_title("Obtained CCs from baseline NN-CC method")
    ax1.set_xlabel(r"$\dot{x}$")
    ax1.set_ylabel(r"f$_1(\dot{x})$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(x_vals, predicted_F2_nosym_shifted , label="Identified", linewidth=2)
    ax2.plot(x_vals, F2(x_vals), '--',color='black', label="Theor.", linewidth=3)
    ax2.set_ylabel(r"f$_2$(x)")
    ax2.set_xlabel("x")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
#    plt.show()

    sol = solve_ivp(NN_nosym_model, t_span, y0, t_eval=t_simul,method='LSODA') 
    x_sim = sol.y[0]
    xdot_sim = sol.y[1]
    plt.figure(figsize=(8,5))
    plt.plot(time_data, x_sim, label="NN-CC", linewidth=2)
    plt.plot(time_data, x_data,"--",color='black', label="Training data", linewidth=2)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.title("Simulation with baseline NN-CC model")
#    plt.show()



    #######################################################
    #############  NN-CC+sym (WITH SYMMETRIES)    ######### 
    #for neurons in [150,20,50,100,200]:
    #neurons=2
    #neurons=100
    # Hyperparameters
    #learning_rate = 1e-4
    #epochs_max = 20000
    #N_constraint = 1000
    
    # Many the lines here are repeated for reusability, 
    # i.e. NN-CC without symmetries block can be commented 
    t_max =  np.max(t_simul)
    t_tensor = torch.tensor(t_simul, dtype=torch.float32).unsqueeze(1).to(device)
    x_tensor = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1).to(device)
    x_dot_tensor = torch.tensor(x_dot_data, dtype=torch.float32).unsqueeze(1).to(device)
    x_ddot_tensor = torch.tensor(x_ddot_data, dtype=torch.float32).unsqueeze(1).to(device)
    F_ext_tensor = torch.tensor(F_ext(time_data), dtype=torch.float32).unsqueeze(1).to(device)
    # define tensors from linear space to then evaluate CCs
    x_vals_tensor = torch.tensor(x_vals, dtype=torch.float32).unsqueeze(1).to('cpu')
    xdot_vals_tensor = torch.tensor(xdot_vals, dtype=torch.float32).unsqueeze(1).to('cpu')

    #x_dot_constraint = torch.linspace(min(x_dot_data), max(x_dot_data), N_constraint).unsqueeze(1).to(device)
    #x_constraint     = torch.linspace(min(x_data),  max(x_data),     N_constraint).unsqueeze(1).to(device)
    x_dot_constraint = torch.linspace(x_dot_data.min(), x_dot_data.max(), N_constraint, device=device).unsqueeze(1)
    x_constraint     = torch.linspace(x_data.min(),     x_data.max(),     N_constraint, device=device).unsqueeze(1)
    
    # Define the Neural Network
    class NN1(nn.Module):
        def __init__(self):
            super(NN1, self).__init__()
            self.fc1 = nn.Linear(1, neurons)
            self.fc2 = nn.Linear(neurons, neurons)
            self.fc3 = nn.Linear(neurons, neurons)
            self.fc4 = nn.Linear(neurons, 1)
        #    self.fc5 = nn.Linear(neurons, 1)
        def forward(self, x):
            #x = torch.relu(self.fc1(x))
            #x = torch.relu(self.fc2(x))
            #x = torch.nn.functional.leaky_relu(self.fc1(x),0.01)
            #x = torch.nn.functional.leaky_relu(self.fc2(x),0.01)
            #x = torch.nn.functional.leaky_relu(self.fc3(x),0.01)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            #x = torch.relu(self.fc3(x))
            #x = torch.relu(self.fc4(x))
            return self.fc4(x)
    # relu tanh sigmoid(good) rrelu nn.functional.softplus
    # rrelu nn.functional.silu rrelu nn.functional.selu
    # rrelu nn.functional.gelu
    class NN2(nn.Module):
        def __init__(self):
            super(NN2, self).__init__()
            self.fc1 = nn.Linear(1, neurons)
            self.fc2 = nn.Linear(neurons, neurons)
            self.fc3 = nn.Linear(neurons, neurons)
            self.fc4 = nn.Linear(neurons, 1)
            #self.fc4 = nn.Linear(neurons, neurons)
            #self.fc5 = nn.Linear(neurons, 1)
        def forward(self, x):
            #x = torch.nn.functional.leaky_relu(self.fc1(x),0.01)
            #x = torch.nn.functional.leaky_relu(self.fc2(x),0.01)
            #x = torch.nn.functional.leaky_relu(self.fc3(x),0.01)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            #x = torch.relu(self.fc3(x))
            #x = torch.relu(self.fc4(x))
            return self.fc4(x)


    # Instantiate the model
    model1 = NN1().to(device)
    model2 = NN2().to(device)
    criterion = nn.MSELoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate )
    optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate )

    #optimizer1 = optim.AdamW(model1.parameters(), lr=learning_rate , weight_decay=weight_decay)
    #optimizer2 = optim.AdamW(model2.parameters(), lr=learning_rate , weight_decay=weight_decay)
    
    #optimizer1 = optim.SGD(model1.parameters(), lr=learning_rate , momentum=momentum) # working well with lr=1e-1
    #optimizer2 = optim.SGD(model2.parameters(), lr=learning_rate , momentum=momentum)

    zero_input = torch.tensor([[0.0]], dtype=torch.float32).to(device)
    time_start=time.time()
    for epoch in range(epochs_max):
        model1.train()
        model2.train()
        predictions = x_ddot_tensor + model1(x_dot_tensor) + model2(x_tensor)
        loss = criterion(predictions, F_ext_tensor)
        # Add constraint: model2(0.0) ≈ 0
        restriction_loss=0.0
        if(apply_restriction):
            #zero_input = torch.tensor([[0.0]], dtype=torch.float32).to(device)
            model2_at_zero = model2(zero_input)
            model1_at_zero = model1(zero_input)
            restriction_loss = lambda_penalty * ((model2_at_zero ** 2).mean() + (model1_at_zero ** 2).mean())  
            total_loss = loss + restriction_loss
        else:
            total_loss = loss
        f1_loss=0.0
        f2_loss=0.0
        if f1_symmetry == 'even':
            f1_loss = lambda_penalty_symm * ((model1(x_dot_constraint) - model1(-x_dot_constraint)) ** 2).mean()
        elif f1_symmetry == 'odd':
            f1_loss = lambda_penalty_symm * ((model1(x_dot_constraint) + model1(-x_dot_constraint)) ** 2).mean()
        total_loss += f1_loss        
        if f2_symmetry == 'even':
            f2_loss = lambda_penalty_symm * ((model2(x_constraint) - model2(-x_constraint)) ** 2).mean()
        elif f2_symmetry == 'odd':
            f2_loss = lambda_penalty_symm * ((model2(x_constraint) + model2(-x_constraint)) ** 2).mean()
        total_loss += f2_loss                
        constraint_loss = restriction_loss + f1_loss + f2_loss
        # Backward pass and optimization
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        total_loss.backward()
        optimizer1.step()
        optimizer2.step()
        # Print the loss
        if epoch == 0 or (epoch + 1) % 100 == 0:
            # uncomment for testing printing each loss term individually
            #print(f"Epoch [{epoch+1}], Loss: {loss.item():.4e}, Constraint: {constraint_loss.item():.4e}, f1_loss: {f1_loss.item():.2e}, f2_loss: {f2_loss.item():.2e}")
            print(f"Epoch [{epoch+1}], Loss: {loss.item():.4e}, Constraints: {constraint_loss.item():.4e}")
        if total_loss.item() < error_threshold:
            print(f"Training stopped at epoch {epoch}, Total Loss: {total_loss.item()}")
            break
    time_end=time.time()
    print(" ")
    print("End training NN-CC+sym")
    print("Neurons :",neurons)
    print(f"Training time: {time_end-time_start} seconds")

    # After training move to cpu
    model1=model1.to('cpu')
    model2=model2.to('cpu')
    t_tensor = t_tensor.to('cpu')
    x_tensor = x_tensor.to('cpu')
    x_dot_tensor = x_dot_tensor.to('cpu')
    x_ddot_tensor = x_ddot_tensor.to('cpu')
    F_ext_tensor = F_ext_tensor.to('cpu')



    model1.eval()
    model2.eval()
    with torch.no_grad():
        predicted_F1_sym = model1(xdot_vals_tensor).numpy()
        predicted_F2_sym = model2(x_vals_tensor).numpy()
        shift_NN_sym=model2(zero_input).numpy()
        predicted_F1_sym_shifted = predicted_F1_sym+shift_NN_sym
        predicted_F2_sym_shifted = predicted_F2_sym-shift_NN_sym
    def NN_sym_model(t, y):
        x = torch.tensor([[y[0]]], dtype=torch.float32)
        x_dot = torch.tensor([[y[1]]], dtype=torch.float32)
        t_tensor = torch.tensor([[t]], dtype=torch.float32)
        F_ext_tensor = torch.tensor([[F_ext(t)]], dtype=torch.float32)
        model1.eval()
        model2.eval()
        with torch.no_grad(): # Neural net-based force computation
            force = F_ext_tensor - model1(x_dot) - model2(x)
        x_ddot = force.item()
        return [y[1], x_ddot] 
        
    # plotting obtained CCs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
    ax1.plot(xdot_vals, predicted_F1_sym_shifted, label="Identified", linewidth=2)
    ax1.plot(xdot_vals, F1(xdot_vals), '--',color='black', label="Theor.", linewidth=2)
    ax1.set_title(r"Obtained CCs from NN-CC$_{+sym}$ method")
    ax1.set_xlabel(r"$\dot{x}$")
    ax1.set_ylabel(r"f$_1(\dot{x})$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(x_vals, predicted_F2_sym_shifted, label="Identified", linewidth=2)
    ax2.plot(x_vals, F2(x_vals), '--',color='black', label="Theor.", linewidth=3)
    ax2.set_ylabel(r"f$_2$(x)")
    ax2.set_xlabel("x")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
#    plt.show()

    sol = solve_ivp(NN_sym_model, t_span, y0, t_eval=t_simul,method='LSODA') 
    x_sim = sol.y[0]
    xdot_sim = sol.y[1]
    plt.figure(figsize=(8,5))
    plt.plot(time_data, x_sim, label=r"NN-CC$_{+sym}$", linewidth=2)
    plt.plot(time_data, x_data,"--",color='black', label="Training data", linewidth=2)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.title(r"Simulation with NN-CC$_{+sym}$ model")
#    plt.show()

    #########################################################################
    #############  NN-CC+sym+post-SR (adding post-processing)    ######### 
    print(" ")
    print("now doing post-SR to NN-CC+sym")
    
    Xdot=xdot_vals.reshape(-1,1)
    Yobjective=predicted_F1_sym_shifted   
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
    Yobjective = predicted_F2_sym_shifted
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
    ax1.plot(xdot_vals, predicted_F1_sym_shifted, label=r"NN-CC$_{+sym}$", linewidth=2)
    ax1.plot(xdot_vals, predicted_F1_sym_SR, label=r"NN-CC$_{+sym+post\!\!-\!\!SR}$", linewidth=2)
    ax1.plot(xdot_vals, F1(xdot_vals), '--',color='black', label="Theor.", linewidth=2)
    ax1.set_title(r"Obtained CCs from NN-CC$_{+sym+post\!\!-\!\!SR}$ model")
    ax1.set_xlabel(r"$\dot{x}$")
    ax1.set_ylabel(r"f$_1(\dot{x})$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(x_vals, predicted_F2_sym_shifted, label=r"NN-CC$_{+sym}$", linewidth=2)
    ax2.plot(x_vals, predicted_F2_sym_SR, label=r"NN-CC$_{+sym+post\!\!-\!\!SR}$", linewidth=2)
    ax2.plot(x_vals, F2(x_vals), '--',color='black', label="Theor.", linewidth=3)
    ax2.set_ylabel(r"f$_2$(x)")
    ax2.set_xlabel("x")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
#    plt.show()


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
    def NN_sym_SR_model(t, y):
        x_val = y[0]
        x_dot_val = y[1]
        F_ext_val = F_ext(t)
        f1_val = f1_lambda(x_dot_val)
        f2_val = f2_lambda(x_val)
        x_ddot = F_ext_val - f1_val - f2_val
        return [x_dot_val, x_ddot]  
        
    sol = solve_ivp(NN_sym_SR_model, t_span, y0, t_eval=t_simul,method='LSODA')     
    x_sim = sol.y[0]
    xdot_sim = sol.y[1]
    plt.figure(figsize=(8,5))
    plt.plot(time_data, x_sim, label=r"NN-CC$_{+sym+post\!\!-\!\!SR}$", linewidth=2)
    plt.plot(time_data, x_data,"--",color='black', label="Training data", linewidth=2)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.title(r"Simulation with NN-CC$_{+sym+post\!\!-\!\!SR}$ model")
#    plt.show()        
 



    ##########################################
    # Evaluation of the f1 and f2 functions
    ##########################################
    predicted_F1_param = f1_param(xdot_vals).flatten() # parametric-CC
    predicted_F1_theor = F1(xdot_vals).flatten() # theor
    predicted_F1_poly = f1_fit_shifted.flatten() # Poly-CC
    predicted_F1_syndycc = f1_sindy(xdot_vals).flatten() #  SINDy-CC
    predicted_F1_SR = f1_fun_sr(xdot_vals).flatten() # SR
    predicted_F1_NNCC = predicted_F1_nosym_shifted.flatten()  # NN-CC
    predicted_F1_NNCC_sym = predicted_F1_sym_shifted.flatten()   # NN-CC(+sym)
    predicted_F1_NNCC_sym_SR = model_f1SR.predict(xdot_vals.reshape(-1,1)) # NN-CC(+sym+post-SR)

    predicted_F2_param = f2_param(x_vals).flatten() # parametric-CC
    predicted_F2_theor = F2(x_vals).flatten() # theor
    predicted_F2_poly = f2_fit_shifted.flatten() # Poly-CC
    predicted_F2_syndycc = f2_sindy(x_vals).flatten() #  SINDy-CC
    predicted_F2_SR = f2_fun_sr(x_vals).flatten() # SR
    predicted_F2_NNCC = predicted_F2_nosym_shifted.flatten()  # NN-CC
    predicted_F2_NNCC_sym = predicted_F2_sym_shifted.flatten()   # NN-CC(+sym)
    predicted_F2_NNCC_sym_SR = model_f2SR.predict(x_vals.reshape(-1,1)) # NN-CC(+sym+post-SR)
    
    

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
        "NN_nosym_f1": rmse(predicted_F1_theor, predicted_F1_NNCC),
        "NN_nosym_f2": rmse(predicted_F2_theor, predicted_F2_NNCC),
        
        "NN_sym_f1": rmse(predicted_F1_theor, predicted_F1_NNCC_sym),
        "NN_sym_f2": rmse(predicted_F2_theor, predicted_F2_NNCC_sym),
        
        "NN_postSR_f1": rmse(predicted_F1_theor, predicted_F1_NNCC_sym_SR),
        "NN_postSR_f2": rmse(predicted_F2_theor, predicted_F2_NNCC_sym_SR),

        # Other Methods
        "SINDy_f1": rmse(predicted_F1_theor, predicted_F1_syndycc),
        "SINDy_f2": rmse(predicted_F2_theor, predicted_F2_syndycc),
        
        "Poly_f1": rmse(predicted_F1_theor, predicted_F1_poly),
        "Poly_f2": rmse(predicted_F2_theor, predicted_F2_poly),
        
        "Param_f1": rmse(predicted_F1_theor, predicted_F1_param),
        "Param_f2": rmse(predicted_F2_theor, predicted_F2_param),
        
        "SR_f1": rmse(predicted_F1_theor, predicted_F1_SR),
        "SR_f2": rmse(predicted_F2_theor, predicted_F2_SR),
    }

    # 3. Print specific console output
    print("-" * 40)
    print(f"NN-CC (No Sym) f1       = {rmse_values['NN_nosym_f1']:.4e}")
    print(f"NN-CC (No Sym) f2       = {rmse_values['NN_nosym_f2']:.4e}")

    print(f"NN-CC (+Sym) f1         = {rmse_values['NN_sym_f1']:.4e}")
    print(f"NN-CC (+Sym) f2         = {rmse_values['NN_sym_f2']:.4e}")

    print(f"NN-CC (+Sym+Post-SR) f1 = {rmse_values['NN_postSR_f1']:.4e}")
    print(f"NN-CC (+Sym+Post-SR) f2 = {rmse_values['NN_postSR_f2']:.4e}")
    print("-" * 40)

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
    
    



#    ################################################################################
#    ################################################################################
#    ################################   VALIDATION   ################################
#    ################################################################################
#    ################################################################################
#
#
#
#    print('Validating the system')
#    print('Integration of model EDOs')
#    n_trials = 20 # number of random initial conditions
#    # 1. Define bounds
#    x_min_train, x_max_train = np.min(x_data), np.max(x_data)
#    v_min_train, v_max_train = np.min(x_dot_data), np.max(x_dot_data)
#    width_train = x_max_train - x_min_train
#    height_train = v_max_train - v_min_train
#
#    # 2. Lists to store ALL points from valid trajectories
#    all_x_points = []
#    all_v_points = []
#    
#    # --- SETUP phase space plot ---
#    fig_phase, ax_phase = plt.subplots(figsize=(6, 6))
#    rect_val = Rectangle((x_min_train, v_min_train), width_train, height_train, 
#                         linewidth=1, linestyle='--', facecolor='gray', alpha=0.3, label='Training data range') #edgecolor='k'
#    #rect_val = Rectangle((x_min_train, v_min_train), width_train, height_train, 
#    #                     linewidth=1.5, edgecolor='#444444', linestyle='--', 
#    #                     facecolor='none', zorder=5, label='Training Domain')
#    ax_phase.add_patch(rect_val)
#    ax_phase.set_xlabel("x",fontsize=24)
#    ax_phase.set_ylabel(r"$\dot{x}$",fontsize=24)
#    #ax_phase.set_title(f"Phase Space: {n_trials} Validation Simulations")
#    ax_phase.grid(True, alpha=0.3)
#    # Choose a colormap (e.g., 'viridis', 'plasma', 'jet', 'tab20','Greys')
#    #colormap = plt.get_cmap('copper')
#    # -------------------------------
#  
#
##    for i in range(n_trials):
## Validation Loop Variables
#    n_trials = 20        # Target number of valid simulations
#    valid_trials = 0     # Counter for valid simulations
#    attempts = 0         # Counter for total attempts (to prevent infinite loops if needed)
#    max_attempts = 1000   # Safety break
#
#    print(f"Starting search for {n_trials} valid simulations inside the training range...")
#
#    while valid_trials < n_trials:
#        attempts += 1
#        if attempts > max_attempts:
#            print("Max attempts reached. Stopping validation.")
#            break
#
#
#        # Random initial conditions uncomment
#        x0_val = np.round(np.random.uniform(-0.5, 0.5),3)
#        v0_val = np.round(np.random.uniform(-0.5, 0.5),3)
#        y0_val = [x0_val, v0_val]
#        Aext = np.round(np.random.uniform(0.1, 0.5),3)
#        Omega = np.round(np.random.uniform(1.1, 1.3),3)
#        
#        #A = np.round(np.random.uniform(1.0, 1.5),3)
#        #Omega = np.round(np.random.uniform(0.2, 0.4),3)
#        #x0_val=x0-0.05
#        #v0_val=v0+0.01
#               
#        #Aext=0.3
#        #alpha=-1.0
#        #beta=1.0
#        #delta=0.3
#        #Omega=1.2
#        #x0=0.5
#        #v0=-0.5
#        #x0_val=x0
#        #y0_val=y0
#        #y0=[x0,v0]
#        #y0_val=y0
#        
#
#        #x0_val=-0.8 #x0
#        #v0_val=0.7 #v0
#        #y0_val = [x0_val,v0_val]
#        #y0_val = y0
#        
#        
#        #kval = np.random.uniform(0.5, 1.)
#        #cval = np.random.uniform(0.1, 0.5)
#        #m = 1.0
#        #mu_N = np.random.uniform(0.5, 1.0)
#        #print(f"Trial {i+1} : x0={x0_val:.4f} ; v0={v0_val:.4f}")
#        #print(r"$\A$="+f"{A:.4f} ; "+r"$\Omega$"+f"={Omega:.4f}")
#        print(f"Trial {attempts} :")
#        #print(f"alpha={alpha}, c={cval}")
#        #print(f"$\Omega$={Omega}, $\mu$*N={mu_N}, $x_0$={x0}, $v_0$={v0}")
#        #print(f"Omega={Omega}, A={A}, $x_0$={x0_val:.4f}, $v_0$={v0_val:.4f}")
#
#        #kval = np.round(np.random.uniform(1, 1.5),3)
#        #cval = np.round(np.random.uniform(0.1, 0.5),3)
#        #m = 1.0
#        #mu_N = np.round(np.random.uniform(0.5, 1.0),3)
#        #Omega = np.round(np.random.uniform(0.2, 0.5),3)
#        #x0 = np.round(np.random.uniform(-0.5, 0.5),3)
#        #v0 = np.round(np.random.uniform(-0.5, 0.5),3)
#
#
#
#        ################ Theoretical Eq ###### validation of the model
#        print("Integrating Theor.")
#        start = time.time()  
#        sol_val = solve_ivp(eq_2nd_ord_veloc, t_span_val, y0_val, t_eval=t_val,method='LSODA')
#        end = time.time()  
#        elapsed = end - start
#        print(f"Solve_ivp finished in {elapsed:.3f} seconds")
#        t_simulated_th = sol_val.t
#        x_simulated_th = sol_val.y[0]       
#        x_dot_simulated_th = sol_val.y[1]   
#        
#        #plt.figure()
#        #plt.plot(t_simulated_th, x_simulated_th, label="Validation: Theor. Integration")
#        #plt.legend()
#        #plt.xlabel("t")
#        #plt.ylabel("x(t)")
#        #plt.show()
#
#        if np.max(x_simulated_th)>np.max(x_data):
#            print("Extrapolation! max(x_sim)>max(x_train) :",np.max(x_simulated_th),">",np.max(x_data))
#        if np.min(x_simulated_th)<np.min(x_data):
#            print("Extrapolation! min(x_sim)<min(x_train) :",np.min(x_simulated_th),"<",np.min(x_data))
#
#        # --- CHECK BOUNDS ---
#        # Check if ANY point in the simulation goes outside the training box
#        is_out_x = np.any(x_simulated_th < x_min_train) or np.any(x_simulated_th > x_max_train)
#        is_out_v = np.any(x_dot_simulated_th < v_min_train) or np.any(x_dot_simulated_th > v_max_train)
#        
#        if is_out_x or is_out_v:
#            # If outside, skip this trial (do not increment valid_trials)
#            print(f"Attempt {attempts}: Out of bounds. Retrying...") 
#            continue 
#        
#        # If we reach here, the curve is strictly inside the box
#        valid_trials += 1
#        print(f"Trial {valid_trials}/{n_trials} found (Attempt {attempts})")
#        
#        # Get a unique color based on the loop index
#        #line_color = colormap(attempts / n_trials)
#       
#        # --- ADD LINE ---
#        # Use a slightly lower alpha and thinner line for cleaner look
#        #ax_phase.plot(x_simulated_th, x_dot_simulated_th, color=line_color, alpha=0.8, linewidth=1.5)
#        
#        # Plot a dot at the initial condition (zorder ensures it's on top of lines)
#        #ax_phase.scatter(x0_val, v0_val, color=line_color, s=35, marker='o', edgecolors='k', zorder=10)
#
#        # contour plot:
#        #ax_phase.plot(x_simulated_th, x_dot_simulated_th, alpha=0.8, linewidth=1.5)
#        #ax_phase.scatter(x0_val, v0_val,  s=35, marker='o', edgecolors='k', zorder=10)
#        #spaghetti plot 
#        ax_phase.plot(x_simulated_th, x_dot_simulated_th, color='k', alpha=0.5, linewidth=1.0)
#
#        # Save data if valid
#        all_x_points.append(x_simulated_th)
#        all_v_points.append(x_dot_simulated_th)
#        
#        ################ parametric-CC ###### validation of the model
#        sol_parametric = solve_ivp(ode_param, t_span_val, y0_val, t_eval=t_val,method='LSODA')
#                        #rtol=1e-9, atol=1e-12, max_step=(t_eval[1] - t_eval[0]))   
#        t_parametric= sol_parametric.t
#        x_simulated_parametric = sol_parametric.y[0]
#        x_dot_simulated_parametric = sol_parametric.y[1]
#        #plt.figure(figsize=(8,5))
#        #plt.plot(t_parametric, x_simulated_parametric, "--", label="Parametric-CC", linewidth=2)
#        #plt.plot(t_simulated_th, x_simulated_th, label="Theor.", linewidth=2)
#        #plt.xlabel("t")
#        #plt.ylabel("x(t)")
#        #plt.legend()
#        #plt.title("Parametric model simulation")
#        #plt.show()
#        
## --- SHOW FINAL PLOT ---
#    # Add a dummy colorbar just to show the range of trials (optional)
#    #sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=1, vmax=n_trials))
#    #cbar = plt.colorbar(sm, ax=ax_phase)
#    #cbar.set_label('Trial Number')
#    
#    #sm = plt.cm.ScalarMappable( norm=plt.Normalize(vmin=1, vmax=n_trials))
#    from matplotlib.lines import Line2D
#    custom_lines = [Line2D([0], [0], color='k', lw=1.5, alpha=0.5),
#                    rect_val]
#    ax_phase.legend(custom_lines, ['Validation trajectories', 'Training data range'], loc='upper center', frameon=True, fontsize=18)
#    ax_phase.grid(True, alpha=0.2)
#    custom_ticks(ax_phase, 1.0, 0.5, 0.5, 0.25)
#    ax_phase.set_xlim(-1.7,1.7)
#    ax_phase.set_ylim(-1.1,1.5)
#    ax_phase.text(0.96, 0.04, "(b)", transform=ax_phase.transAxes, fontsize=24, va='bottom', ha='right')
#    plt.figure(fig_phase.number)
#    plt.tight_layout()
#    plt.savefig("phase_space_20_trajs.pdf")
#    plt.show()
#    
#    
#    
## 4. Concatenate all data
#    #x_flat = np.concatenate(all_x_points)
#    #v_flat = np.concatenate(all_v_points)
#    #fig_phase, ax_phase = plt.subplots(figsize=(8, 6))
#    #from matplotlib.patches import Rectangle
#    #rect_val = Rectangle((x_min_train, v_min_train), width_train, height_train, 
#    #                     linewidth=2, edgecolor='k', linestyle='--', facecolor='none', label='Training Domain')
#    #ax_phase.add_patch(rect_val)
#    #print("Calculating density map...")
#    ## Calculate the point density
#    #xy = np.vstack([x_flat, v_flat])
#    #z = gaussian_kde(xy)(xy)
#    ## Sort the points by density, so that the densest points are plotted last
#    #idx = z.argsort()
#    #x_flat, v_flat, z = x_flat[idx], v_flat[idx], z[idx]
#
#    ## Option A: Contour Plot (Smooth Level Curves)
#    ## Create a grid for contouring
#    #x_grid = np.linspace(x_min_train, x_max_train, 100)
#    #v_grid = np.linspace(v_min_train, v_max_train, 100)
#    #Xgrid, Vgrid = np.meshgrid(x_grid, v_grid)
#    #positions = np.vstack([Xgrid.ravel(), Vgrid.ravel()])
#    #kernel = gaussian_kde(xy)
#    #Zgrid = np.reshape(kernel(positions).T, Xgrid.shape)
#
#    ## Plot filled contours (The "Map")
#    ## cmap='Greys' makes it look like a B&W density map. Use 'viridis' for color.
#    #cf = ax_phase.contourf(Xgrid, Vgrid, Zgrid, levels=15, cmap='Greys')
#    ## Optional: Add contour lines on top for definition
#    #ax_phase.contour(Xgrid, Vgrid, Zgrid, levels=15, colors='k', linewidths=0.5, alpha=0.5)
#    ## Add colorbar for density
#    #cbar = plt.colorbar(cf, ax=ax_phase)
#    #cbar.set_label('Trajectory Density')
#    #ax_phase.set_xlabel("x")
#    #ax_phase.set_ylabel(r"$\dot{x}$")
#    ##ax_phase.set_title(f"Density Map of {valid_trials} Valid Trajectories")
#    #ax_phase.grid(True, alpha=0.3)
#    #ax_phase.set_xlim(-1.7,1.7)
#    #ax_phase.set_ylim(-1.1,1.1)
#    #plt.tight_layout()
#    #plt.show()    
