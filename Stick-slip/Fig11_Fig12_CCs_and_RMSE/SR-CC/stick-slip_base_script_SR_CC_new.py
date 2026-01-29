import matplotlib
matplotlib.use('Agg')
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
time_chaos_x_SR_CC_list=[]
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
rmse_x_SR_CC_list = []
rmse_x_dot_SR_CC_list = []
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
print("EDO: x'' + f1(x') + f2 (x) = F_ext(t)")
print("S1: stick-slip")
print("f1(x')= [c*x'+Ff(x')]/m")
print("f2(x)=[k x]/m")
print("F_ext(t)=F_ext_true(t)/m")

#parameters stick slip
#m=1.0 # kg
#cval=0.386 #0.1 # Ns/m (viscous damping coefficient)
#kval=1.274 # 1.0 # N/m (stiffness)
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


SNR_dB_list = [np.inf] + list(np.linspace(40, -20, 61 ))  # ∞, 20, 17.5, ..., -5
#SNR_dB_list = list(np.linspace(40, -20, 61 ))  # ∞, 20, 17.5, ..., -5
SNR_dB_list = list(np.linspace(-19, -20, 2 ))  # ∞, 20, 17.5, ..., -5

#SNR_dB_list = [np.inf] + list(np.linspace(40, 5, 36 ))  # ∞, 20, 17.5, ..., -5
#SNR_dB_list = list(np.linspace(-18, -20, 3 ))  # ∞, 20, 17.5, ..., -5

#SNR_dB_list = list(np.linspace(5, -20, 26 ))  # ∞, 20, 17.5, ..., -5
#SNR_dB_list = list(np.linspace(-5, -20, 16 ))  # ∞, 20, 17.5, ..., -5

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
    plt.tight_layout()
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
    plt.show()


    
 

    ############# SR black box #################
    target_SR=F_ext_noise_data-x_ddot_data 
    X_SR = np.column_stack([x_data, x_dot_data])
    model = PySRRegressor(
        niterations=200,
        binary_operators=["+", "-", "*", "pow"],  #"/"
        unary_operators=["tanh"],#"log", "abs","sign","sin", "cos", "exp"],
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
    
    SR_crossed_terms = False # Initialize this to False
    
    
    for term in expr_sr.as_ordered_terms():
        free_syms = term.free_symbols
        if free_syms == {xdot_sym}:
            f1_terms_sr.append(term)
        elif free_syms == {x_sym}:
            f2_terms_sr.append(term)
        else:
            mixed_terms_sr.append(term)
            SR_crossed_terms=True
    f1_expr_sr = sum(f1_terms_sr)
    f2_expr_sr = sum(f2_terms_sr)
    # Check for crossed terms and log them
    if mixed_terms_sr:
        warning_msg = (
            f"WARNING: Crossed terms found in Symbolic Regression.\n"
            f"SNR Value: {SNR_dB if 'SNR_value' in locals() else 'N/A'}\n"
            f"Full Equation: {best_expr}\n"
            f"Mixed Terms: {mixed_terms_sr}\n"
            f"{'-'*40}\n"
        )
        
        # Append the warning to a log file
        with open("sr_warnings.log", "a") as f:
            f.write(warning_msg)
            
        print("Warning: Crossed terms detected. Logged to sr_warnings.log")



    if(SR_crossed_terms==False):

        f1_fun_sr = sp.lambdify(xdot_sym, f1_expr_sr, "numpy")
        f2_fun_sr = sp.lambdify(x_sym, f2_expr_sr, "numpy")
        # --- FIX: Helper to handle scalar outputs (when expression is 0) ---
        def evaluate_safe(func, input_arr):
            val = func(input_arr)
            # If result is scalar (e.g. 0), create an array of that value
            if np.isscalar(val) or (isinstance(val, np.ndarray) and val.ndim == 0):
                return np.full_like(input_arr, val)
            return val
        y_sr_f1 = evaluate_safe(f1_fun_sr, xdot_vals)
        y_sr_f2 = evaluate_safe(f2_fun_sr, x_vals)
        # Use the safely calculated y_sr_f1
        ax1.plot(xdot_vals, y_sr_f1, label="Identified", linewidth=2)
        ax1.plot(xdot_vals, F1(xdot_vals), '--', color='black', label="Theor.", linewidth=2)
        ax1.set_title("Obtained CCs from SR method")
        ax1.set_xlabel(r"$\dot{x}$")
        ax1.set_ylabel(r"f$_1(\dot{x})$")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Use the safely calculated y_sr_f2
        ax2.plot(x_vals, y_sr_f2, label="Identified", linewidth=2)
        ax2.plot(x_vals, F2(x_vals), '--', color='black', label="Theor.", linewidth=3)
        ax2.set_ylabel(r"f$_2$(x)")
        ax2.set_xlabel("x")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()        
        
        # plotting obtained CCs
        #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
        #ax1.plot(xdot_vals[0:len(f1_fun_sr(xdot_vals))], f1_fun_sr(xdot_vals), label="Identified", linewidth=2)
        #ax1.plot(xdot_vals, F1(xdot_vals), '--',color='black', label="Theor.", linewidth=2)
        #ax1.set_title("Obtained CCs from SR method")
        #ax1.set_xlabel(r"$\dot{x}$")
       # ax1.set_ylabel(r"f$_1(\dot{x})$")
       # ax1.legend()
        #ax1.grid(True, alpha=0.3)
        #ax2.plot(x_vals, f2_fun_sr(x_vals), label="Identified", linewidth=2)
        #ax2.plot(x_vals, F2(x_vals), '--',color='black', label="Theor.", linewidth=3)
        #ax2.set_ylabel(r"f$_2$(x)")
        #ax2.set_xlabel("x")
        #ax2.legend()
        #ax2.grid(True, alpha=0.3)
        #plt.tight_layout()
        #plt.show()
        
        
        f_SR = sp.lambdify((x_sym, xdot_sym), best_expr, "numpy")
        def ode_sr(t, state):
            x, xdot = state
            xddot = F_ext(t) - f_SR(x, xdot)
            return [xdot, xddot]
        
 
        sol = solve_ivp(ode_sr, t_span, y0, t_eval=t_simul,method='LSODA') 
        x_sim = sol.y[0]
        xdot_sim = sol.y[1]
        plt.figure(figsize=(8,5))
        plt.plot(time_data[0:len(x_sim)], x_sim, label="SR", linewidth=2)
        plt.plot(time_data, x_data,"--",color='black', label="Training data", linewidth=2)
        plt.xlabel("t")
        plt.ylabel("x(t)")
        plt.legend()
        plt.title("Simulation with SR model")
        plt.show()
        
 
    ############################################
    ############# SR-CC (Structured) ###########
    ############################################
    
    # 1. Define the total target (LHS of the equation rearranged)
    # Equation: x'' + f1(x') + f2(x) = F_ext
    # Rearranged: f1(x') + f2(x) = F_ext - x''
    total_target = F_ext_noise_data - x_ddot_data
    
    # 2. Initialization
    n_iterations_cc = 8  # Number of alternating loops (usually 2-3 is enough)
    f1_pred = np.zeros_like(total_target) # Start assuming f1 = 0
    f2_pred = np.zeros_like(total_target)
    
    # Variables to store best expressions
    best_f1_expr = sp.parse_expr("0")
    best_f2_expr = sp.parse_expr("0")
    
    # PySR configuration (slightly reduced iterations for inner loops to save time)
    sr_params = dict(
        niterations=50,  # Fast inner loops
        binary_operators=["+", "-", "*", "pow"],
        unary_operators=["tanh"],
        loss="loss(x, y) = (x - y)^2",
        populations=10,
        population_size=100,
        maxsize=20,
        verbosity=0,      # Reduce noise in output
        progress=False    # Disable progress bar for inner loops
    )

    print("------------------------------------------------")
    print("Starting SR-CC (Alternating Optimization)...")
    
    start_sr_cc = time.time()
    
    for i in range(n_iterations_cc):
        print(f"--- Iteration {i+1}/{n_iterations_cc} ---")
        
        # --- Step A: Optimize f2(x) ---
        # Hypothesis: f2(x) = Total - f1(x')
        target_for_f2 = total_target - f1_pred
        
        model_f2 = PySRRegressor(**sr_params)
        model_f2.fit(x_data.reshape(-1, 1), target_for_f2, variable_names=["x"])
        
        f2_pred = model_f2.predict(x_data.reshape(-1, 1))
        best_f2_expr = model_f2.sympy()
        print(f"  > Found f2(x): {best_f2_expr}")

        # --- Step B: Optimize f1(x') ---
        # Hypothesis: f1(x') = Total - f2(x)
        target_for_f1 = total_target - f2_pred
        
        model_f1 = PySRRegressor(**sr_params)
        model_f1.fit(x_dot_data.reshape(-1, 1), target_for_f1, variable_names=["xdot"])
        
        f1_pred = model_f1.predict(x_dot_data.reshape(-1, 1))
        best_f1_expr = model_f1.sympy()
        print(f"  > Found f1(x'): {best_f1_expr}")

    end_sr_cc = time.time()
    print(f"SR-CC Training finished in {end_sr_cc - start_sr_cc:.3f} seconds")
    print("------------------------------------------------")

    # 3. Final Result Compilation
    print("\nFinal SR-CC Expressions:")
    print(f"f1(xdot) = {best_f1_expr}")
    print(f"f2(x)    = {best_f2_expr}")
    
    # Create lambda functions for plotting/simulation
    x_sym, xdot_sym = sp.symbols("x xdot")
    
    # Handle cases where the model returns a constant (no symbol in expression)
    # We use Sp.lambdify but ensure the variable is passed even if not used
    f1_fun_sr_cc = sp.lambdify(xdot_sym, best_f1_expr, "numpy")
    f2_fun_sr_cc = sp.lambdify(x_sym, best_f2_expr, "numpy")

    # Helper wrapper to handle array inputs safely even if function is constant
    def safe_f1(val):
        res = f1_fun_sr_cc(val)
        return np.full_like(val, res) if np.isscalar(res) else res
        
    def safe_f2(val):
        res = f2_fun_sr_cc(val)
        return np.full_like(val, res) if np.isscalar(res) else res

    # 4. Plotting obtained CCs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    
    # Plot f1 vs xdot
    ax1.plot(xdot_vals, safe_f1(xdot_vals), label="SR-CC Identified", linewidth=2)
    ax1.plot(xdot_vals, F1(xdot_vals), '--', color='black', label="Theor.", linewidth=2)
    ax1.set_title("SR-CC Identified Functions")
    ax1.set_xlabel(r"$\dot{x}$")
    ax1.set_ylabel(r"f$_1(\dot{x})$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot f2 vs x
    ax2.plot(x_vals, safe_f2(x_vals), label="SR-CC Identified", linewidth=2)
    ax2.plot(x_vals, F2(x_vals), '--', color='black', label="Theor.", linewidth=3)
    ax2.set_ylabel(r"f$_2$(x)")
    ax2.set_xlabel("x")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
 
     
 
    plt.tight_layout()
    plt.show()

    # 5. Forward Simulation Validation
    def ode_sr_cc(t, state):
        x, xdot = state
        # Force: x'' = F_ext - f1(x') - f2(x)
        # Note: Depending on symbols used by PySR, evaluate carefully
        term_f1 = float(safe_f1(np.array([xdot]))[0])
        term_f2 = float(safe_f2(np.array([x]))[0])
        xddot = F_ext(t) - term_f1 - term_f2
        return [xdot, xddot]
    
    sol = solve_ivp(ode_sr_cc, t_span, y0, t_eval=t_simul, method='LSODA') 
    x_sim = sol.y[0]
    
 
    plt.figure(figsize=(8,5))
    plt.plot(time_data[0:len(x_sim)], x_sim, label="SR-CC Model", linewidth=2)
    plt.plot(time_data, x_data, "--", color='black', label="Training data", linewidth=2)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.title("Simulation with SR-CC model")
    plt.show()


 
    ##########################################
    # Evaluation of the f1 and f2 functions
    ##########################################
    predicted_F1_param = f1_param(xdot_vals).flatten() # parametric-CC
    predicted_F1_theor = F1(xdot_vals).flatten() # theor
#    predicted_F1_poly = f1_fit_shifted.flatten() # Poly-CC
#    predicted_F1_syndycc = f1_sindy(xdot_vals).flatten() #  SINDy-CC
    if(SR_crossed_terms==False):
        # 1. Compute the raw output
        raw_f1 = f1_fun_sr(xdot_vals)   
        # 2. Check if the result is a scalar (e.g., 0 or 5) instead of an array
        if np.isscalar(raw_f1) or (isinstance(raw_f1, np.ndarray) and raw_f1.ndim == 0):
            # If scalar, fill an array of the correct size with that value
            predicted_F1_SR = np.full_like(xdot_vals, raw_f1).flatten()
        else:
            # If it's already an array, just flatten it
            predicted_F1_SR = raw_f1.flatten()
        #predicted_F1_SR = f1_fun_sr(xdot_vals).flatten() # SR
    else:
        predicted_F1_SR = np.full_like(xdot_vals,1e6).flatten()
#    predicted_F1_NNCC = predicted_F1_nosym_shifted.flatten()  # NN-CC
#    predicted_F1_NNCC_nosym_SR = model_f1SR_nosym.predict(xdot_vals.reshape(-1,1)) # NN-CC(+sym+post-SR)
#    predicted_F1_NNCC_sym = predicted_F1_sym_shifted.flatten()   # NN-CC(+sym)
#    predicted_F1_NNCC_sym_SR = model_f1SR.predict(xdot_vals.reshape(-1,1)) # NN-CC(+sym+post-SR)
    predicted_F1_SR_CC = safe_f1(np.array([xdot_vals])).flatten() # SR-CC
    
    
    predicted_F2_param = f2_param(x_vals).flatten() # parametric-CC
    predicted_F2_theor = F2(x_vals).flatten() # theor
#    predicted_F2_poly = f2_fit_shifted.flatten() # Poly-CC
#    predicted_F2_syndycc = f2_sindy(x_vals).flatten() #  SINDy-CC
    if(SR_crossed_terms==False):
        raw_f2 = f2_fun_sr(x_vals)   
        # 2. Check if the result is a scalar (e.g., 0 or 5) instead of an array
        if np.isscalar(raw_f2) or (isinstance(raw_f2, np.ndarray) and raw_f2.ndim == 0):
            # If scalar, fill an array of the correct size with that value
            predicted_F2_SR = np.full_like(x_vals, raw_f2).flatten()
        else:
            # If it's already an array, just flatten it
            predicted_F2_SR = raw_f2.flatten()
        #predicted_F2_SR = f2_fun_sr(x_vals).flatten() # SR
    else:
        predicted_F2_SR = np.full_like(x_vals,1e6).flatten()
#    predicted_F2_NNCC = predicted_F2_nosym_shifted.flatten()  # NN-CC
#    predicted_F2_NNCC_nosym_SR = model_f2SR_nosym.predict(x_vals.reshape(-1,1)) # NN-CC(+sym+post-SR)
#    predicted_F2_NNCC_sym = predicted_F2_sym_shifted.flatten()   # NN-CC(+sym)
#    predicted_F2_NNCC_sym_SR = model_f2SR.predict(x_vals.reshape(-1,1)) # NN-CC(+sym+post-SR)
    predicted_F2_SR_CC = safe_f2(np.array([x_vals])).flatten() # SR-CC
    
    

    # 1. Define RMSE function
    def rmse(y_true, y_pred):
        # Flattening ensures shapes like (N,1) and (N,) don't cause broadcasting errors
        return np.sqrt(np.mean((y_true.flatten() - y_pred.flatten()) ** 2))

    # 2. Construct the Dictionary with new variables
    rmse_values = {
        "noise_perc_th": noise_percentage_th,
        "noise_perc": noise_percentage,
        "SNR_dB": SNR_dB,
        
    #    # Neural Network Variants
    #    "NN_nosym_f1": rmse(predicted_F1_theor, predicted_F1_NNCC),
    #    "NN_nosym_f2": rmse(predicted_F2_theor, predicted_F2_NNCC),
    #
    #    "NN_postSR_f1_nosym": rmse(predicted_F1_theor, predicted_F1_NNCC_nosym_SR),
    #    "NN_postSR_f2_nosym": rmse(predicted_F2_theor, predicted_F2_NNCC_nosym_SR),
    #    
    #    "NN_sym_f1": rmse(predicted_F1_theor, predicted_F1_NNCC_sym),
    #    "NN_sym_f2": rmse(predicted_F2_theor, predicted_F2_NNCC_sym),
    #    
    #    "NN_postSR_f1": rmse(predicted_F1_theor, predicted_F1_NNCC_sym_SR),
    #    "NN_postSR_f2": rmse(predicted_F2_theor, predicted_F2_NNCC_sym_SR),
    #
    #    # Other Methods
    #    "SINDy_f1": rmse(predicted_F1_theor, predicted_F1_syndycc),
    #    "SINDy_f2": rmse(predicted_F2_theor, predicted_F2_syndycc),
    #    
    #    "Poly_f1": rmse(predicted_F1_theor, predicted_F1_poly),
    #    "Poly_f2": rmse(predicted_F2_theor, predicted_F2_poly),
    #    
        "Param_f1": rmse(predicted_F1_theor, predicted_F1_param),
        "Param_f2": rmse(predicted_F2_theor, predicted_F2_param),
        
        "SR_f1": rmse(predicted_F1_theor, predicted_F1_SR),
        "SR_f2": rmse(predicted_F2_theor, predicted_F2_SR),
        
        "SR_CC_f1": rmse(predicted_F1_theor, predicted_F1_SR_CC),
        "SR_CC_f2": rmse(predicted_F2_theor, predicted_F2_SR_CC),
    }

    # 3. Print specific console output
    #print("-" * 40)
    #print(f"NN-CC (No Sym) f1       = {rmse_values['NN_nosym_f1']:.4e}")
    #print(f"NN-CC (No Sym) f2       = {rmse_values['NN_nosym_f2']:.4e}")
    #
    #print(f"NN-CC (+Post-SR) f1 = {rmse_values['NN_postSR_f1_nosym']:.4e}")
    #print(f"NN-CC (+Post-SR) f2 = {rmse_values['NN_postSR_f2_nosym']:.4e}")
    #
    #print(f"NN-CC (+Sym) f1         = {rmse_values['NN_sym_f1']:.4e}")
    #print(f"NN-CC (+Sym) f2         = {rmse_values['NN_sym_f2']:.4e}")
    #
    #print(f"NN-CC (+Sym+Post-SR) f1 = {rmse_values['NN_postSR_f1']:.4e}")
    #print(f"NN-CC (+Sym+Post-SR) f2 = {rmse_values['NN_postSR_f2']:.4e}")
    #print("-" * 40)
    #
    ## 4. Write to file
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
        


        ### SR has integration issues for high noise values 
        ################ SR ###### validation of the model
        if(SR_crossed_terms==False):
            print("Integrating SR")
            start = time.time()  
            x_simulated_SR=[]
            x_dot_simulated_SR=[]
            try:
                sol_sr = solve_ivp(ode_sr, t_span_val, y0_val, t_eval=t_val , method='LSODA') #,rtol=1e-7,atol=1e-7) #,  method='DOP853', rtol=1e-9, atol=1e-12)
                t_SR = sol_sr.t
                x_simulated_SR = sol_sr.y[0]
                x_dot_simulated_SR = sol_sr.y[1]
                # check for NaNs or infs just in case
                if np.any(np.isnan(x_simulated_SR)) or np.any(np.isinf(x_simulated_SR)):
                    print(f"Skipping trial {valida_trials}: SR simulation returned NaNs or infs.")
            except Exception as e:
                print(f"Skipping trial {valid_trials}: Exception during SR simulation -> {e}")
            end = time.time()  
            elapsed = end - start
            print(f"Solve_ivp finished in {elapsed:.3f} seconds")
##            sol_sr = solve_ivp(ode_sr, t_span_val, [x0_val, v0_val], t_eval=t_val,method='LSODA')
##            # Results
##            t_SR = sol_sr.t
##            x_simulated_SR = sol_sr.y[0]
###            x_dot_simulated_SR = sol_sr.y[1]
#           
            plt.figure()
            plt.plot(t_SR, x_simulated_SR, label="SR", linewidth=2)
            plt.plot(t_simulated_th,x_simulated_th,label="Theor.", linewidth=2)
            plt.legend()
            plt.xlabel("t")
            plt.ylabel("x(t)")
            plt.show()




 

 

         ### SR-CC  
        ################ SR ###### validation of the model
        
            # 5. Forward Simulation Validation
        def ode_sr_cc(t, state):
            x, xdot = state
            # Force: x'' = F_ext - f1(x') - f2(x)
            # Note: Depending on symbols used by PySR, evaluate carefully
            term_f1 = float(safe_f1(np.array([xdot]))[0])
            term_f2 = float(safe_f2(np.array([x]))[0])
            xddot = F_ext(t) - term_f1 - term_f2
            return [xdot, xddot]
    
 
        try:
            sol_sr_cc = solve_ivp(ode_sr_cc, t_span_val, y0_val, t_eval=t_val , method='LSODA') #,rtol=1e-7,atol=1e-7) #,  method='DOP853', rtol=1e-9, atol=1e-12)
            t_SR_CC = sol_sr_cc.t
            x_simulated_SR_CC = sol_sr_cc.y[0]
            x_dot_simulated_SR_CC = sol_sr_cc.y[1]
            # check for NaNs or infs just in case
            if np.any(np.isnan(x_simulated_SR_CC)) or np.any(np.isinf(x_simulated_SR_CC)):
                print(f"Skipping trial {valida_trials}: SR simulation returned NaNs or infs.")
        except Exception as e:
            print(f"Skipping trial {valid_trials}: Exception during SR simulation -> {e}")
        end = time.time()  
        elapsed = end - start
        print(f"Solve_ivp finished in {elapsed:.3f} seconds")
     
        plt.figure(figsize=(8,5))
        plt.plot(t_SR_CC, x_simulated_SR_CC, label="SR-CC Model", linewidth=2)
        plt.plot(t_simulated_th,x_simulated_th,label="Theor.", linewidth=2)
        #plt.plot(time_data, x_data, "--", color='black', label="Training data", linewidth=2)
        plt.xlabel("t")
        plt.ylabel("x(t)")
        plt.legend()
        plt.title("Simulation with SR-CC model")
        plt.show()


 

 
 
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
        plt.show()


 

 


        # CHECK if some integration failed
        if(SR_crossed_terms==False):
            if len(x_simulated_SR) != len(t_val):
                print("Warning:")
                print("NN-CC-SR finished integration before maximum simulation time")
 
                print("time:",t_val[len(x_simulated_SR)-1])

        
        # Plot the simulation results
     #   plt.figure(figsize=(15, 5))
     #   plt.subplot(1, 2, 1)
     #   plt.plot(t_val, x_simulated_th,'-',color='blue', label="$x_{th}$",linewidth='2')
     #   plt.plot(t_val, x_simulated_NN,"--",color='orange', label="$x_{val} NN$",linewidth='3')
     #   plt.plot(t_val[0:len(x_simulated_LS)], x_simulated_LS,"--",color='red', label="$x_{val} LS$",linewidth='3')
     #   plt.plot(t_val[0:len(x_simulated_Sindy)], x_simulated_Sindy,"--",color='violet', label="$x_{val} Sindy-CC$",linewidth='3')
     #   
     #   plt.plot(t_val[0:len(x_simulated_Sindy)], x_simulated_Sindy,"--",color='darkgreen', label="$x_{val} Sindy$",linewidth='3')
     #   plt.ylim(np.min(x_simulated_th)-0.2,np.max(x_simulated_th)+0.2)
     #   plt.xlabel("Time $t$")
     #   plt.ylabel("Position")
     #   plt.title("Validation Test of Position over Time")
     #   plt.legend()
     #   plt.grid(True)
     #   plt.subplot(1, 2, 2)
     #   plt.plot(t_val, x_dot_simulated_th,color='blue', label="$\\dot{x}_{th}$")
     #   plt.plot(t_val, x_dot_simulated_NN,"--",colo:w='orange', label="$\\dot{x}_{val} NN$", linestyle="dashed",linewidth='3')
     #   plt.plot(t_val[0:len(x_dot_simulated_LS)], x_dot_simulated_LS,"--",color='red', label="$\\dot{x}_{val} LS$", linestyle="dashed",linewidth='3')
     #   plt.plot(t_val[0:len(x_dot_simulated_Sindy)], x_dot_simulated_Sindy,"--",color='darkgreen', label="$\\dot{x}_{val} Sindy$",linewidth='3')
#    #    plt.plot(t_val[0:-1], x_val_sindy[:,1],"--",color='darkgreen', label="$\\dot{x}_{val} Sindy$",linewidth='3')
     #   plt.ylim(np.min(x_dot_simulated_th)-0.2,np.max(x_dot_simulated_th)+0.2)
     #   #plt.ylim(-0.75,0.75)
     #   plt.xlabel("Time $t$")
     #   plt.ylabel("Velocity")
     #   plt.title("Neural Network Simulation of Velocity over Time")
     #   plt.legend()
     #   plt.grid(True)
     #   plt.show()



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



     #   fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=False)

     #   # ---- (a) x(t) values ----
     #   axs[0, 0].plot(t_val, x_simulated_th, '-', color='blue', label="Theor.", linewidth=3)
     #   axs[0, 0].plot(t_val, x_simulated_NN, "--", color='orange', label=r"NN-CC$_{+sym}$", linewidth=3)
     #   axs[0, 0].plot(t_val, x_simulated_NN_nosym, "--", color='violet', label="NN-CC nosym", linewidth=3)
     #   axs[0, 0].plot(t_val, x_simulated_NN_SR, "--", color='magenta', label=r"NN-CC$_{+sym+post\!\!-\!\!SR}$", linewidth=3)
     #   axs[0, 0].plot(t_val[:len(x_simulated_LS)], x_simulated_LS, "--", color='red', label="Poly-CC", linewidth=3)
     #   axs[0, 0].plot(t_val[0:len(x_simulated_Sindy)], x_simulated_Sindy, "--", color='darkgreen', label="SINDy-CC", linewidth=3)
     #   axs[0, 0].set_ylim(np.min(x_simulated_LS)-0.2, np.max(x_simulated_LS)+0.2)
     #   axs[0, 0].set_ylabel("$x$", fontsize=24)
     #   axs[0, 0].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1))
     #   custom_ticks(axs[0, 0], 20, 0.5 , 5 , 0.25)
     #   axs[0, 0].text(0.98, 0.04, "(a)", transform=axs[0, 0].transAxes, fontsize=24, va='bottom', ha='right')

     #   # ---- (b) x_dot(t) values ----
     #   axs[0, 1].plot(t_val, x_dot_simulated_th, '-', color='blue', label="Theor.", linewidth=3)
     #   axs[0, 1].plot(t_val, x_dot_simulated_NN, "--", color='orange', label=r"NN-CC$_{+sym}$", linewidth=3)
     #   axs[0, 1].plot(t_val, x_dot_simulated_NN_nosym, "--", color='violet', label="NN-CC nosym", linewidth=3)
     #   axs[0, 1].plot(t_val, x_dot_simulated_NN_SR, "--", color='magenta', label=r"NN-CC$_{+sym+post\!\!-\!\!SR}$", linewidth=3)
     #   axs[0, 1].plot(t_val[:len(x_dot_simulated_LS)], x_dot_simulated_LS, "--", color='red', label="Poly-CC", linewidth=3)
     #   axs[0, 1].plot(t_val[0:len(x_dot_simulated_Sindy)], x_dot_simulated_Sindy, "--", color='darkgreen', label="SINDy-CC", linewidth=3)
     #   axs[0, 1].set_ylim(np.min(x_dot_simulated_LS)-0.2, np.max(x_dot_simulated_LS)+0.2)
     #   axs[0, 1].set_ylabel("$\dot{x}$", fontsize=24)
     #   axs[0, 1].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1))
     #   custom_ticks(axs[0, 1], 20, 0.5, 5, 0.25)
     #   axs[0, 1].text(0.98, 0.04, "(b)", transform=axs[0, 1].transAxes, fontsize=24, va='bottom', ha='right')

     #   # ---- (c) Residuals: x_model - x_theor ----
     #   axs[1, 0].plot(t_val, x_simulated_NN - x_simulated_th, '--', color='orange', label=r"NN-CC$_{+sym}$", linewidth=3)
     #   axs[1, 0].plot(t_val[:len(x_simulated_LS)], x_simulated_LS - x_simulated_th[:len(x_simulated_LS)], '--', color='red', label="Poly-CC", linewidth=3)
     #   axs[1, 0].plot(t_val[0:len(x_simulated_Sindy)], x_simulated_Sindy - x_simulated_th[0:len(x_simulated_Sindy)], '--', color='darkgreen', label="SINDy-CC", linewidth=3)
     #   axs[1, 0].axhline(0, color='black', linewidth=1)
     #   axs[1, 0].set_ylabel("$x-x_{th.}$", fontsize=22)
     #   axs[1, 0].set_xlabel("$t$", fontsize=24)
     #   axs[1, 0].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1))
     #   custom_ticks(axs[1, 0], 20, 0.2, 5, 0.1)
     #   axs[1, 0].text(0.9, 0.04, "(c)", transform=axs[1, 0].transAxes, fontsize=24, va='bottom', ha='right')

     #   # ---- (d) Residuals: x_dot_model - x_dot_theor ----
     #   axs[1, 1].plot(t_val, x_dot_simulated_NN - x_dot_simulated_th, '--', color='orange', label=r"NN-CC$_{+sym}$", linewidth=3)
     #   axs[1, 1].plot(t_val[:len(x_dot_simulated_LS)], x_dot_simulated_LS - x_dot_simulated_th[:len(x_dot_simulated_LS)], '--', color='red', label="Poly-CC", linewidth=3)
     #   axs[1, 1].plot(t_val[0:len(x_dot_simulated_Sindy)], x_dot_simulated_Sindy - x_dot_simulated_th[0:len(x_simulated_Sindy)], '--', color='darkgreen', label="SINDy-CC", linewidth=3)
     #   axs[1, 1].axhline(0, color='black', linewidth=1)
     #   axs[1, 1].set_ylabel("$\dot{x}-\dot{x}_{th.}$", fontsize=22)
     #   axs[1, 1].set_xlabel("$t$", fontsize=24)
     #   axs[1, 1].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1))
     #   custom_ticks(axs[1, 1], 20, 0.1, 5, 0.05)
     #   axs[1, 1].text(0.9, 0.04, "(d)", transform=axs[1, 1].transAxes, fontsize=24, va='bottom', ha='right')

     #   plt.tight_layout()

     #   # Save
     #   folder_path = output_path
     #   # "/content/drive/My Drive/Colab Notebooks/Plots"
     #   os.makedirs(folder_path, exist_ok=True)
     #   file_name = "valid_duffing.pdf"
     #   file_path = os.path.join(folder_path, file_name)
     #   plt.savefig(file_path, format='pdf', bbox_inches='tight')
#    #   plt.show()
     #   print(f"Saved to: {file_path}")




        # Calculation of absolute value separation
        threshold_chaos = 0.2 # 4
        
 
        if(SR_crossed_terms==False):
#            time_chaos_x_SR_list.append(t_val[-1])
            time_chaos_x_SR=t_val[len(x_simulated_SR)-1]
            for i in range(len(x_simulated_SR)):
                diff = abs(x_simulated_SR[i] - x_simulated_th[i])
                if diff > threshold_chaos:
#                    time_chaos_x_SR_list.append(t_val[i])
                    time_chaos_x_SR=t_val[i]
                    break
            time_chaos_x_SR_list.append(time_chaos_x_SR)
#            time_chaos_x_parametric_list.append(t_val[-1])
        else:
            time_chaos_x_SR=0.0
            time_chaos_x_SR_list.append(0.0)
        time_chaos_x_parametric=t_val[-1]
        for i in range(len(x_simulated_parametric)):
            diff = abs(x_simulated_parametric[i] - x_simulated_th[i])
            if diff > threshold_chaos:
#                time_chaos_x_parametric_list.append(t_val[i])
                time_chaos_x_parametric=t_val[i]
                break
        time_chaos_x_parametric_list.append(time_chaos_x_parametric)
        #time_chaos_x_NN_list.append(t_val[-1])

        time_chaos_x_SR_CC=t_val[len(x_simulated_SR_CC)-1]
        for i in range(len(x_simulated_SR_CC)):
            diff = abs(x_simulated_SR_CC[i] - x_simulated_th[i])
            if diff > threshold_chaos:
                #time_chaos_x_SR_CC_list.append(t_val[i])
                time_chaos_x_SR_CC =t_val[i]
                break
        time_chaos_x_SR_CC_list.append(time_chaos_x_SR_CC)
        #time_chaos_x_NN_nosym_list.append(t_val[-1])

     #   time_chaos_x_NN_nosym=t_val[len(x_simulated_NN_nosym)-1]
     #   for i in range(len(x_simulated_NN_nosym)-1):
     #       diff = abs(x_simulated_NN_nosym[i] - x_simulated_th[i])
     #       if diff > threshold_chaos:
     #           #time_chaos_x_NN_nosym_list.append(t_val[i])
     #           time_chaos_x_NN_nosym=t_val[i]
     #           break
     #   time_chaos_x_NN_nosym_list.append(time_chaos_x_NN_nosym)
     #   #time_chaos_x_NN_SR_list.append(t_val[-1])

     #   time_chaos_x_NN_SR_nosym=t_val[len(x_simulated_NN_SR_nosym)-1]
     #   for i in range(len(x_simulated_NN_SR_nosym)):
     #       diff = abs(x_simulated_NN_SR_nosym[i] - x_simulated_th[i])
     #       if diff > threshold_chaos:
     #           #time_chaos_x_NN_SR_list.append(t_val[i])
     #           time_chaos_x_NN_SR_nosym=t_val[i]
     #           break
     #   time_chaos_x_NN_SR_nosym_list.append(time_chaos_x_NN_SR_nosym)
     #   #time_chaos_x_LS_list.append(t_val[-1])

     #   time_chaos_x_NN_SR=t_val[len(x_simulated_NN_SR)-1]
     #   for i in range(len(x_simulated_NN_SR)):
     #       diff = abs(x_simulated_NN_SR[i] - x_simulated_th[i])
     #       if diff > threshold_chaos:
     #           #time_chaos_x_NN_SR_list.append(t_val[i])
     #           time_chaos_x_NN_SR=t_val[i]
     #           break
     #   time_chaos_x_NN_SR_list.append(time_chaos_x_NN_SR)
     #   #time_chaos_x_LS_list.append(t_val[-1])

     #   time_chaos_x_LS=t_val[len(x_simulated_LS)-1]
     #   for i in range(len(x_simulated_LS)):
     #       diff = abs(x_simulated_LS[i] - x_simulated_th[i])
     #       if diff > threshold_chaos:
     #           #time_chaos_x_LS_list.append(t_val[i])
     #           time_chaos_x_LS=t_val[i]
     #           break
     #   time_chaos_x_LS_list.append(time_chaos_x_LS)
     #   #time_chaos_x_Sindy_list.append(t_val[-1])

     #   time_chaos_x_Sindy=t_val[len(x_simulated_Sindy)-1]
     #   for i in range(len(x_simulated_Sindy)):
     #       diff = abs(x_simulated_Sindy[i] - x_simulated_th[i])
     #       if diff > threshold_chaos:
     #           #time_chaos_x_Sindy_list.append(t_val[i])
     #           time_chaos_x_Sindy=t_val[i]
     #           break
     #   time_chaos_x_Sindy_list.append(time_chaos_x_Sindy)
     #   #time_chaos_x_Sindy_ku0_list.append(t_val[-1])
     #   
     #   time_chaos_x_Sindy_ku0=t_val[-1]
     #   for i in range(len(x_simulated_Sindy_ku0)):
     #       diff = abs(x_simulated_Sindy_ku0[i] - x_simulated_th[i])
     #       if diff > threshold_chaos:
     #           #time_chaos_x_Sindy_ku0_list.append(t_val[i])
     #           time_chaos_x_Sindy_ku0=t_val[i]
     #           break
     #   time_chaos_x_Sindy_ku0_list.append(time_chaos_x_Sindy_ku0)

        time_matrix_append = np.column_stack([
            noise_percentage_th,
            noise_percentage,
            SNR_dB,
         #   time_chaos_x_NN_nosym,
         #   time_chaos_x_NN_SR_nosym,
         #   time_chaos_x_NN,
         #   time_chaos_x_NN_SR,
         #   time_chaos_x_Sindy,
         #   time_chaos_x_Sindy_ku0,
         #   time_chaos_x_LS,
            time_chaos_x_parametric,
            time_chaos_x_SR,
            time_chaos_x_SR_CC,
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
                       header="nois_th nois nois_db  Parametric SR   SR_CC" if not file_exists else '',
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

        
        
        if(SR_crossed_terms==False):
            rmse_x_SR = np.sqrt(np.mean((x_simulated_th[0:len(x_simulated_SR)] - x_simulated_SR) ** 2))
            rmse_x_dot_SR = np.sqrt(np.mean((x_dot_simulated_th[0:len(x_dot_simulated_SR)] - x_dot_simulated_SR) ** 2))
            print(f"SR results\t - x: {rmse_x_SR:.6f}, x': {rmse_x_dot_SR:.6f}")
            rmse_x_SR_list.append(rmse_x_SR)
            rmse_x_dot_SR_list.append(rmse_x_dot_SR)
        else:
            rmse_x_SR=1e6
            rmse_x_dot_SR=1e6
            rmse_x_SR_list.append(rmse_x_SR)
            rmse_x_dot_SR_list.append(rmse_x_dot_SR)

        rmse_x_SR_CC = np.sqrt(np.mean((x_simulated_th[0:len(x_simulated_SR_CC)] - x_simulated_SR_CC) ** 2))
        rmse_x_dot_SR_CC = np.sqrt(np.mean((x_dot_simulated_th[0:len(x_dot_simulated_SR_CC)] - x_dot_simulated_SR_CC) ** 2))
        print(f"SR-CC results\t - x: {rmse_x_SR_CC:.6f}, x': {rmse_x_dot_SR_CC:.6f}")
        rmse_x_SR_CC_list.append(rmse_x_SR_CC)
        rmse_x_dot_SR_CC_list.append(rmse_x_dot_SR_CC)

        rmse_matrix_append = np.column_stack([
            noise_percentage_th,
            noise_percentage,
            SNR_dB,
            rmse_x_SR,
            rmse_x_dot_SR,
            rmse_x_SR_CC,
            rmse_x_dot_SR_CC,
            rmse_x_parametric,
            rmse_x_dot_parametric,
        ])
        folder_path = output_path
        os.makedirs(folder_path, exist_ok=True)
        file_name = "rmse_noise_duffing_SR_CC.txt"
        file_path = os.path.join(folder_path, file_name)
        # Save with header and space as delimiter
        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a') as f:
            np.savetxt(f, rmse_matrix_append,
                       header="noise_th noise noise_db xSR vSR xSR-CC vSR-CC xparam vparam" if not file_exists else '',
                       fmt="%.8f", delimiter=" ", comments='')


    # Compute overall RMSE and standard deviation
    total_rmse_x_SR = np.mean(rmse_x_SR_list)
    std_rmse_x_SR = np.std(rmse_x_SR_list)
    total_rmse_x_dot_SR = np.mean(rmse_x_dot_SR_list)
    std_rmse_x_dot_SR = np.std(rmse_x_dot_SR_list)
    total_rmse_x_SR_CC = np.mean(rmse_x_SR_CC_list)
    std_rmse_x_SR_CC = np.std(rmse_x_SR_CC_list)
    total_rmse_x_dot_SR_CC = np.mean(rmse_x_dot_SR_CC_list)
    std_rmse_x_dot_SR_CC = np.std(rmse_x_dot_SR_CC_list)

    # Print results
    print("\n======= Total RMSE over all trials (mean ± std, % std) =======")
    print("SR results")
    print(f"Position (x):     {total_rmse_x_SR:.6f} ± {std_rmse_x_SR:.6f}  ({std_rmse_x_SR/total_rmse_x_SR*100:.6f}%)")
    print(f"Velocity (x'):    {total_rmse_x_dot_SR:.6f} ± {std_rmse_x_dot_SR:.6f}  ({std_rmse_x_dot_SR/total_rmse_x_dot_SR*100:.6f}%)")
    print("SR-CC results")
    print(f"Position (x):     {total_rmse_x_SR_CC:.6f} ± {std_rmse_x_SR_CC:.6f}  ({std_rmse_x_SR_CC/total_rmse_x_SR_CC*100:.6f}%)")
    print(f"Velocity (x'):    {total_rmse_x_dot_SR_CC:.6f} ± {std_rmse_x_dot_SR_CC:.6f}  ({std_rmse_x_dot_SR_CC/total_rmse_x_dot_SR_CC*100:.6f}%)")
    

    # These lists must already be defined
    # Each one should contain RMSE values over multiple trials
    # e.g., rmse_x_NN_list = [rmse_trial1, rmse_trial2, ..., rmse_trialN]
    # same for other methods
    rmse_data = [
        rmse_x_SR_list,
        rmse_x_SR_CC_list
    ]
    labels = ['SR','SR-CC']
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
    plt.tight_layout()
    plt.show()
    rmse_dot_data = [
        rmse_x_dot_SR_list,
        rmse_x_dot_SR_CC_list
    ]
    labels = ['SR','SR-CC']
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
    plt.tight_layout()
    plt.show()
    # Stack the RMSE arrays column-wise
    rmse_matrix = np.column_stack([
        rmse_x_SR_list,
        rmse_x_dot_SR_list,
        rmse_x_SR_CC_list,
        rmse_x_dot_SR_CC_list,
    ])
    folder_path = output_path
    os.makedirs(folder_path, exist_ok=True)
    file_name = "rmse_results_stick_slip_total_err.txt"
    file_path = os.path.join(folder_path, file_name)
    # Save with header and space as delimiter
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a') as f:
        np.savetxt(f, rmse_matrix,
                   header="noise_th noise_meas noise_db er_x_SR er_x_dot_SR er_x_SR_CC er_x_dot_SR_CC" if not file_exists else '',
                   fmt="%.8f", delimiter=" ", comments='')
    #np.savetxt(file_path, rmse_matrix,
    #           header="rmse_x_NN rmse_x_dot_NN rmse_x_Sindy rmse_x_dot_Sindy rmse_x_LS rmse_x_dot_LS",
    #           fmt="%.8f", delimiter=" ")



#test_predictions = model(test_inputs).cpu().numpy()  # Move predictions to CPU for plotting



