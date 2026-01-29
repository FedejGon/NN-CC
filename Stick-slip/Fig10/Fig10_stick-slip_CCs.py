
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
import os
import copy
import time
  
output_path = "./"
output_file_log = open("output_log.txt", "w")

from pysr import PySRRegressor
import sympy as sp


def custom_ticks(ax, major_x_interval, major_y_interval, minor_x_interval, minor_y_interval):
    # Set major ticks
    ax.xaxis.set_major_locator(ticker.MultipleLocator(major_x_interval))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(major_y_interval))
    # Set minor ticks
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_x_interval))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(minor_y_interval))
    # Customize tick appearance
    ax.tick_params(axis='x', direction='in', which='major', length=8, width=1.5, labelsize=16, top=True, bottom=True)
    ax.tick_params(axis='y', direction='in', which='major', length=8, width=1.5, labelsize=16, left=True, right=True)
    ax.tick_params(axis='x', direction='in', which='minor', length=5, width=1, top=True, bottom=True)
    ax.tick_params(axis='y', direction='in', which='minor', length=5, width=1, left=True, right=True)

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
time_chaos_x_Sindy_list =[]
time_chaos_x_LS_list =[]
time_chaos_x_Sindy_ku0_list=[]

rmse_x_SR_list = []
rmse_x_dot_SR_list = []
rmse_x_parametric_list = []
rmse_x_dot_parametric_list = []
rmse_x_NN_nosym_list = []
rmse_x_dot_NN_nosym_list = []
rmse_x_NN_list = []
rmse_x_dot_NN_list = []
 
rmse_x_NN_SR_list = []
rmse_x_dot_NN_SR_list = []
#rmse_x_Sindy_list = []
#rmse_x_dot_Sindy_list = []
#rmse_x_LS_list = []
#rmse_x_dot_LS_list = []
#rmse_x_Sindy_k0_list = []
#rmse_x_dot_Sindy_k0_list = []    
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


#SNR_dB_list = [np.inf] + list(np.linspace(40, -20, 61 ))  # ∞, 20, 17.5, ..., -5
SNR_dB_list = [np.inf] + list(np.linspace(40, 5, 36 ))  # ∞, 20, 17.5, ..., -5
#SNR_dB_list = list(np.linspace(-18, -20, 3 ))  # ∞, 20, 17.5, ..., -5

SNR_dB_list = list(np.linspace(5, -20, 26 ))  # ∞, 20, 17.5, ..., -5

SNR_dB_list = [20.0]

#repeat 3 times each value in the list
SNR_dB_list = np.repeat(SNR_dB_list, 1)


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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
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

    
    # --- Plotting Training dataset ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True) # Increased height for better visibility
    ax1.plot(time_data, x_data, color='black')
    ax1.set_ylabel("x(t)  (m)", fontsize=18) # Increased fontsize to match tick labels
    ax2.plot(time_data, F_ext_noise_data, color='black', linestyle='-')
    ax2.set_xlabel("t  (s)", fontsize=18)
    ax2.set_ylabel(r"F$_{ext}$(t)  (N)", fontsize=18)
    # Apply custom ticks (Adjust intervals based on your data range)
    custom_ticks(ax1, major_x_interval=10, major_y_interval=1, minor_x_interval=5, minor_y_interval=0.5)
    custom_ticks(ax2, major_x_interval=10, major_y_interval=2, minor_x_interval=5, minor_y_interval=1)
    ax1.text(0.98, 0.86, '(a)', transform=ax1.transAxes, fontsize=18, va='bottom', ha='right')
    ax2.text(0.98, 0.86, '(b)', transform=ax2.transAxes, fontsize=18, va='bottom', ha='right')
    plt.tight_layout()
    plt.savefig('Fig10ab.pdf')
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
    plt.plot(time_data, x_sim, label="Parametric", linewidth=2)
    plt.plot(time_data, x_data, "--", color='black', label="Training data", linewidth=2)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.title("Simulation with Parametric-CC model")
    plt.show()


    
 
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
    F_ext_tensor = torch.tensor(F_ext_noise_data, dtype=torch.float32).unsqueeze(1).to(device)
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
            restriction_loss = lambda_penalty * ((model2_at_zero ** 2).mean() )#+ (model1_at_zero ** 2).mean())  # squared penalty
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
    zero_input = zero_input.to('cpu')
    
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
    plt.show()

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
    plt.show()



 
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
    F_ext_tensor = torch.tensor(F_ext_noise_data, dtype=torch.float32).unsqueeze(1).to(device)
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
    plt.show()

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
    plt.show()

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
        unary_operators=["tanh"],# "sqrt","sign"],
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
        unary_operators=["tanh"],#
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
    plt.show()


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
    plt.show()        
 




    ##########################################
    # Evaluation of the f1 and f2 functions
    ##########################################
    predicted_F1_param = f1_param(xdot_vals).flatten() # parametric-CC
    predicted_F1_theor = F1(xdot_vals).flatten() # theor
 
    predicted_F1_NNCC = predicted_F1_nosym_shifted.flatten()  # NN-CC
 
    predicted_F1_NNCC_sym = predicted_F1_sym_shifted.flatten()   # NN-CC(+sym)
    predicted_F1_NNCC_sym_SR = model_f1SR.predict(xdot_vals.reshape(-1,1)) # NN-CC(+sym+post-SR)

    predicted_F2_param = f2_param(x_vals).flatten() # parametric-CC
    predicted_F2_theor = F2(x_vals).flatten() # theor
 
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
 
        "Param_f1": rmse(predicted_F1_theor, predicted_F1_param),
        "Param_f2": rmse(predicted_F2_theor, predicted_F2_param),
 
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
    
    
    ############ plotting CCs Fig 3 (c) and (d)
    
    # Create extrapolated x values
    n_window = 1000 # number of points to calculate the envelope around each edge
    #range_interp = 0.2 # percentage of values near edges for obtaining the envelope
    range_extrap = 0.5  # percentage of extrapolated range with respect to total range
    #n_extra = 50 # number of new data points for each direction
    #deg_extrap = 1 # degree of polynomial extrapolation

    x_extrap_min=min(x_data)-range_extrap*(max(x_data)-min(x_data))    
    x_extrap_max=max(x_data)+range_extrap*(max(x_data)-min(x_data))
    x_extrap= np.linspace(x_extrap_min,x_extrap_max,n_window)
    x_extrap_tensor=torch.tensor(x_extrap, dtype=torch.float32).unsqueeze(1)
    
    x_dot_extrap_min=min(x_dot_data)-range_extrap*(max(x_dot_data)-min(x_dot_data))    
    x_dot_extrap_max=max(x_dot_data)+range_extrap*(max(x_dot_data)-min(x_dot_data))
    x_dot_extrap= np.linspace(x_dot_extrap_min,x_dot_extrap_max,n_window)
    x_dot_extrap_tensor=torch.tensor(x_dot_extrap, dtype=torch.float32).unsqueeze(1)
    
    model1.eval()
    model1_nosym.eval()
    model2.eval()
    model2_nosym.eval()
    with torch.no_grad():
        f1_nosym = (model1_nosym(x_dot_extrap_tensor).squeeze().numpy()+shift_NN).flatten()
        f2_nosym = (model2_nosym(x_extrap_tensor).squeeze().numpy()-shift_NN).flatten()
        f1_sym   = model1(x_dot_extrap_tensor).squeeze().numpy()
        f2_sym   = model2(x_extrap_tensor).squeeze().numpy()
    

    def custom_ticks(ax, major_x_interval, major_y_interval, minor_x_interval, minor_y_interval):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(major_x_interval))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(major_y_interval))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_x_interval))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(minor_y_interval))
        ax.tick_params(axis='x', direction='in', which='major', length=8, width=1.5, labelsize=24, top=True, bottom=True)
        ax.tick_params(axis='y', direction='in', which='major', length=8, width=1.5, labelsize=24, left=True, right=True)
        ax.tick_params(axis='x', direction='in', which='minor', length=5, width=1, top=True,bottom=True)
        ax.tick_params(axis='y', direction='in', which='minor', length=5, width=1, left=True, right=True)


    fig, axs = plt.subplots(2, 1, figsize=(5, 8), sharex=False)
    axs[0].axvspan(np.min(x_dot_data), np.max(x_dot_data), color='lightgray', alpha=0.6, zorder=0.5)
    axs[0].plot(x_dot_extrap, (f1_lambda(x_dot_extrap)).flatten(), "-", color='blue', label=r"NN-CC$_\text{+sym+post-SR}$", lw=2.5)
    axs[0].plot(x_dot_extrap, f1_sym, "-", color='teal', label=r"NN-CC$_\text{+sym}$", lw=2)
    axs[0].plot(x_dot_extrap, f1_nosym, "-", color='magenta', label="NN-CC", lw=2)
    axs[0].plot(x_dot_extrap, F1(x_dot_extrap), "--", color='black',dashes=(4, 6), label="Theor.", lw=2)
    axs[0].set_xlabel("$\\dot{x}$  (m/s)", fontsize=24)
    axs[0].set_ylabel("$f_1(\\dot{x})$  (N)", fontsize=24)
    F1_th_range = np.max(F1_th) - np.min(F1_th)
    axs[0].set_ylim(np.min(F1_th) - F1_th_range * 0.5, np.max(F1_th) + F1_th_range * 0.5)
    axs[0].legend(fontsize=15, loc='upper left', bbox_to_anchor=(0, 1),labelspacing=0.05,frameon=False)
    custom_ticks(axs[0], major_x_interval=1, major_y_interval=2, minor_x_interval=0.5, minor_y_interval=1)
    #custom_ticks(axs[0], major_x_interval=1, major_y_interval=0.4, minor_x_interval=0.5, minor_y_interval=0.2)
    axs[0].text(0.96, 0.15, "(c)", transform=axs[0].transAxes, fontsize=20, va='top', ha='right')
    # Second subplot: F2
    axs[1].axvspan(np.min(x_data), np.max(x_data), color='lightgray', alpha=0.6, zorder=0)
    
    
    axs[1].plot(x_extrap, (f2_lambda(x_extrap)).flatten(), "-", color='blue', label=r'NN-CC$_\text{+sym+post-SR}$', lw=2.5)
    axs[1].plot(x_extrap, f2_sym, "-", color='teal', label=r"NN-CC$_\text{+sym}$", lw=2)
    axs[1].plot(x_extrap, f2_nosym, "-", color='magenta', label="NN-CC", lw=2)
    axs[1].plot(x_extrap, F2(x_extrap), "--", color='black',dashes=(4, 6), label="Theor.", lw=2)
    axs[1].set_xlabel("$x$  (m)", fontsize=24)
    axs[1].set_ylabel("$f_2(x)$  (N)", fontsize=24)
    F2_th_range = np.max(F2_th) - np.min(F2_th)
    axs[1].set_ylim(np.min(F2_th) - F2_th_range * 0.5, np.max(F2_th) + F2_th_range * 0.5)
    axs[1].legend(fontsize=15,loc='upper left',labelspacing=0.05,frameon=False)
    custom_ticks(axs[1], major_x_interval=2, major_y_interval=2, minor_x_interval=0.5, minor_y_interval=1.0)
    axs[1].text(0.96, 0.15, "(d)", transform=axs[1].transAxes, fontsize=20, va='top', ha='right')
    plt.tight_layout()
    folder_path = output_path #"/content/drive/My Drive/Colab Notebooks/Plots"
    os.makedirs(folder_path, exist_ok=True)
    file_name = "Fig10_cd_stick-slip_CC.pdf"
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    print(f"Saved to: {file_path}")
    
    



    ################################################################################
    ################################################################################
    ################################   VALIDATION   ################################
    ################################################################################
    ################################################################################



    print('Validating the system')
    print('Integration of model EDOs')

    #n_trials = 10  # number of random initial conditions
    #rmse_x_NN_list = []
    #rmse_x_dot_NN_list = []
    #rmse_x_Sindy_list = []
    #rmse_x_dot_Sindy_list = []
    #rmse_x_LS_list = []
    #rmse_x_dot_LS_list = []


#    for i in range(n_trials):
# Validation Loop Variables
    n_trials = 20        # Target number of valid simulations
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
        plt.show()


        ################ NN-CC ###### 
        def learned_dynamics(t, y):
            x = torch.tensor([[y[0]]], dtype=torch.float32)
            x_dot = torch.tensor([[y[1]]], dtype=torch.float32)
            t_tensor = torch.tensor([[t]], dtype=torch.float32)
            F_ext_tensor = torch.tensor([[F_ext(t)]], dtype=torch.float32)
            model1_nosym.eval()
            model2_nosym.eval()
            with torch.no_grad(): # Neural net-based force computation
                force = F_ext_tensor - model1_nosym(x_dot) - model2_nosym(x)
            x_ddot = force.item()
            return [y[1], x_ddot]  # dx/dt = x_dot, dx_dot/dt = x_ddot
        print("Integrating NN")
        start = time.time()  
        x_simulated_NN=[]
        x_dot_simulated_NN=[]
        try:
            sol = solve_ivp(learned_dynamics, t_span_val, y0_val, t_eval=t_val , method='LSODA') #,rtol=1e-7,atol=1e-7) #,  method='DOP853', rtol=1e-9, atol=1e-12)
            x_simulated_NN_nosym = sol.y[0]
            x_dot_simulated_NN_nosym = sol.y[1]            # check for NaNs or infs just in case
            t_NN_nosym=sol.t
            if np.any(np.isnan(x_simulated_NN_nosym)) or np.any(np.isinf(x_simulated_NN_nosym)):
                print(f"Skipping trial {valid_trials}: NN simulation returned NaNs or infs.")
        except Exception as e:
            print(f"Skipping trial {valid_trials}: Exception during NN simulation -> {e}")
        end = time.time()  
        elapsed = end - start
        print(f"Solve_ivp finished in {elapsed:.3f} seconds")

        plt.figure()
        plt.plot(t_NN_nosym, x_simulated_NN_nosym, label="NN-CC")
        plt.plot(t_simulated_th, x_simulated_th, label="Theor.")
        #plt.plot(time_data, x_data, label="true")
        plt.legend()
        plt.xlabel("t")
        plt.ylabel("x(t)")
        plt.show()


        ################ NN-CC+sym ###### 
        def learned_dynamics(t, y):
            x = torch.tensor([[y[0]]], dtype=torch.float32)
            x_dot = torch.tensor([[y[1]]], dtype=torch.float32)
            t_tensor = torch.tensor([[t]], dtype=torch.float32)
            F_ext_tensor = torch.tensor([[F_ext(t)]], dtype=torch.float32)
            model1.eval()
            model2.eval()
            with torch.no_grad(): # Neural net-based force computation
                force = F_ext_tensor - model1(x_dot) - model2(x)
            x_ddot = force.item()
            return [y[1], x_ddot]  # dx/dt = x_dot, dx_dot/dt = x_ddot
        print("Integrating NN")
        start = time.time()  
        x_simulated_NN=[]
        x_dot_simulated_NN=[]
        try:
            sol = solve_ivp(learned_dynamics, t_span_val, y0_val, t_eval=t_val , method='LSODA') #,rtol=1e-7,atol=1e-7) #,  method='DOP853', rtol=1e-9, atol=1e-12)
            x_simulated_NN = sol.y[0]
            x_dot_simulated_NN = sol.y[1]            # check for NaNs or infs just in case
            t_NN=sol.t
            if np.any(np.isnan(x_simulated_NN)) or np.any(np.isinf(x_simulated_NN)):
                print(f"Skipping trial {valid_trials}: NN simulation returned NaNs or infs.")
        except Exception as e:
            print(f"Skipping trial {valid_trials}: Exception during NN simulation -> {e}")
        end = time.time()  
        elapsed = end - start
        print(f"Solve_ivp finished in {elapsed:.3f} seconds")

        plt.figure()
        plt.plot(t_NN, x_simulated_NN, label=r"NN-CC$_{+sym}$")
        plt.plot(t_simulated_th, x_simulated_th, label="Theor.")
        #plt.plot(time_data, x_data, label="true")
        plt.legend()
        plt.xlabel("t")
        plt.ylabel("x(t)")
        plt.show()


        ################ NN+SR ###### validation of the model
        print("Integrating NN-SR")

        # For f1
        expr_f1 = model_f1SR.sympy()
        f1_lambda = sp.lambdify(sp.symbols("x0"), expr_f1, "numpy")
        
        expr_f1_smooth = expr_f1.replace(sp.sign, lambda arg: sp.tanh(500*arg))
        expr_f1_smooth = expr_f1_smooth.replace(sp.Abs, lambda arg: sp.sqrt(arg**2+1e-6))
        print(model_f1SR.sympy())
        print(expr_f1_smooth)
        f1_lambda = sp.lambdify(sp.symbols("x0"), expr_f1_smooth, "numpy")

        # For f2
        expr_f2 = model_f2SR.sympy()
        f2_lambda = sp.lambdify(sp.symbols("x0"), expr_f2, "numpy")
        def learned_dynamics_SR(t, y):
            x_val = y[0]
            x_dot_val = y[1]
            F_ext_val = F_ext(t)
            f1_val = f1_lambda(x_dot_val)
            f2_val = f2_lambda(x_val)
            x_ddot = F_ext_val - f1_val - f2_val
            return [x_dot_val, x_ddot]
        start = time.time()  
        try:
            sol_nn_sr = solve_ivp(learned_dynamics_SR, t_span_val, y0_val, 
                              t_eval=t_val, method="LSODA") #Radau DOP853, rtol=1e-6, atol=1e-9)
            t_NN_SR = sol_nn_sr.t
            x_simulated_NN_SR = sol_nn_sr.y[0]
            x_dot_simulated_NN_SR = sol_nn_sr.y[1]
            # check for NaNs or infs just in case
            if np.any(np.isnan(x_simulated_NN_SR)) or np.any(np.isinf(x_simulated_NN_SR)):
                print(f"Skipping trial {valid_trials}: NN-CC-SR simulation returned NaNs or infs.")

        except Exception as e:
            print(f"Skipping trial {valid_trials}: Exception during NN-CC-SR simulation -> {e}")

        
        end = time.time()  
        elapsed = end - start
        print(f"Solve_ivp finished in {elapsed:.3f} seconds")
        

        plt.figure()
        plt.plot(t_NN_SR, x_simulated_NN_SR, label=r"NN-CC$_{+sym+post\!\!-\!\!SR}$")
        plt.plot(t_simulated_th, x_simulated_th, label="Theor.")
        #plt.plot(time_data, x_data, label="true")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("x(t)")
        plt.show()


        # CHECK if some integration failed
        if len(x_simulated_NN) != len(t_val):
            print("Warning:")
            print("NN-CC finished integration before maximum simulation time")
            print("time:",t_val[len(x_simulated_NN)-1])
        if len(x_simulated_NN_nosym) != len(t_val):
            print("Warning:")
            print("NN-CC nosym finished integration before maximum simulation time")
            print("time:",t_val[len(x_simulated_NN_nosym)-1])
        if len(x_simulated_NN_SR) != len(t_val):
            print("Warning:")
            print("NN-CC-SR finished integration before maximum simulation time")
            print("time:",t_val[len(x_simulated_NN_SR)-1])

        # Plot the simulation results
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(t_val, x_simulated_th,'-',color='blue', label="$x_{th}$",linewidth='2')
        plt.plot(t_val, x_simulated_NN,"--",color='orange', label="$x_{val} NN$",linewidth='3')
        plt.ylim(np.min(x_simulated_th)-0.2,np.max(x_simulated_th)+0.2)
        plt.xlabel("Time $t$")
        plt.ylabel("Position")
        plt.title("Validation Test of Position over Time")
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(t_val, x_dot_simulated_th,color='blue', label="$\\dot{x}_{th}$")
        plt.plot(t_val, x_dot_simulated_NN,"--",color='orange', label="$\\dot{x}_{val} NN$", linestyle="dashed",linewidth='3')
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



        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=False)

        # ---- (a) x(t) values ----
        axs[0, 0].plot(t_val, x_simulated_th, '-', color='blue', label="Theor.", linewidth=3)
        axs[0, 0].plot(t_val, x_simulated_NN, "--", color='orange', label=r"NN-CC$_\text{+sym}$", linewidth=3)
        axs[0, 0].plot(t_val, x_simulated_NN_nosym, "--", color='violet', label="NN-CC nosym", linewidth=3)
        axs[0, 0].plot(t_val, x_simulated_NN_SR, "--", color='magenta', label=r"NN-CC$_\text{+sym+post-SR}$", linewidth=3)
        axs[0, 0].set_ylabel("$x$", fontsize=24)
        axs[0, 0].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1))
        custom_ticks(axs[0, 0], 20, 0.5 , 5 , 0.25)
        axs[0, 0].text(0.98, 0.04, "(a)", transform=axs[0, 0].transAxes, fontsize=24, va='bottom', ha='right')

        # ---- (b) x_dot(t) values ----
        axs[0, 1].plot(t_val, x_dot_simulated_th, '-', color='blue', label="Theor.", linewidth=3)
        axs[0, 1].plot(t_val, x_dot_simulated_NN, "--", color='orange', label=r"NN-CC$_\text{+sym}$", linewidth=3)
        axs[0, 1].plot(t_val, x_dot_simulated_NN_nosym, "--", color='violet', label="NN-CC nosym", linewidth=3)
        axs[0, 1].plot(t_val, x_dot_simulated_NN_SR, "--", color='magenta', label=r"NN-CC$_\text{+sym+post-SR}$", linewidth=3)
        axs[0, 1].set_ylabel("$\dot{x}$", fontsize=24)
        axs[0, 1].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1))
        custom_ticks(axs[0, 1], 20, 0.5, 5, 0.25)
        axs[0, 1].text(0.98, 0.04, "(b)", transform=axs[0, 1].transAxes, fontsize=24, va='bottom', ha='right')

        # ---- (c) Residuals: x_model - x_theor ----
        axs[1, 0].plot(t_val, x_simulated_NN - x_simulated_th, '--', color='orange', label=r"NN-CC$_\text{+sym}$", linewidth=3)
        axs[1, 0].axhline(0, color='black', linewidth=1)
        axs[1, 0].set_ylabel("$x-x_{th.}$", fontsize=22)
        axs[1, 0].set_xlabel("$t$", fontsize=24)
        axs[1, 0].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1))
        custom_ticks(axs[1, 0], 20, 0.2, 5, 0.1)
        axs[1, 0].text(0.9, 0.04, "(c)", transform=axs[1, 0].transAxes, fontsize=24, va='bottom', ha='right')

        # ---- (d) Residuals: x_dot_model - x_dot_theor ----
        axs[1, 1].plot(t_val, x_dot_simulated_NN - x_dot_simulated_th, '--', color='orange', label=r"NN-CC$_\text{+sym}$", linewidth=3)
        axs[1, 1].axhline(0, color='black', linewidth=1)
        axs[1, 1].set_ylabel("$\dot{x}-\dot{x}_{th.}$", fontsize=22)
        axs[1, 1].set_xlabel("$t$", fontsize=24)
        axs[1, 1].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1))
        custom_ticks(axs[1, 1], 20, 0.1, 5, 0.05)
        axs[1, 1].text(0.9, 0.04, "(d)", transform=axs[1, 1].transAxes, fontsize=24, va='bottom', ha='right')

        plt.tight_layout()

        # Save
        folder_path = output_path
        # "/content/drive/My Drive/Colab Notebooks/Plots"
        os.makedirs(folder_path, exist_ok=True)
        file_name = "valid_duffing.pdf"
        file_path = os.path.join(folder_path, file_name)
        plt.savefig(file_path, format='pdf', bbox_inches='tight')
#       plt.show()
        print(f"Saved to: {file_path}")




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

        time_chaos_x_NN=t_val[len(x_simulated_NN)-1]
        for i in range(len(x_simulated_NN)):
            diff = abs(x_simulated_NN[i] - x_simulated_th[i])
            if diff > threshold_chaos:
                #time_chaos_x_NN_list.append(t_val[i])
                time_chaos_x_NN =t_val[i]
                break
        time_chaos_x_NN_list.append(time_chaos_x_NN)
        #time_chaos_x_NN_nosym_list.append(t_val[-1])

        time_chaos_x_NN_nosym=t_val[len(x_simulated_NN_nosym)-1]
        for i in range(len(x_simulated_NN_nosym)-1):
            diff = abs(x_simulated_NN_nosym[i] - x_simulated_th[i])
            if diff > threshold_chaos:
                time_chaos_x_NN_nosym=t_val[i]
                break
        time_chaos_x_NN_nosym_list.append(time_chaos_x_NN_nosym)

        time_chaos_x_NN_SR=t_val[len(x_simulated_NN_SR)-1]
        for i in range(len(x_simulated_NN_SR)):
            diff = abs(x_simulated_NN_SR[i] - x_simulated_th[i])
            if diff > threshold_chaos:
                #time_chaos_x_NN_SR_list.append(t_val[i])
                time_chaos_x_NN_SR=t_val[i]
                break
        time_chaos_x_NN_SR_list.append(time_chaos_x_NN_SR)
        #time_chaos_x_LS_list.append(t_val[-1])

        time_matrix_append = np.column_stack([
            noise_percentage_th,
            noise_percentage,
            SNR_dB,
            time_chaos_x_NN_nosym,
            time_chaos_x_NN,
            time_chaos_x_NN_SR,
            time_chaos_x_parametric
        ])
        folder_path = output_path
        os.makedirs(folder_path, exist_ok=True)
        file_name = "times_noise_chaos_duffing_SR_and_param.txt"
        file_path = os.path.join(folder_path, file_name)
        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a') as f:
            np.savetxt(f, time_matrix_append,
                       header="nois_th nois nois_db  NN-CC  NN-CC+sym NN-CC+sym+post-SR  Parametric" if not file_exists else '',
                       fmt="%.2f", delimiter=" ", comments='')



        # Calculation of RMSE values
        #  print(f"Trial {i+1} : x0={x0_val:.4f} ; v0={v0_val:.4f}")
        print("RMSE values:")
        rmse_x_NN = np.sqrt(np.mean((x_simulated_th[0:len(x_simulated_NN)] - x_simulated_NN) ** 2))
        rmse_x_dot_NN = np.sqrt(np.mean((x_dot_simulated_th[0:len(x_dot_simulated_NN)] - x_dot_simulated_NN) ** 2))
        # Print the results
        print(f"NN results   \t - x: {rmse_x_NN:.6f}, x': {rmse_x_dot_NN:.6f}")
        rmse_x_NN_list.append(rmse_x_NN)
        rmse_x_dot_NN_list.append(rmse_x_dot_NN)

        rmse_x_NN_nosym = np.sqrt(np.mean((x_simulated_th[0:len(x_simulated_NN_nosym)] - x_simulated_NN_nosym) ** 2))
        rmse_x_dot_NN_nosym = np.sqrt(np.mean((x_dot_simulated_th[0:len(x_dot_simulated_NN_nosym)] - x_dot_simulated_NN_nosym) ** 2))
        print(f"NN results retrained \t - x: {rmse_x_NN_nosym:.6f}, x': {rmse_x_dot_NN_nosym:.6f}")
        rmse_x_NN_nosym_list.append(rmse_x_NN_nosym)
        rmse_x_dot_NN_nosym_list.append(rmse_x_dot_NN_nosym)
        rmse_x_NN_SR = np.sqrt(np.mean((x_simulated_th[0:len(x_simulated_NN_SR)] - x_simulated_NN_SR) ** 2))
        rmse_x_dot_NN_SR = np.sqrt(np.mean((x_dot_simulated_th[0:len(x_dot_simulated_NN_SR)] - x_dot_simulated_NN_SR) ** 2))
        print(f"NN-SR \t - x: {rmse_x_NN_SR:.6f}, x': {rmse_x_dot_NN_SR:.6f}")
        rmse_x_NN_SR_list.append(rmse_x_NN_SR)
        rmse_x_dot_NN_SR_list.append(rmse_x_dot_NN_SR)


        rmse_matrix_append = np.column_stack([
            noise_percentage_th,
            noise_percentage,
            SNR_dB,
            rmse_x_NN_nosym,
            rmse_x_dot_NN_nosym,
            rmse_x_NN,
            rmse_x_dot_NN,
            rmse_x_NN_SR,
            rmse_x_dot_NN_SR
        ])
        folder_path = output_path
        os.makedirs(folder_path, exist_ok=True)
        file_name = "rmse_noise_duffing_NN_without_symmetry.txt"
        file_path = os.path.join(folder_path, file_name)
        # Save with header and space as delimiter
        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a') as f:
            np.savetxt(f, rmse_matrix_append,
                       header="noise_th noise noise_db rmse_x_NN_nosym rmse_x_dot_NN_nosym rmse_x_NN rmse_x_dot_NN  rmse_x_NN-CC+sym+SR rmse_x_dot_NN-CC+sym+SR" if not file_exists else '',
                       fmt="%.8f", delimiter=" ", comments='')


    # Compute overall RMSE and standard deviation
    total_rmse_x_NN = np.mean(rmse_x_NN_list)
    std_rmse_x_NN = np.std(rmse_x_NN_list)
    total_rmse_x_dot_NN = np.mean(rmse_x_dot_NN_list)
    std_rmse_x_dot_NN = np.std(rmse_x_dot_NN_list)
    total_rmse_x_NN_nosym = np.mean(rmse_x_NN_nosym_list)
    std_rmse_x_NN_nosym = np.std(rmse_x_NN_nosym_list)
    total_rmse_x_dot_NN_nosym = np.mean(rmse_x_dot_NN_nosym_list)
    std_rmse_x_dot_NN_nosym = np.std(rmse_x_dot_NN_nosym_list)
    total_rmse_x_NN_SR = np.mean(rmse_x_NN_SR_list)
    std_rmse_x_NN_SR = np.std(rmse_x_NN_SR_list)
    total_rmse_x_dot_NN_SR = np.mean(rmse_x_dot_NN_SR_list)
    std_rmse_x_dot_NN_SR = np.std(rmse_x_dot_NN_SR_list)

    # Print results
    print("\n======= Total RMSE over all trials (mean ± std, % std) =======")
    print("NN-CC+sym results")
    print(f"Position (x):     {total_rmse_x_NN:.6f} ± {std_rmse_x_NN:.6f}  ({std_rmse_x_NN/total_rmse_x_NN*100:.6f}%)")
    print(f"Velocity (x'):    {total_rmse_x_dot_NN:.6f} ± {std_rmse_x_dot_NN:.6f}  ({std_rmse_x_dot_NN/total_rmse_x_dot_NN*100:.6f}%)")
    print("NN-CC nosym results")
    print(f"Position (x):     {total_rmse_x_NN_nosym:.6f} ± {std_rmse_x_NN_nosym:.6f}  ({std_rmse_x_NN_nosym/total_rmse_x_NN_nosym*100:.6f}%)")
    print(f"Velocity (x'):    {total_rmse_x_dot_NN_nosym:.6f} ± {std_rmse_x_dot_NN_nosym:.6f}  ({std_rmse_x_dot_NN_nosym/total_rmse_x_dot_NN_nosym*100:.6f}%)")
    print("NN-CC+sym+post-SR results")
    print(f"Position (x):     {total_rmse_x_NN_SR:.6f} ± {std_rmse_x_NN_SR:.6f}  ({std_rmse_x_NN_nosym/total_rmse_x_NN_SR*100:.6f}%)")
    print(f"Velocity (x'):    {total_rmse_x_dot_NN_SR:.6f} ± {std_rmse_x_dot_NN_SR:.6f}  ({std_rmse_x_dot_NN_nosym/total_rmse_x_dot_NN_SR*100:.6f}%)")


    # These lists must already be defined
    # Each one should contain RMSE values over multiple trials
    # e.g., rmse_x_NN_list = [rmse_trial1, rmse_trial2, ..., rmse_trialN]
    # same for other methods
    rmse_data = [
        rmse_x_NN_nosym_list,
        rmse_x_NN_list,
        rmse_x_NN_SR_list
    ]
    labels = ['NN-CC_nosym','NN-CC+sym','NN-CC+sym+post-SR']
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
        rmse_x_dot_NN_nosym_list,
        rmse_x_dot_NN_list,
        rmse_x_dot_NN_SR_list
    ]
    labels = ['NN-CC_nosym','NN-CC+sym','NN-CC+sym+post-SR']
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
        rmse_x_NN_nosym_list,
        rmse_x_dot_NN_nosym_list,
        rmse_x_NN_list,
        rmse_x_dot_NN_list,
        rmse_x_NN_SR_list,
        rmse_x_dot_NN_SR_list
    ])
    folder_path = output_path
    os.makedirs(folder_path, exist_ok=True)
    file_name = "rmse_results_duffing_NN.txt"
    file_path = os.path.join(folder_path, file_name)
    # Save with header and space as delimiter
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a') as f:
        np.savetxt(f, rmse_matrix,
                   header="noise_th noise_meas noise_db rmse_x_NN_nosym rmse_x_dot_NN_nosym rmse_x_NN+sym rmse_x_dot_NN+sym rmse_x_NN+sym+SR rmse_x_dot_NN+sym+SR" if not file_exists else '',
                   fmt="%.8f", delimiter=" ", comments='')
    #np.savetxt(file_path, rmse_matrix,
    #           header="rmse_x_NN rmse_x_dot_NN rmse_x_Sindy rmse_x_dot_Sindy rmse_x_LS rmse_x_dot_LS",
    #           fmt="%.8f", delimiter=" ")



#test_predictions = model(test_inputs).cpu().numpy()  # Move predictions to CPU for plotting



