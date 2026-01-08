
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
#from google.colab import drive
#drive.mount('/content/drive')
#output_path = "/content/drive/My Drive/Colab Notebooks/Second_order_noise/Python"
#output_path = "/content/drive/Shared with me/Federico2024_System_Identification/Python"
output_path = "./"
output_file_log = open("output_log.txt", "w")

from pysr import PySRRegressor
import sympy as sp

def printF(*args, **kwargs):
    """
    Prints to the console and simultaneously writes to a log file.
    It mimics the behavior of the built-in print() function.
    """
    # Print to the console
    print(*args, **kwargs)
    
    # Construct the string to be written to the file
    # `sep` and `end` are handled based on the kwargs provided
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    
    # Convert all arguments to strings and join them
    message = sep.join(map(str, args)) + end
    
    # Write the message to the file
    output_file_log.write(message)
    # Ensure the data is written to the file immediately
    output_file_log.flush()



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
# Definición del sistema stick-slip:
# EDO: x'' + f1(x') + f2 (x) = F_ext(t)


# S1: dieterich-ruina DR
# EDO: m x'' + c * x' + Ff(x') + k * x  = F_ext(t)
# EDO: f1(x')= [c*x'+Ff(x')]/m
# EDO: f2(x)= k*x/m
# EDO: F_ext= A cos(Omega * t)
# wn=sqrt(k/m)
# c=zeta*2*sqrt(k*m)
#Ff_dr={Ff+a*ln[(|x'|+epsilon)/Vf]+b*ln[c+Vf/(|x'|+epsilon)]}sgn(x')
#Ff_coul= mu N sgn(x')

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

noise=0.0



#friction definition Dietrich-Ruina
#Ff=0.5 # N
#a=0.07
#b=0.09
##c=0.022
#Vf=0.003 # m/s
#epsilon=1e-6 # m/s
Tsimul=40
Nsimul=1000
Tval=2*Tsimul
Nval=2*Nsimul
np.random.seed(0) # to have repetitivity
torch.manual_seed(10)  # Replace 1 with any integer

t_span = (0, Tsimul)  # time interval for training dataset
t_simul = np.linspace(*t_span, Nsimul)  
t_span_val = (0, Tval)  # time interval for forward simulation
t_val = np.linspace(*t_span_val, Nval)   

time_chaos_x_SR_list=[]
time_chaos_x_parametric_list=[]
time_chaos_x_NN_list =[]
time_chaos_x_NN_retrain_list =[]
time_chaos_x_NN_SR_list =[]
time_chaos_x_Sindy_list =[]
time_chaos_x_LS_list =[]
time_chaos_x_Sindy_ku0_list=[]

rmse_x_SR_list = []
rmse_x_dot_SR_list = []
rmse_x_parametric_list = []
rmse_x_dot_parametric_list = []
rmse_x_NN_list = []
rmse_x_dot_NN_list = []
rmse_x_NN_retrain_list = []
rmse_x_dot_NN_retrain_list = []
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
printF("EDO: x'' + f1(x') + f2 (x) = F_ext(t)")
printF("S1: stick-slip")
printF("f1(x')= [c*x'+Ff(x')]/m")
printF("f2(x)=[k x]/m")
printF("F_ext(t)=F_ext_true(t)/m")



#SNR_dB_list = [np.inf] + list(np.linspace(40, -20, 61 ))  # ∞, 20, 17.5, ..., -5
SNR_dB_list = [np.inf] + list(np.linspace(40, 5, 36 ))  # ∞, 20, 17.5, ..., -5
#SNR_dB_list = list(np.linspace(-18, -20, 3 ))  # ∞, 20, 17.5, ..., -5

SNR_dB_list = list(np.linspace(5, -20, 26 ))  # ∞, 20, 17.5, ..., -5

SNR_dB_list = [20.0]

#repeat 3 times each value in the list
SNR_dB_list = np.repeat(SNR_dB_list, 1)


#SNR_dB_list = np.repeat(SNR_dB_list,B_list = list(np.linspace(5, -5, 3))  # ∞, 20, 17.5, ..., -5


for SNR_dB in SNR_dB_list:


    #test_first in list
    #SNR_dB = [np.inf]
    #SNR_dB = 2.0
    #SNR_dB = -5.0
    # Continue with identification using `noisy_signal`

    #parameters stick-slip
    #A=2
    ##A = np.round(np.random.uniform(0.5, 1.5),3)
    #kval = np.round(np.random.uniform(1, 1.5),3)
    #cval = np.round(np.random.uniform(0.1, 0.5),3)
    ##m = 1.0
    #mu_N = np.round(np.random.uniform(0.5, 1.0),3)
    #Omega = np.round(np.random.uniform(0.2, 0.5),3)
    #x0 = np.round(np.random.uniform(-0.5, 0.5),3)
    #v0 = np.round(np.random.uniform(-0.5, 0.5),3)
    #print(f"Aext={Aext}, k={kval}, c={cval}")
    #print(f"Omega={Omega}, mu_N={mu_N}, $x_0$={x0}, $v_0$={v0}")
    ##print(rf"$\Omega$={Omega}, $\mu N$={mu_N}, $x_0$={x0}, $v_0$={v0}")
    print(f"SNR_dB={SNR_dB}")
    #alpha=-1.0
    #beta=1.0
    #delta=0.3
    #x0=0.5
    #v0=-0.5
    #Aext=2.0
    #Omega=1.2
    print(f"alpha={alpha}, beta={beta}, delta={delta}")
    print(f"Omega={Omega}, Aext={Aext}, $x_0$={x0}, $v_0$={v0}")
    printF(f"alpha={alpha}, beta={beta}, delta={delta}")
    printF(f"Omega={Omega}, Aext={Aext}, $x_0$={x0}, $v_0$={v0}")

    y0 = [x0, v0]  # [x(0), x'(0)]
    
    # Hyperparameters for NN
    learning_rate = 1e-4
    epochs_max = 20000
    neurons=100
    error_threshold = 1e-8
    f1_symmetry='odd'
    f2_symmetry='odd'
    lambda_penalty = 1e-4  # You can adjust this weight if needed
    lambda_penalty_symm = 1e1
    apply_restriction=True
    weight_decay = 1e-6 # 0.0 # 1e-6 was the better, 0.0 default
    momentum=0.99
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

    #  1 x'' + 0.1 * x' + 0.5 sign(x') + k * x  = F_ext(t)

    # EDO: m x'' + c * x' + Ff(x') + k * x  = F_ext(t)
    # wn=sqrt(k/m)
    # c=zeta*2*sqrt(k*m)
    # F_ext= A cos(Omega * t)
    #Ff={Ff+a*ln[(|x'|+epsilon)/Vf]+b*ln[c+Vf/(|x'|+epsilon)]}sgn(x')

    #def van_der_pol_with_time_F_discontinuous(t,y):
    #    x, x_dot = y  # x, x', and time
    #    if x > 1:  # Introducing a discontinuity when x > 1
    ##        x_ddot = -x  # Ignore the Van der Pol term and set x_ddot to just -x
    #        f = 2*mu * (2 - x**2)  # Ignore the Van der Pol term and set x_ddot to just -x
    #    else:
    #        f = mu * (1 - x**2)  # Original Van der Pol term when x <= 1
    #    x_ddot = f * x_dot - x
    #    return [x_dot, x_ddot]


    # Generar datos de la EDO con solve_ivp
    #sol = solve_ivp(van_der_pol_with_time_F, t_span, y0, t_eval=t_eval)

    #sol = solve_ivp(eq_2nd_ord_veloc, t_span, y0, t_eval=t_simul)

    #def stick_event(t, y):
    #    return y[1]  # Detect when velocity crosses zero
    #stick_event.terminal = False
    #stick_event.direction = 0  # Detect all zero crossings
    #sol = solve_ivp(eq_2nd_ord_veloc, t_span, y0, t_eval=t_simul,
    #                events=stick_event, method='Radau')

    #sol = solve_ivp(eq_2nd_ord_veloc, t_span, y0, t_eval=t_simul)
    sol = solve_ivp(eq_2nd_ord_veloc, t_span, y0, t_eval=t_simul,method='LSODA') #LSODA



    #, method='BDF', rtol=1e-6, atol=1e-8, dense_output=True)
    print(sol.status)   # 0 = success, 1 = reached event, -1 = failed
    print(sol.message)

    #, method='DOP853', rtol=1e-9, atol=1e-12)
    #, method='Radau', rtol=1e-6, atol=1e-8
    #, method='BDF'


    plt.plot(sol.t, sol.y[0])
    plt.xlabel("Time")
    plt.ylabel("x(t)")
    plt.title("Displacement")
    plt.grid(True)
    plt.show()


    x_data = sol.y[0]+np.random.normal(0,0.1)*noise      # Posición
    x_dot_data = sol.y[1] #+np.random.normal(0,0.01)   # Velocidad
    time_data = sol.t      # Time (x2)





    # Add noise to x_data for a given SNR (in dB)
    #SNR_dB = 0  # desired signal-to-noise ratio in decibels
    #signal_power = np.mean(x_data**2)
    #noise_power = signal_power / (10**(SNR_dB / 10))
    #noise_std = np.sqrt(noise_power)
    #x_data_noisy = x_data + np.random.normal(0, noise_std, size=x_data.shape)
    #x_noise_substraction = x_data_noisy - x_data
    #signal_power = np.mean(x_data**2)
    #noise_power = np.mean(x_noise_substraction**2)
    #snr_measured = 10 * np.log10(signal_power / noise_power)
    #print(f"Desired SNR: {SNR_dB} dB")
    #print(f"Measured SNR: {snr_measured:.2f} dB")

    # Add noise to x_data for a given SNR (in dB)

    if np.isinf(SNR_dB):
        print("Running with SNR = ∞ dB (no noise)")
        print("noise=",noise)
        printF("Running with SNR = ∞ dB (no noise)")
        printF("noise=",noise)
        F_ext_val = F_ext(time_data)+np.random.normal(0,0.1)*noise
        noise_percentage=0.0
        noise_percentage_th=0.0
    else:
        print(f"Running with SNR = {SNR_dB:.2f} dB")
        printF(f"Running with SNR = {SNR_dB:.2f} dB")
        # Add noise based on current SNR_dB

        #SNR_dB = 4  # desired signal-to-noise ratio in decibels
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
        #F_fr=Ff_dr(x_dot_data)
        #F1_th=F1(x_dot_data)
        #F2_th=F2(x_data)
        #F_ext_val = F_ext(time_data)+noise*np.random.normal(0,0.5)
        print(f"Desired SNR in Fext: {SNR_dB} dB")
        print(f"Measured SNR in Fext: {snr_measured:.2f} dB")
        print(f"Noise percentage in Fext: {noise_percentage:.2f}%")
        print(f"Noise percentage in Fext (theoretical): {noise_percentage_th:.2f}%")
        printF(f"Desired SNR in Fext: {SNR_dB} dB")
        printF(f"Measured SNR in Fext: {snr_measured:.2f} dB")
        printF(f"Noise percentage in Fext: {noise_percentage:.2f}%")
        printF(f"Noise percentage in Fext (theoretical): {noise_percentage_th:.2f}%")
        # --- now apply a Savitzky–Golay filter ---
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
        printF(f"SNR after SG filter: {snr_after:.1f} dB")

        plt.figure(figsize=(6, 4))
        plt.plot(time_data, F_ext(time_data),         label='Fext (true)')
        plt.plot(time_data, F_ext_val_noisy,          label='Fext + noise', alpha=0.7)
        plt.plot(time_data, F_ext_filtered,           label='SG-filtered', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Fₑₓₜ')
        plt.title('Original vs Noisy vs SG-Filtered Forcing')
        plt.legend()
        plt.tight_layout()
        plt.show()


        #F_fr=Ff_dr(x_dot_data)
        #F1_th=F1(x_dot_data)
        #F2_th=F2(x_data)
        F_ext_val = F_ext(time_data)+np.random.normal(0,0.1)*noise
        F_ext_val = F_ext_filtered
        F_ext_val = F_ext_val_noisy

    F1_th=F1(x_dot_data)
    F2_th=F2(x_data)

    #F1_th_noisy=F1(x_dot_data_noisy)
    #F2_th_noisy=F2(x_data_noisy)


    print("minmax_x", np.min(x_data), np.max(x_data))
    printF("minmax_x", np.min(x_data), np.max(x_data))
    print("minmax_x_dot", np.min(x_dot_data), np.max(x_dot_data))
    printF("minmax_x_dot", np.min(x_dot_data), np.max(x_dot_data))
    #print("minmax_F_ext", np.min(F_ext_val), np.max(F_ext_val))





    #F1_th=F1_anderson2009(x_dot_data,F_ext_val)
    #F2_th=F2_anderson2009(x_data)


    #x_ddot_data = (F_ext_val - F1_th - F2_th) / m
    #x_ddot_data = np.gradient(x_dot_data, sol.t)  # Aceleración (derivada numérica)
    x_ddot_data = np.array([eq_2nd_ord_veloc(t, y)[1] for t, y in zip(sol.t, sol.y.T)])
    #x_ddot_data = np.array([eq_2nd_ord_veloc_anderson2009(t, y)[1] for t, y in zip(sol.t, sol.y.T)])


    plt.figure()
    plt.plot(time_data,(x_ddot_data-F_ext_val+F1_th+F2_th)**2)
    plt.show()



    ############################################
    ########### IDENTIFICATION #################
    ############################################


    ############# SR black box #################
    target_SR=F_ext_val-x_ddot_data 
    X_SR = np.column_stack([x_data, x_dot_data])
    model = PySRRegressor(
        niterations=200,
        binary_operators=["+", "-", "*", "pow"],  #"/"
        #unary_operators=[ "log", "abs","sign"], 
        #unary_operators=["sin", "cos", "exp", "log", "abs"],
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
    printF(f"Training finished in {elapsed:.3f} seconds")

    

    print(model)
    print(model.get_best())
    y_pred_SR = model.predict(X_SR)
    best_expr = model.sympy()
    print("\nSymbolic expression")
    print("f(x, xdot):", best_expr)
    x_sym, xdot_sym = sp.symbols("x xdot")

    # comment the following for noise lower than 5dB
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
    f1_expr_sr = sum(f1_terms_sr)
    f2_expr_sr = sum(f2_terms_sr)
    f1_fun_sr = sp.lambdify(xdot_sym, f1_expr_sr, "numpy")
    f2_fun_sr = sp.lambdify(x_sym, f2_expr_sr, "numpy")
    # data ranges
    x_vals = np.linspace(np.min(x_data), np.max(x_data), 200)
    xdot_vals = np.linspace(np.min(x_dot_data), np.max(x_dot_data), 200)
    plt.figure()
    plt.plot(xdot_vals, f1_fun_sr(xdot_vals), label="Identified f1(xdot)")
    plt.plot(xdot_vals, F1(xdot_vals), '.', alpha=0.3, label="True f1(xdot)")
    plt.legend(); plt.title("f1(xdot)")
    plt.show()
    plt.figure()
    plt.plot(x_vals, f2_fun_sr(x_vals), label="Identified f2(x)")
    plt.plot(x_vals, F2(x_vals), '.', alpha=0.3, label="True f2(x)")
    plt.legend(); plt.title("f2(x)")
    plt.show()



    f_SR = sp.lambdify((x_sym, xdot_sym), best_expr, "numpy")
    def ode_sr(t, state):
        x, xdot = state
        xddot = F_ext(t) - f_SR(x, xdot)
        return [xdot, xddot]

#    # test integration
#    x0 = x0
#    v0 = v0
#    print("x0=", x0)
#    print("v0=", v0)
#    # Integrate
#    #t_span = (0, 10)
#    #t_val = np.linspace(t_span[0], t_span[1], 500)
#    sol = solve_ivp(ode_sr, t_span_val, [x0, v0], t_eval=t_val)
#
#    # Results
#    t_SR = sol.t
#    x_SR = sol.y[0]
#    xdot_SR = sol.y[1]
#
#    plt.figure()
#    plt.plot(t_SR, x_SR, label="SR")
#    plt.plot(time_data, x_data, label="true")
#    plt.legend()
#    plt.show()


    ############# Parametric  #################   least squares

    ##### method 1 BEST ####
    # Right-hand side
    rhs = F_ext_val -  x_ddot_data # m *
    # Design matrix: [x_dot, x, x^3]
    A = np.vstack([
        x_dot_data,
        x_data,
        x_data**3
    ]).T  # shape: (N, 3)
    # Solve least squares: A @ [delta, alpha, beta] = rhs
    start = time.time()
    params, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    delta_ident_param, alpha_ident_param, beta_ident_param = params
    print("Parametric model")
    end = time.time()  
    elapsed = end - start
    print(f"Training finished in {elapsed:.3f} seconds")
    printF(f"Training finished in {elapsed:.3f} seconds")
    print("theoretical values")
    print(f"delta = {delta:.6e}, alpha = {alpha:.6e}, beta = {beta:.6e}")
    print("identified values")
    print(f"delta = {delta_ident_param:.6e}, alpha = {alpha_ident_param:.6e}, beta = {beta_ident_param:.6e}")


    def ode_param(t, state):
        x, xdot = state
        xddot = (F_ext(t) - delta_ident_param*xdot - alpha_ident_param*x - beta_ident_param*x**3) # / m
        return [xdot, xddot]
    def f1_param(x_dot):
        return delta_ident_param * x_dot
    def f2_param(x):
        return alpha_ident_param * x + beta_ident_param * x**3
    
    
    # Integrate over same time range as data
    t_span = (time_data[0], time_data[-1])
    t_eval = time_data

    sol = solve_ivp(ode_param, t_span, [x0, v0], t_eval=t_eval,
                    rtol=1e-9, atol=1e-12, max_step=(t_eval[1] - t_eval[0]))
    
    # Extract simulation results
    x_sim = sol.y[0]
    xdot_sim = sol.y[1]

    # Plot comparison
    plt.figure(figsize=(8,5))
    plt.plot(time_data, x_data, label="True x(t)", linewidth=2)
    plt.plot(time_data, x_sim, "--", label="Simulated x(t)", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("x(t)")
    plt.legend()
    plt.title("Parametric model simulation")
    plt.show()


    ##### method 2 ####
    # Example weights: you can also compute based on noise variance
    weights = 1 / (np.abs(x_ddot_data) + 1e-6)  # Avoid division by zero
    # Apply weights
    W = np.diag(weights)
    A_weighted = W @ A
    rhs_weighted = W @ rhs
    params2, _, _, _ = np.linalg.lstsq(A_weighted, rhs_weighted, rcond=None)
    delta_ident2, alpha_ident2, beta_ident2 = params2
    print("theoretical values")
    print(f"delta = {delta:.6e}, alpha = {alpha:.6e}, beta = {beta:.6e}")
    print("identified values")
    print(f"delta = {delta_ident2:.6e}, alpha = {alpha_ident2:.6e}, beta = {beta_ident2:.6e}")


    ##### method 3 ####
    # Mean and std normalization
    x_mean, x_std = x_data.mean(), x_data.std()
    x_dot_mean, x_dot_std = x_dot_data.mean(), x_dot_data.std()
    x_n = (x_data - x_mean) / x_std
    x_dot_n = (x_dot_data - x_dot_mean) / x_dot_std
    x_cubic_n = ((x_data**3) - (x_data**3).mean()) / (x_data**3).std()
    A = np.vstack([
        x_dot_n,
        x_n,
        x_cubic_n
    ]).T
    params3, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    # Back-transform the parameters
    #delta_ident3 = params3[0] / x_dot_std
    #alpha_ident3 = params3[1] / x_std
    #beta_ident3 = params3[2] / (x_data**3).std()
    delta_ident3, alpha_ident3, beta_ident3 = params3
    print("theoretical values")
    print(f"delta = {delta:.6e}, alpha = {alpha:.6e}, beta = {beta:.6e}")
    print("identified values")
    print(f"delta = {delta_ident3:.6e}, alpha = {alpha_ident3:.6e}, beta = {beta_ident3:.6e}")




    ############# Poly-CC  #################   least squares
    N_order=10
    ## Prepare features
    min_x = np.min(np.abs(x_data))
    max_x = np.max(np.abs(x_data))
    min_xd = np.min(np.abs(x_dot_data))
    max_xd = np.max(np.abs(x_dot_data))
    A0_x=(max_x+min_x)/2.0
    A1_x=(max_x-min_x)/2.0
    A0_xd=(max_xd+min_xd)/2.0
    A1_xd=(max_xd-min_xd)/2.0
    ##A0_x=0
    ##A0_xd=0
    ##A1_x=1
    ##A1_xd=1
    X_poly = np.vstack([((x_data-A0_x)/A1_x)**i for i in range(N_order + 1)]).T
    X_dot_poly = np.vstack([((x_dot_data-A0_xd)/A1_xd)**i for i in range(N_order + 1)]).T  # [N x (N_order+1)]
    # Combine both: A @ coeffs = b
    Amat = np.hstack([-X_dot_poly, -X_poly])  # Minus signs because x'' = F_ext - f1 - f2
    bmat = x_ddot_data -  F_ext_val #F_ext(t_simul)       # Leftover = f1(x') + f2(x)
    # ---- Least squares fit ---- #
    start=time.time()
    coeffs, _, _, _ = np.linalg.lstsq(Amat, bmat, rcond=None)
    end = time.time()  
    elapsed = end - start
    print('End Training Poly-CC')
    print(f"Training finished in {elapsed:.3f} seconds")
    printF(f"Training finished in {elapsed:.3f} seconds")

    # Separate coefficients
    c_f1 = coeffs[:N_order+1]
    c_f2 = coeffs[N_order+1:]
    # Generate polynomial features
    print(f"Coefficients of f1(x') (order {N_order}): {c_f1}")
    print(f"Coefficients of f2(x)  (order {N_order}): {c_f2}")
    def f1_fit(x_dot):
        return sum(c * ((x_dot-A0_xd)/A1_xd)**i for i, c in enumerate(c_f1))
    def f2_fit(x):
        return sum(c * ((x-A0_x)/A1_x)**i for i, c in enumerate(c_f2))
    def fitted_model_LS(t, y):
        x, x_dot = y
        x_ddot = F_ext(t) - f1_fit(x_dot) - f2_fit(x)
        return [x_dot, x_ddot]


    # Print coefficients for f1_fit
    print("Coefficients for f1_fit($\\dot{x}$):")
    for i, c in enumerate(c_f1):
        print(f"  c[{i}] = {c:.6e}")
    # Print coefficients for f2_fit
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



    ############## LS - MC  LEGENDRE #################   least squares
    # # uncomment (this works but ignored for being more complex than simple LS)
    #N_order=15
    #x_scaled = (x_data - A0_x) / A1_x # Prepare features
    #xd_scaled = (x_dot_data - A0_xd) / A1_xd # Generate Legendre polynomial features
    #X_poly = legvander(x_scaled, N_order)       # shape (n_samples, N_order+1)
    #X_dot_poly = legvander(xd_scaled, N_order)
    #A = np.hstack([-X_dot_poly, -X_poly]) # Build system
    #b = x_ddot_data - F_ext(t_simul)
    #coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    #c_f1 = coeffs[:N_order+1]
    #c_f2 = coeffs[N_order+1:]
    #print(f"Coefficients of f1 (Legendre basis): {c_f1}")
    #print(f"Coefficients of f2 (Legendre basis): {c_f2}")
    #def legendre_eval(x_scaled, coeffs): # Define f1 and f2 using Legendre basis
    #    """Efficient evaluation using Horner's method for Legendre polynomials"""
    #    return np.polynomial.legendre.legval(x_scaled, coeffs)
    #def f1_fit(x_dot):
    #    xd_s = (x_dot - A0_xd) / A1_xd
    #    return legendre_eval(xd_s, c_f1)
    #def f2_fit(x):
    #    x_s = (x - A0_x) / A1_x
    #    return legendre_eval(x_s, c_f2)
    #def fitted_model_LS(t, y): # Define fitted model for integration
    #    x, x_dot = y
    #    x_ddot = F_ext(t) - f1_fit(x_dot) - f2_fit(x)
    #    return [x_dot, x_ddot]
    #




    #############  SINDY  #################   for modeling with sindy
    # Preparar datos para SINDy (con las variables de posición y velocidad)
    #data = np.stack((x_data, x_dot_data), axis=-1)
    data = np.stack((x_data, x_dot_data,time_data), axis=-1)
    #X_dot_sindy=np.array([x_dot_data,x_ddot_data for i, t in enumerate(t_simul)])
    #u_sindy = np.array([F_ext(sol.t) for t in sol.t])

    X_sindy = np.stack((x_data,x_dot_data),axis=1) #time_data # sol.y.T  # Transpose to get shape (N, 2)
    X_dot_sindy = np.stack((x_dot_data, x_ddot_data), axis=1)
    u_sindy = F_ext_val # np.stack((F_ext_val),axis=1)
    u_sindy_resh = F_ext_val.reshape(-1,1) #flatten()
    X_combined = np.hstack((X_sindy, u_sindy_resh))




    #polynomials and not crossed terms
    # fourier_library = ps.FourierLibrary(n_frequencies=1)
    #combined_library = ps.GeneralizedLibrary([poly_library, fourier_library])

    # working uncomment!!!!
    print("Sindy Polynomial with u0")
    poly_library = ps.PolynomialLibrary(degree=10, include_interaction=False,order='c',include_bias=False) #, include_input=False) #, include_input=False)#,interaction_only=True
    optimizer_sindy = ps.STLSQ(threshold=1e-4, max_iter=10000)
    model = ps.SINDy(feature_library=poly_library,optimizer=optimizer_sindy)
    start = time.time()
    model.fit(X_sindy, t=time_data, x_dot=X_dot_sindy, u=u_sindy)
    end = time.time()  
    elapsed = end - start
    print('End Training SINDy with u0')
    print(f"Training finished in {elapsed:.3f} seconds")
    printF(f"Training finished in {elapsed:.3f} seconds")

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
    printF(f"Training finished in {elapsed:.3f} seconds")

    

    print(" ")
    print("Sindy restricted with term +1.0*u0")
    F_p = poly_library.n_output_features_         # number of state‐only features
    F_u = input_library.n_output_features_        # should be 1 (just u₀)
    F   = F_p + F_u                                # total features per target
    T   = X_sindy.shape[1]                        # number of state equations, here 2
    C = np.zeros((1, F * T), dtype=float)
    d = np.array([1.0])  # RHS: force coefficient == 1
    idx = F + F_p
    C[0, idx] = 1.0
    # 3) Instantiate optimizer with correct keywords
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

    # 4) Fit exactly as before
    start= time.time()
    model = SINDy(feature_library=combined_library, optimizer=optimizer, feature_names=['x0', 'x1', 'u0'])
    model.fit(X_sindy, t=time_data, x_dot=X_dot_sindy, u=u_sindy)
    model.print()
    end = time.time()  
    elapsed = end - start
    print('End Training SINDy with 1*u0')
    print(f"Training finished in {elapsed:.3f} seconds")
    printF(f"Training finished in {elapsed:.3f} seconds")

#import numpy as np
#import matplotlib.pyplot as plt
#import dill  # pip install dill

    # === 1) (Optional) Load a saved model ===
    # with open("sindy_model.dill", "rb") as f:
    #     model = dill.load(f)

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

    # Define Python functions for f1 and f2
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

    # === 4) Plot each over a chosen domain ===
    x_min, x_max = -2, 2
    n_pts = 400
    x_vals = np.linspace(x_min, x_max, n_pts)

    plt.figure(figsize=(6,4))
    plt.plot(x_vals, f1_sindy(x_vals), label=r'$f_1(x_0)$')
    plt.xlabel(r'$x_0$')
    plt.ylabel(r'$f_1(x_0)$ Sindy')
    plt.title(r'$f_1$ vs. $x_0$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(6,4))
    plt.plot(x_vals, f2_sindy(x_vals), label=r'$f_2(x_1)$')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$f_2(x_1)$ Sindy')
    plt.title(r'$f_2$ vs. $x_1$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

#    # Total features = poly_library.n_output_features_ + input_library.n_output_features_
#    F_p = poly_library.n_output_features_    # number of x-monomials
#    F_tot = F_p + input_library.n_output_features_
#
#    # C is 1×F_tot, zero everywhere except a 1 at the u0 index
#    C = np.zeros((1, F_tot), dtype=float)
#    C[0, F_p] = 1.0
#
#    # d = [1] to force that coefficient to equal 1
#    d = np.array([1.0])
#    optimizer = ConstrainedSR3(
#        constraint_lhs=C,
#        constraint_rhs=d,
#        nu=1e-6,            # relaxation parameter
#       # reg_weight_lam=1e-3,       # regularization strength
#        max_iter=10000
#    )
##   ConstrainedSR3(
##       reg_weight_lam=0.005,
##       regularizer="l0",
##       relax_coeff_nu=1.0,
##       tol=1e-5,
##       max_iter=30,
##       trimming_fraction=0.0,
##       trimming_step_size=1.0,
##       constraint_lhs=C,
##       constraint_rhs=d,
##       constraint_order="target",
##       normalize_columns=False,
##       copy_X=True,
##       initial_guess=None,
##       equality_constraints=False,
##       inequality_constraints=False,
##       constraint_separation_index=None,
##       verbose=False,
##       verbose_cvxpy=False,
##       unbias=False,
##   )
#
#    model = ps.SINDy(
#        feature_library=combined_library,
#        optimizer=optimizer
#    )
#    model.fit(X_sindy, t=sol.t, x_dot=X_dot_sindy, u=u_sindy)
#    model.print()

  #other attempts
  #  # Define the features you want in the library (only up to first-order in u0)
  #  features = [
  #      lambda x: x[:, 0],  # x0
  #      lambda x: x[:, 1],  # x1
  #      lambda x: x[:, 2],  # u0 (only first-order)
  #      lambda x: x[:, 0] ** 2,  # x0^2
  #      lambda x: x[:, 1] ** 2,  # x1^2
  #      lambda x: x[:, 0] ** 3,  # x0^3
  #      lambda x: x[:, 1] ** 3,  # x1^3
  #  ]
  #  # Define custom library (safe for 1D or 2D x
#
  #  # Create the custom library
  #  custom_library = CustomLibrary(library_functions=features)
  #
  #  # Fit the model
  #  optimizer_sindy = ps.STLSQ(threshold=1e-3, max_iter=10000)
  #  model = ps.SINDy(feature_library=custom_library, optimizer=optimizer_sindy)
  #  model.fit(X_sindy, t=sol.t, x_dot=X_dot_sindy, u=u_sindy)
  #  model.print()

#    poly_library = ps.PolynomialLibrary(degree=5, include_interaction=False) #, include_input=False) #, include_input=False)#,interaction_only=True
#    optimizer_sindy = ps.STLSQ(threshold=1e-3, max_iter=10000)
#    model = ps.SINDy(feature_library=poly_library,optimizer=optimizer_sindy)
#    model.fit(X_combined, t=sol.t, x_dot=X_dot_sindy)
#    #model = ps.SINDy(feature_library=combined_library)
#    #model.fit(X_sindy, t=sol.t, u=u_sindy, x_dot=X_dot_sindy)  # ¡Aquí pasamos u!
#    print("\nIdentified SINDy model (exponential format, no labels):")
#    model.print()



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

    # Generate time data
    #t = np.linspace(0, 10, 1000)
    #t= data[0,:]
    #print(t[3])


#    # Plot the data
#    plt.figure(figsize=(15, 5))
#    plt.subplot(1, 4, 1) # Plot position x_k vs time
#    plt.plot(time_data, x_data, label='Position $x_k$', color='blue')
#    plt.title('Position $x_k$ vs Time $t$')
#    plt.xlabel('Time $t$')
#    plt.ylabel('Position $x_k$')
#    plt.grid(True)
#    plt.legend()# Plot velocity x'_k vs time
#    plt.subplot(1, 4, 2)
#    plt.plot(time_data, x_dot_data, label='Velocity $\\dot{x}_k$', color='red')
#    plt.title('Velocity $\\dot{x}_k$ vs Time $t$')
#    plt.xlabel('Time $t$')
#    plt.ylabel('Velocity $\\dot{x}_k$')
#    plt.grid(True)
#    plt.legend()
#    plt.subplot(1, 4, 3)# Plot acceleration x''_k vs time
#    plt.plot(time_data, x_ddot_data, label='Acceleration $\\ddot{x}_k$', color='red')
#    plt.title('Acceleration $\\ddot{x}_k$ vs Time $t$')
#    plt.xlabel('Time $t$')
#    plt.ylabel('Acceleration $\\ddot{x}_k$')
#    plt.grid(True)
#    plt.legend()
#    plt.subplot(1, 4, 4)# Plot F_ext vs time
#    plt.plot(time_data, F_ext_val, label='External Force F$_{ext}$', color='red')
#    plt.title('External Force F$_{ext}$ vs Time $t$')
#    plt.xlabel('Time $t$')
#    plt.ylabel('F$_{ext}$ (t)')
#    plt.grid(True)
#    plt.legend()
#    plt.tight_layout()
#    plt.show()
#    plt.figure()
#    plt.scatter(x_dot_data,F1_th)
#    plt.show()






    #############  NN-CC WITHOUT SYMMETRIES  ######### this next part is for training NNS
    #for neurons in [150,20,50,100,200]:
    #neurons=2
    #neurons=100
    # Hyperparameters
    #Nlearning_rate = 1e-4
    #epochs_max = 20000
    #N_constraint = 1000
    
    # Convert data to tensors
    t_max =  np.max(t_simul)
    t_norm = t_simul / t_max
    #t_tensor = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1)
    t_tensor = torch.tensor(t_simul, dtype=torch.float32).unsqueeze(1).to(device)
    x_tensor = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1).to(device)
    x_dot_tensor = torch.tensor(x_dot_data, dtype=torch.float32).unsqueeze(1).to(device)
    x_ddot_tensor = torch.tensor(x_ddot_data, dtype=torch.float32).unsqueeze(1).to(device)
    F_ext_tensor = torch.tensor(F_ext_val, dtype=torch.float32).unsqueeze(1).to(device)

    #x_dot_constraint = torch.linspace(min(x_dot_data), max(x_dot_data), N_constraint).unsqueeze(1).to(device)
    #x_constraint     = torch.linspace(min(x_data),  max(x_data),     N_constraint).unsqueeze(1).to(device)
    

    #x_dot_constraint = torch.linspace(x_dot_data.min(), x_dot_data.max(), N_constraint, device=device).unsqueeze(1)
    #x_constraint     = torch.linspace(x_data.min(),     x_data.max(),     N_constraint, device=device).unsqueeze(1)
    
    #error_threshold = 1e-8
    # Step 2: Define the Neural Network
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
    # Step 3: Train the Neural Network
    # Hyperparameters
    #learning_rate = 1e-4
    #epochs_max = 5000
    #error_threshold = 1e-8
    criterion = nn.MSELoss()
    optimizer1 = optim.Adam(model1_nosym.parameters(), lr=learning_rate ) # , weight_decay=weight_decay) # working well with lr=1e-4
    optimizer2 = optim.Adam(model2_nosym.parameters(), lr=learning_rate ) #, weight_decay=weight_decay)

    #optimizer1 = optim.AdamW(model1_nosym.parameters(), lr=learning_rate , weight_decay=weight_decay)
    #optimizer2 = optim.AdamW(model2_nosym.parameters(), lr=learning_rate , weight_decay=weight_decay)
    
    #optimizer1 = optim.SGD(model1_nosym.parameters(), lr=learning_rate , momentum=momentum) # working well with lr=1e-1
    #optimizer2 = optim.SGD(model2_nosym.parameters(), lr=learning_rate , momentum=momentum)


    #if (device.type=='cuda'):
    #    from torch.utils.data import DataLoader, TensorDataset
    #    batch_size = 100  # set to 1 if you want to disable batching
    #    # Example: assume x_train and y_train are tensors on CPU/GPU
    #    dataset = TensorDataset(x_tensor, x_dot_tensor, x_ddot_tensor, F_ext_tensor)
    #    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    zero_input = torch.tensor([[0.0]], dtype=torch.float32).to(device)
    time_start=time.time()
    for epoch in range(epochs_max):
        model1_nosym.train()
        model2_nosym.train()
        # Forward pass
        #if device.type == 'cuda':
        #    # print("training with cuda")
        #    for x_b, x_dot_b, x_ddot_b, F_ext_b in dataloader:
        #        # Move batch to device (just in case DataLoader gave CPU tensors)
        #        #x_b = x_b.to(device)
        #        #x_dot_b = x_dot_b.to(device)
        #        #x_ddot_b = x_ddot_b.to(device)
        #        #F_ext_b = F_ext_b.to(device)
        #        
        #        # Forward pass
        #        predictions = x_ddot_b + model1(x_dot_b) + model2(x_b)
        #        loss = criterion(predictions, F_ext_b)
        #        
        #        # Constraint loss
        #        if apply_restriction:
        #            model2_at_zero = model2(zero_input)
        #            model1_at_zero = model1(zero_input)
        #            constraint_loss = lambda_penalty * ((model2_at_zero ** 2).mean() + (model1_at_zero ** 2).mean())
        #            total_loss = loss + constraint_loss
        #        else:
        #            total_loss = loss
        #        
        #        # Backward + step
        #        optimizer1.zero_grad()
        #        optimizer2.zero_grad()
        #        total_loss.backward()
        #        optimizer1.step()
        #        optimizer2.step()
        #
        #else: # in this case device.type == cpu
        predictions = x_ddot_tensor + model1_nosym(x_dot_tensor) + model2_nosym(x_tensor)
        loss = criterion(predictions, F_ext_tensor)
        # Add constraint: model2(0.0) ≈ 0
        restriction_loss=0.0*model1_nosym(zero_input)
        if(apply_restriction):
            #zero_input = torch.tensor([[0.0]], dtype=torch.float32).to(device)
            model2_at_zero = model2_nosym(zero_input)
            model1_at_zero = model1_nosym(zero_input)
            restriction_loss = lambda_penalty * ((model2_at_zero ** 2).mean() + (model1_at_zero ** 2).mean())  # squared penalty
            #constraint_loss = lambda_penalty * (model2_at_zero ** 2).mean()  # squared penalty
            total_loss = loss + restriction_loss
        else:
            total_loss = loss
        # ## uncomment for adding symmetries
        f1_loss=0.0
        f2_loss=0.0
        #if f1_symmetry=='even':
        #    f1_loss = lambda_penalty_symm * ((model1(x_dot_tensor)-model1(-x_dot_tensor))** 2).mean()
        #    total_loss = total_loss + f1_loss
        #elif f1_symmetry=='odd':
        #    f1_loss =  lambda_penalty_symm * ((model1(x_dot_tensor)+model1(-x_dot_tensor))** 2).mean()
        #    total_loss = total_loss + f1_loss
        #if f2_symmetry=='even':
        #    f2_loss =  lambda_penalty_symm * ((model2(x_tensor)-model2(-x_tensor))** 2).mean()
        #    total_loss = total_loss + f2_loss
        #elif f2_symmetry=='odd':
        #    f2_loss =  lambda_penalty_symm * ((model2(x_tensor)+model2(-x_tensor))** 2).mean()
        #    total_loss = total_loss + f2_loss
    
        #N_constraint = 1000
        #x_dot_constraint = torch.linspace(min(x_dot_data), max(x_dot_data), N_constraint).unsqueeze(1)
        #x_constraint     = torch.linspace(min(x_data),  max(x_data),     N_constraint).unsqueeze(1)
        # Apply F1 symmetry constraint over the fixed range
#        if f1_symmetry == 'even':
#            f1_loss = lambda_penalty_symm * ((model1(x_dot_constraint) - model1(-x_dot_constraint)) ** 2).mean()
#        elif f1_symmetry == 'odd':
#            f1_loss = lambda_penalty_symm * ((model1(x_dot_constraint) + model1(-x_dot_constraint)) ** 2).mean()
#        total_loss += f1_loss
#        
#        # Apply F2 symmetry constraint over the fixed range
#        if f2_symmetry == 'even':
#            f2_loss = lambda_penalty_symm * ((model2(x_constraint) - model2(-x_constraint)) ** 2).mean()
#        elif f2_symmetry == 'odd':
#            f2_loss = lambda_penalty_symm * ((model2(x_constraint) + model2(-x_constraint)) ** 2).mean()
#        total_loss += f2_loss        
        
        constraint_loss = restriction_loss + f1_loss + f2_loss
        # Backward pass and optimization
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        total_loss.backward()
        optimizer1.step()
        optimizer2.step()
        # Print the loss
        if epoch == 0 or (epoch + 1) % 100 == 0:
            #print(f"Epoch [{epoch+1}], Loss: {loss.item():.4e}, Constraint: {constraint_loss.item():.4e}, f1_loss: {f1_loss.item():.2e}, f2_loss: {f2_loss.item():.2e}")
            print(f"Epoch [{epoch+1}], Loss: {loss.item():.4e}, Constraints: {constraint_loss.item():.4e}")
            printF(f"Epoch [{epoch+1}], Loss: {loss.item():.4e}, Constraints: {constraint_loss.item():.4e}")
        if total_loss.item() < error_threshold:
            print(f"Training stopped at epoch {epoch}, Total Loss: {total_loss.item()}")
            printF(f"Training stopped at epoch {epoch}, Total Loss: {total_loss.item()}")
            break
    time_end=time.time()
    print(" ")
    print("End training NN-CC (without sym)")
    print("Neurons :",neurons)
    print(f"Training time: {time_end-time_start} seconds")
    printF(" ")
    printF("Neurons :",neurons)
    printF(f"Training time: {time_end-time_start} seconds")


    # After training move to cpu
    model1_nosym=model1_nosym.to('cpu')
    model2_nosym=model2_nosym.to('cpu')
    t_tensor = t_tensor.to('cpu')
    x_tensor = x_tensor.to('cpu')
    x_dot_tensor = x_dot_tensor.to('cpu')
    x_ddot_tensor = x_ddot_tensor.to('cpu')
    F_ext_tensor = F_ext_tensor.to('cpu')




    #############  NN-CC+sym WITH SYMMETRIES    ######### this next part is for training NNS
    #for neurons in [150,20,50,100,200]:
    #neurons=2
    #neurons=100
    # Hyperparameters
    learning_rate = 1e-4
    epochs_max = 20000
    N_constraint = 1000
    
    # Convert data to tensors
    t_max =  np.max(t_simul)
    t_norm = t_simul / t_max
    #t_tensor = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1)
    t_tensor = torch.tensor(t_simul, dtype=torch.float32).unsqueeze(1).to(device)
    x_tensor = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1).to(device)
    x_dot_tensor = torch.tensor(x_dot_data, dtype=torch.float32).unsqueeze(1).to(device)
    x_ddot_tensor = torch.tensor(x_ddot_data, dtype=torch.float32).unsqueeze(1).to(device)
    F_ext_tensor = torch.tensor(F_ext_val, dtype=torch.float32).unsqueeze(1).to(device)

    #x_dot_constraint = torch.linspace(min(x_dot_data), max(x_dot_data), N_constraint).unsqueeze(1).to(device)
    #x_constraint     = torch.linspace(min(x_data),  max(x_data),     N_constraint).unsqueeze(1).to(device)
    

    x_dot_constraint = torch.linspace(x_dot_data.min(), x_dot_data.max(), N_constraint, device=device).unsqueeze(1)
    x_constraint     = torch.linspace(x_data.min(),     x_data.max(),     N_constraint, device=device).unsqueeze(1)
    
    #error_threshold = 1e-8
    # Step 2: Define the Neural Network
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
    # Step 3: Train the Neural Network
    # Hyperparameters
    #learning_rate = 1e-4
    #epochs_max = 5000
    #error_threshold = 1e-8
    criterion = nn.MSELoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate )
    optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate )

    #optimizer1 = optim.AdamW(model1.parameters(), lr=learning_rate , weight_decay=weight_decay)
    #optimizer2 = optim.AdamW(model2.parameters(), lr=learning_rate , weight_decay=weight_decay)
    
    #optimizer1 = optim.SGD(model1.parameters(), lr=learning_rate , momentum=momentum) # working well with lr=1e-1
    #optimizer2 = optim.SGD(model2.parameters(), lr=learning_rate , momentum=momentum)


    #if (device.type=='cuda'):
    #    from torch.utils.data import DataLoader, TensorDataset
    #    batch_size = 100  # set to 1 if you want to disable batching
    #    # Example: assume x_train and y_train are tensors on CPU/GPU
    #    dataset = TensorDataset(x_tensor, x_dot_tensor, x_ddot_tensor, F_ext_tensor)
    #    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    zero_input = torch.tensor([[0.0]], dtype=torch.float32).to(device)
    time_start=time.time()
    for epoch in range(epochs_max):
        model1.train()
        model2.train()
        # Forward pass
        #if device.type == 'cuda':
        #    # print("training with cuda")
        #    for x_b, x_dot_b, x_ddot_b, F_ext_b in dataloader:
        #        # Move batch to device (just in case DataLoader gave CPU tensors)
        #        #x_b = x_b.to(device)
        #        #x_dot_b = x_dot_b.to(device)
        #        #x_ddot_b = x_ddot_b.to(device)
        #        #F_ext_b = F_ext_b.to(device)
        #        
        #        # Forward pass
        #        predictions = x_ddot_b + model1(x_dot_b) + model2(x_b)
        #        loss = criterion(predictions, F_ext_b)
        #        
        #        # Constraint loss
        #        if apply_restriction:
        #            model2_at_zero = model2(zero_input)
        #            model1_at_zero = model1(zero_input)
        #            constraint_loss = lambda_penalty * ((model2_at_zero ** 2).mean() + (model1_at_zero ** 2).mean())
        #            total_loss = loss + constraint_loss
        #        else:
        #            total_loss = loss
        #        
        #        # Backward + step
        #        optimizer1.zero_grad()
        #        optimizer2.zero_grad()
        #        total_loss.backward()
        #        optimizer1.step()
        #        optimizer2.step()
        #
        #else: # in this case device.type == cpu
        predictions = x_ddot_tensor + model1(x_dot_tensor) + model2(x_tensor)
        loss = criterion(predictions, F_ext_tensor)
        # Add constraint: model2(0.0) ≈ 0
        restriction_loss=0.0
        if(apply_restriction):
            #zero_input = torch.tensor([[0.0]], dtype=torch.float32).to(device)
            model2_at_zero = model2(zero_input)
            model1_at_zero = model1(zero_input)
            restriction_loss = lambda_penalty * ((model2_at_zero ** 2).mean() + (model1_at_zero ** 2).mean())  # squared penalty
            #constraint_loss = lambda_penalty * (model2_at_zero ** 2).mean()  # squared penalty
            total_loss = loss + restriction_loss
        else:
            total_loss = loss
        # ## uncomment for adding symmetries
        f1_loss=0.0
        f2_loss=0.0
        #if f1_symmetry=='even':
        #    f1_loss = lambda_penalty_symm * ((model1(x_dot_tensor)-model1(-x_dot_tensor))** 2).mean()
        #    total_loss = total_loss + f1_loss
        #elif f1_symmetry=='odd':
        #    f1_loss =  lambda_penalty_symm * ((model1(x_dot_tensor)+model1(-x_dot_tensor))** 2).mean()
        #    total_loss = total_loss + f1_loss
        #if f2_symmetry=='even':
        #    f2_loss =  lambda_penalty_symm * ((model2(x_tensor)-model2(-x_tensor))** 2).mean()
        #    total_loss = total_loss + f2_loss
        #elif f2_symmetry=='odd':
        #    f2_loss =  lambda_penalty_symm * ((model2(x_tensor)+model2(-x_tensor))** 2).mean()
        #    total_loss = total_loss + f2_loss
    
        #N_constraint = 1000
        #x_dot_constraint = torch.linspace(min(x_dot_data), max(x_dot_data), N_constraint).unsqueeze(1)
        #x_constraint     = torch.linspace(min(x_data),  max(x_data),     N_constraint).unsqueeze(1)
        # Apply F1 symmetry constraint over the fixed range
        if f1_symmetry == 'even':
            f1_loss = lambda_penalty_symm * ((model1(x_dot_constraint) - model1(-x_dot_constraint)) ** 2).mean()
        elif f1_symmetry == 'odd':
            f1_loss = lambda_penalty_symm * ((model1(x_dot_constraint) + model1(-x_dot_constraint)) ** 2).mean()
        total_loss += f1_loss
        
        # Apply F2 symmetry constraint over the fixed range
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
            #print(f"Epoch [{epoch+1}], Loss: {loss.item():.4e}, Constraint: {constraint_loss.item():.4e}, f1_loss: {f1_loss.item():.2e}, f2_loss: {f2_loss.item():.2e}")
            print(f"Epoch [{epoch+1}], Loss: {loss.item():.4e}, Constraints: {constraint_loss.item():.4e}")
            printF(f"Epoch [{epoch+1}], Loss: {loss.item():.4e}, Constraints: {constraint_loss.item():.4e}")
        if total_loss.item() < error_threshold:
            print(f"Training stopped at epoch {epoch}, Total Loss: {total_loss.item()}")
            printF(f"Training stopped at epoch {epoch}, Total Loss: {total_loss.item()}")
            break
    time_end=time.time()
    print(" ")
    print("End training NN-CC+sym")
    print("Neurons :",neurons)
    print(f"Training time: {time_end-time_start} seconds")
    printF(" ")
    printF("Neurons :",neurons)
    printF(f"Training time: {time_end-time_start} seconds")


    # After training move to cpu
    model1=model1.to('cpu')
    model2=model2.to('cpu')
    t_tensor = t_tensor.to('cpu')
    x_tensor = x_tensor.to('cpu')
    x_dot_tensor = x_dot_tensor.to('cpu')
    x_ddot_tensor = x_ddot_tensor.to('cpu')
    F_ext_tensor = F_ext_tensor.to('cpu')


    ##### Post_Training Extrapolation #####
    model1_retrain = copy.deepcopy(model1)
    model2_retrain = copy.deepcopy(model2)
    optimizer1_retrain = optim.Adam(model1_retrain.parameters(), lr=learning_rate )
    optimizer2_retrain = optim.Adam(model2_retrain.parameters(), lr=learning_rate )


    ## post-training augmentation step or self-supervised extrapolation regularization
    ## or pseudo-labeling  to stabilize beyond the training domain
     # Create extrapolated x values
    n_window = 50 # number of points to calculate the envelope around each edge
    range_interp = 0.2 # percentage of values near edges for obtaining the envelope
    range_extrap = 0.5  # percentage of extrapolated range with respect to total range
    n_extra = 50 # number of new data points for each direction
    deg_extrap = 1 # degree of polynomial extrapolation

    x_dot_data_min=min(x_dot_data)
    x_dot_data_max=max(x_dot_data)
    x_dot_data_size= max(x_dot_data)-min(x_dot_data)
    x_data_min=min(x_data)
    x_data_max=max(x_data)
    x_data_size= max(x_data)-min(x_data)
    x_interp_min_left = x_data_min
    x_interp_max_left = x_data_min+range_interp*x_data_size
    x_interp_min_right = x_data_max-range_interp*x_data_size
    x_interp_max_right = x_data_max
    x_dot_interp_min_left = x_dot_data_min
    x_dot_interp_max_left = x_dot_data_min+range_interp*x_dot_data_size
    x_dot_interp_min_right = x_dot_data_max-range_interp*x_dot_data_size
    x_dot_interp_max_right = x_dot_data_max
    x_extrap_min_left = x_data_min-range_extrap*x_data_size
    x_extrap_max_left = x_data_min
    x_extrap_min_right = x_data_max
    x_extrap_max_right = x_data_max+range_extrap*x_data_size
    x_dot_extrap_min_left = x_dot_data_min-range_extrap*x_dot_data_size
    x_dot_extrap_max_left = x_dot_data_min
    x_dot_extrap_min_right = x_dot_data_max
    x_dot_extrap_max_right = x_dot_data_max+range_extrap*x_dot_data_size


    model1.eval()
    with torch.no_grad():
        # LEFT SIDE: [a, a + 0.1*(b - a)]
        x_dot_interp_left = np.linspace(x_dot_interp_min_left,x_dot_interp_max_left, n_window)
        x_dot_interp_left_tensor = torch.tensor(x_dot_interp_left, dtype=torch.float32).unsqueeze(1)
        f1_interp_left = model1(x_dot_interp_left_tensor).squeeze().numpy()
        # RIGHT SIDE: [b - 0.1*(b - a), b]
        x_dot_interp_right = np.linspace(x_dot_interp_min_right, x_dot_interp_max_right, n_window)
        x_dot_interp_right_tensor = torch.tensor(x_dot_interp_right, dtype=torch.float32).unsqueeze(1)
        f1_interp_right = model1(x_dot_interp_right_tensor).squeeze().numpy()
    model2.eval()
    with torch.no_grad():
        # LEFT SIDE: [a, a + 0.1*(b - a)]
        x_interp_left = np.linspace(x_interp_min_left,x_interp_max_left, n_window)
        x_interp_left_tensor = torch.tensor(x_interp_left, dtype=torch.float32).unsqueeze(1)
        f2_interp_left = model2(x_interp_left_tensor).squeeze().numpy()
        # RIGHT SIDE: [b - 0.1*(b - a), b]
        x_interp_right = np.linspace(x_interp_min_right, x_interp_max_right, n_window)
        x_interp_right_tensor = torch.tensor(x_interp_right, dtype=torch.float32).unsqueeze(1)
        f2_interp_right = model2(x_interp_right_tensor).squeeze().numpy()


    f1_coeff_left = np.polyfit(x_dot_interp_left, f1_interp_left, deg=deg_extrap)
    f1_coeff_right = np.polyfit(x_dot_interp_right, f1_interp_right, deg=deg_extrap)
    f2_coeff_left = np.polyfit(x_interp_left, f2_interp_left, deg=deg_extrap)
    f2_coeff_right = np.polyfit(x_interp_right, f2_interp_right, deg=deg_extrap)
    # Extrapolate
    x_dot_extrap_left = torch.linspace(x_dot_extrap_min_left,x_dot_extrap_max_left, n_extra).unsqueeze(1)
    x_dot_extrap_right = torch.linspace(x_dot_extrap_min_right,x_dot_extrap_max_right, n_extra).unsqueeze(1)
    f1_extrap_left = np.polyval(f1_coeff_left, x_dot_extrap_left)
    f1_extrap_right = np.polyval(f1_coeff_right, x_dot_extrap_right)
    x_extrap_left = torch.linspace(x_extrap_min_left,x_extrap_max_left, n_extra).unsqueeze(1)
    x_extrap_right = torch.linspace(x_extrap_min_right,x_extrap_max_right, n_extra).unsqueeze(1)
    f2_extrap_left = np.polyval(f2_coeff_left, x_extrap_left)
    f2_extrap_right = np.polyval(f2_coeff_right, x_extrap_right)

    f1_extrap_left_tensor=torch.tensor(f1_extrap_left, dtype=torch.float32).unsqueeze(1).view(-1, 1)
    f1_extrap_right_tensor=torch.tensor(f1_extrap_right, dtype=torch.float32).unsqueeze(1).view(-1, 1)
    x_extrap_left_tensor=torch.tensor(x_extrap_left, dtype=torch.float32).unsqueeze(1).view(-1, 1)
    x_extrap_right_tensor=torch.tensor(x_extrap_right, dtype=torch.float32).unsqueeze(1).view(-1, 1)
    f2_extrap_left_tensor=torch.tensor(f2_extrap_left, dtype=torch.float32).unsqueeze(1).view(-1, 1)
    f2_extrap_right_tensor=torch.tensor(f2_extrap_right, dtype=torch.float32).unsqueeze(1).view(-1, 1)
    x_dot_extrap_left_tensor=torch.tensor(x_dot_extrap_left, dtype=torch.float32).unsqueeze(1).view(-1, 1)
    x_dot_extrap_right_tensor=torch.tensor(x_dot_extrap_right, dtype=torch.float32).unsqueeze(1).view(-1, 1)



    x_dot_tensor_augmented = torch.cat([x_dot_tensor, x_dot_extrap_left_tensor, x_dot_extrap_right_tensor], dim=0)
    x_tensor_augmented = torch.cat([x_tensor, x_extrap_left_tensor, x_extrap_right_tensor], dim=0)

    model1.eval()
    with torch.no_grad():
        f1 = model1(x_dot_tensor).squeeze().numpy()
    model2.eval()
    with torch.no_grad():
        f2 = model2(x_tensor).squeeze().numpy()
    f1_tensor = torch.tensor(f1, dtype=torch.float32).unsqueeze(1)
    f2_tensor = torch.tensor(f2, dtype=torch.float32).unsqueeze(1)

    f1_tensor_augmented = torch.cat([f1_tensor, f1_extrap_left_tensor, f1_extrap_right_tensor], dim=0)
    f2_tensor_augmented = torch.cat([f2_tensor, f2_extrap_left_tensor, f2_extrap_right_tensor], dim=0)


    epochs_max_retrain=1000


    for epoch in range(epochs_max_retrain):
        model1_retrain.train()
        # Forward pass
        m1 = model1_retrain(x_dot_tensor_augmented)
        loss1 = criterion(f1_tensor_augmented, m1)
        # Add constraint: model2(0.0) ≈ 0
        # Backward pass and optimization
        optimizer1_retrain.zero_grad()
        loss1.backward()
        optimizer1_retrain.step()
        # Print the loss
        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}], Loss f1: {loss1.item():.4e}")
            printF(f"Epoch [{epoch+1}], Loss f1: {loss1.item():.4e}")
        if loss1.item() < error_threshold:
            print(f"Training stopped at epoch {epoch}, Loss: {loss1.item()}")
            printF(f"Training stopped at epoch {epoch}, Loss: {loss1.item()}")
            break

    for epoch in range(epochs_max_retrain):
        model2_retrain.train()
        # Forward pass
        m2 = model2_retrain(x_tensor_augmented)
        loss2 = criterion(f2_tensor_augmented, m2)
        # Add constraint: model2(0.0) ≈ 0
        # Backward pass and optimization
        optimizer2_retrain.zero_grad()
        loss2.backward()
        optimizer2_retrain.step()
        # Print the loss
        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}], Loss f2: {loss2.item():.4e}")
            printF(f"Epoch [{epoch+1}], Loss f2: {loss2.item():.4e}")
        if loss2.item() < error_threshold:
            print(f"Training stopped at epoch {epoch}, Loss: {loss2.item()}")
            printF(f"Training stopped at epoch {epoch}, Loss: {loss2.item()}")
            break




    ## Training loop
    #for epoch in range(epochs_max):
    #    model1.train()
    #    model2.train()
    #    # Forward pass
    #    predictions = x_ddot_tensor+model1(x_dot_tensor)+model2(x_tensor)
    #    loss = criterion(predictions, F_ext_tensor)
    #    # Backward pass and optimization
    #    optimizer1.zero_grad()
    #    optimizer2.zero_grad()
    #    loss.backward()
    #    optimizer1.step()
    #    optimizer2.step()
    #    # Print the loss each 100 steps
    #    if epoch == 0 or (epoch + 1) % 100 == 0:
    #    #if (epoch + 1) % 100 == 0:
    #        print(f'Epoch [{epoch+1}], Loss: {loss.item():.16f}')
    #    # Check if the loss is below the threshold
    #    if loss.item() < error_threshold:
    #        print(f'Training stopped at epoch {epoch}, Loss: {loss.item()}')
    #        break




    # Evaluate the model by setting x from x_data_min to x_data_max
    x_dot_lin = np.linspace(x_dot_extrap_min_left,x_dot_extrap_max_right, Nsimul)
    x_dot_lin_tensor = torch.tensor(x_dot_lin, dtype=torch.float32).unsqueeze(1) # .view(-1, 1)
    x_dot_lin_data = np.linspace(x_dot_data_min,x_dot_data_max, Nsimul)
    x_dot_lin_tensor_data = torch.tensor(x_dot_lin_data, dtype=torch.float32).unsqueeze(1) # .view(-1, 1)
    model1.eval()
    model1_retrain.eval()
    with torch.no_grad():
        predicted_lin_F1_data = model1(x_dot_lin_tensor_data).numpy()
        predicted_lin_F1 = model1(x_dot_lin_tensor).numpy()
        predicted_lin_F1_retrain = model1_retrain(x_dot_lin_tensor).numpy()
        predicted_lin_F1_retrain_data = model1_retrain(x_dot_lin_tensor_data).numpy()
        predicted_lin_F1_data_nosym = model1_nosym(x_dot_lin_tensor_data).numpy()
        predicted_lin_F1_nosym = model1(x_dot_lin_tensor).numpy()
        
    #x_data_size = max(x_data)-min(x_data)
    #x_data_min = min(x_data)-x_data_size*0.50
    #x_data_max = max(x_data)+x_data_size*0.50
    x_lin = np.linspace(x_extrap_min_left, x_extrap_max_right, Nsimul)
    x_lin_tensor = torch.tensor(x_lin, dtype=torch.float32).unsqueeze(1) # .view(-1, 1)
    x_lin_data = np.linspace(x_data_min, x_data_max, Nsimul)
    x_lin_tensor_data = torch.tensor(x_lin_data, dtype=torch.float32).unsqueeze(1)
    model2.eval()
    model2_retrain.eval()
    with torch.no_grad():
        predicted_lin_F2_data = model2(x_lin_tensor_data).numpy()
        predicted_lin_F2 = model2(x_lin_tensor).numpy()
        predicted_lin_F2_retrain = model2_retrain(x_lin_tensor).numpy()
        predicted_lin_F2_retrain_data = model2_retrain(x_lin_tensor_data).numpy()
        predicted_lin_F2_nosym = model2_nosym(x_lin_tensor_data).numpy()
    # Step 4: Evaluate the Model
    model1_retrain.eval()
    with torch.no_grad():
        predicted_F1_retrain = model1_retrain(x_dot_tensor).numpy()
        predicted_F2_retrain = model2_retrain(x_tensor).numpy()
    #    predicted_xddot=  (model3(t_tensor)-model1(x_tensor)*x_dot_tensor-model2(x_tensor))
        predicted_xddot=  (F_ext_tensor-model1_retrain(x_dot_tensor)-model2_retrain(x_tensor))

    # Plot the results
    plt.plot(x_data, x_ddot_data, label='Actual xddot\"',color='black', linestyle='dashed',linewidth=2)
    plt.plot(x_data, predicted_xddot, label='Predicted xddot\"',color='blue', linestyle='dashed',linewidth=2)
    plt.plot(x_data, predicted_F1_retrain, label='Predicted (from input x) -$\\ddot{x}$', color='red', linestyle='dashed',linewidth=2)
    plt.plot(x_lin, predicted_lin_F1_retrain, label='Predicted (linear range) -$\\ddot{x_{lin}}$', color='green', linestyle='dashed',linewidth=2)#, marker='o',markersize=3, alpha=0.1)
    #plt.plot(x, predicted_acc, label='Predicted (from input x) -$\\ddot{x}$', color='red', linestyle='dashed', marker='x',markersize=4, alpha=0.7)
    plt.legend()
    plt.show()
    model2_retrain.eval()
    with torch.no_grad():
        bias_correction = model2_retrain(torch.tensor([[0.0]], dtype=torch.float32)).numpy()
        bias_correction = model2(torch.tensor([[0.0]], dtype=torch.float32)).numpy()
        #bias_correction = model1(torch.tensor([[0.0]], dtype=torch.float32)).numpy()
    #bias_correction=model2(torch.tensor(0.0, dtype=torch.float32).unsqueeze(1)).numpy()
    # Visualización de resultados
    plt.figure(figsize=(15, 5))
    # Plot position x_k vs time
    plt.subplot(1, 2, 1)
    #plt.plot(x_dot_data, F1_th, "-", label="Función F1 teórica", lw=2)
    plt.scatter(x_dot_data, F1_th,color='blue', label="Función F1 teórica", lw=1,s=5)
    plt.plot(x_dot_lin_tensor,predicted_lin_F1_retrain+bias_correction,"--",color='orange', label="NN f1($\\dot{x}$)", lw=2)
    plt.plot(x_dot_lin_tensor, f1_fit(x_dot_lin_tensor)+f2_fit(0), 'r', label='LS $f_1$($\\dot{x}$)')
    #plt.plot(x_tensor, predicted_f, label="Posición (datos reales)", lw=2)
    plt.xlabel("$\\dot{x}$")
    plt.ylabel("F1")
    F1_th_range=np.max(F1_th)-np.min(F1_th)
    plt.ylim(np.min(F1_th)-F1_th_range*0.5,np.max(F1_th)+F1_th_range*0.5)
    plt.legend()
    plt.title("Comparación $F_1(\\dot{x})$")
    plt.subplot(1,2,2)
    plt.plot(x_data, F2_th, "-", color='blue', label="Función F2 teórica", lw=2)
    plt.plot(x_lin_tensor,predicted_lin_F2_retrain-bias_correction,"--",color='orange', label="NN $f_1$(x)", lw=2)
    plt.plot(x_lin_tensor, f2_fit(x_lin_tensor)-f2_fit(0), 'r', label='LS $f_2(x)$')
    #plt.plot(x_tensor, predicted_g, label="Posición (datos reales)", lw=2)
    plt.xlabel("x")
    plt.ylabel("$F_2$")
    F2_th_range=np.max(F2_th)-np.min(F2_th)
    plt.ylim(np.min(F2_th)-F2_th_range*0.5,np.max(F2_th)+F2_th_range*0.5)
    plt.legend()
    plt.title("Comparación $F_2$(x)")
    #plt.subplot(1,3,3)
    ##plt.plot(t_eval, F_ext, "-", label="F$_{ext}$ teórica", lw=2)
    #plt.plot(t_simul, x_tensor, "-", label="x$_{data}$ teórica", lw=2)
    #plt.plot(t_simul, x_data, "-", label="x$_{data}$ NN eval DB", lw=2)
    ##plt.plot(t_eval,predicted_T_ext,"--", label="F$_{ext}$ estimada", lw=2)
    ##plt.plot(x_tensor, predicted_f, label="Posición (datos reales)", lw=2)
    #plt.xlabel("x$_{ext}$")
    #plt.ylabel("Tiempo (s)")
    #plt.legend()
    #plt.title("Comparision of x(t)")
    plt.show()




    ####!pip install pysr

    #from pysr import PySRRegressor
    #import multiprocessing
    #n_cores = multiprocessing.cpu_count()
    print(" ")
    print("now doing post-SR to NN-CC+sym")
    Xdot = x_dot_lin_data.reshape(-1, 1)
    Xdotpred = x_dot_lin.reshape(-1, 1)
    correction_extra=0.0
    start=time.time()
    y = predicted_lin_F1_data + bias_correction + correction_extra
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
    model_f1SR.fit(Xdot, y)
    print(model_f1SR)
    best_equation = model_f1SR.get_best()
    y_pred = model_f1SR.predict(Xdot)
    pred_f1SR = model_f1SR.predict(Xdotpred)
    plt.figure()
    plt.plot(x_dot_lin_data, y, label="Identified NN-CC", lw=2)
    plt.plot(x_dot_lin_data, y_pred, "--", label="PySR fit", lw=2)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f1(x')")
    plt.title("PySR Symbolic Regression Fit")
    plt.grid(True)
    plt.show()

    X = x_lin_data.reshape(-1, 1)
    Xpred = x_lin.reshape(-1, 1)
    y = predicted_lin_F2_data - bias_correction - correction_extra
    model_f2SR = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*"],
        #unary_operators=["log", "abs", "sqrt"],
        #unary_operators=["sin", "cos", "exp", "log", "abs", "sqrt"],
        loss="loss(x, y) = (x - y)^2",
        maxsize=20,
        populations=20,
        #verbosity=1,
    )
    model_f2SR.fit(X, y)
    print(model_f2SR)
    printF(model_f2SR)
    best_equation = model_f2SR.get_best()
    y_pred = model_f2SR.predict(X)
    end = time.time()  
    elapsed = end - start
    print('End doing post-SR to NN-CC+sym . i.e. NN-CC+sym+post-SR')
    print(f"Training finished in {elapsed:.3f} seconds")
    printF(f"Training finished in {elapsed:.3f} seconds")

    pred_f2SR = model_f2SR.predict(Xpred)
    plt.figure()
    plt.plot(x_lin_data, y, label="Identified NN-CC", lw=2)
    plt.plot(x_lin_data, y_pred, "--", label="PySR fit", lw=2)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f2(x)")
    plt.title("PySR Symbolic Regression Fit")
    plt.grid(True)
    plt.show()


    #############################################################################################################################

    with torch.no_grad():
        predicted_lin_F1_data = model1(x_dot_lin_tensor_data).numpy()
        predicted_lin_F1 = model1(x_dot_lin_tensor).numpy()
        predicted_lin_F1_retrain = model1_retrain(x_dot_lin_tensor).numpy()
        predicted_lin_F1_retrain_data = model1_retrain(x_dot_lin_tensor_data).numpy()
        predicted_lin_F1_data_nosym = model1_nosym(x_dot_lin_tensor_data).numpy()
        predicted_lin_F1_nosym = model1(x_dot_lin_tensor).numpy()

    print("max_x=",max(x_lin_tensor_data))
    print("min_x=",min(x_lin_tensor_data))

    #x_data_size = max(x_data)-min(x_data)
    #x_data_min = min(x_data)-x_data_size*0.50
    #x_data_max = max(x_data)+x_data_size*0.50
    x_lin = np.linspace(x_extrap_min_left, x_extrap_max_right, Nsimul)
    x_lin_tensor = torch.tensor(x_lin, dtype=torch.float32).unsqueeze(1) # .view(-1, 1)
    x_lin_data = np.linspace(x_data_min, x_data_max, Nsimul)
    x_lin_tensor_data = torch.tensor(x_lin_data, dtype=torch.float32).unsqueeze(1)
    model2.eval()
    model2_retrain.eval()
    with torch.no_grad():
        predicted_lin_F2_data = model2(x_lin_tensor_data).numpy()
        predicted_lin_F2 = model2(x_lin_tensor).numpy()
        predicted_lin_F2_retrain = model2_retrain(x_lin_tensor).numpy()
        predicted_lin_F2_retrain_data = model2_retrain(x_lin_tensor_data).numpy()
        predicted_lin_F2_data_nosym = model2_nosym(x_lin_tensor_data).numpy()
        predicted_lin_F2_nosym = model2_nosym(x_lin_tensor).numpy()


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


    fig, axs = plt.subplots(2, 1, figsize=(5, 10), sharex=False)
    # First subplot: F1
    axs[0].axvspan(np.min(x_dot_data), np.max(x_dot_data), color='lightgray', alpha=0.6, zorder=0.5)
    #uncomment if required
    #axs[0].scatter(x_dot_data, F1_th, color='blue', label="Training DS", lw=1, s=5, marker='s')
    correction_extra=0.0#-0.1
    correction_SR=0.0#0.1
    axs[0].plot(x_dot_extrap_left, f1_extrap_left + bias_correction+correction_extra, "-", color='violet', label="NN-CC (lin_extrap)", lw=2.5)
    axs[0].plot(x_dot_lin_data, predicted_lin_F1_data + bias_correction+correction_extra, "-", color='violet', lw=2.5)
    axs[0].plot(x_dot_extrap_right, f1_extrap_right + bias_correction+correction_extra, "-", color='violet', lw=2.5)

    axs[0].plot(x_dot_lin_tensor, predicted_lin_F1_retrain + bias_correction+correction_extra, "-", color='orange', label="NN-CC (retrained lin-extrap)", lw=2.5)
    axs[0].plot(x_dot_lin, (pred_f1SR+ bias_correction+correction_extra+correction_SR).flatten(), "-", color='magenta', label="NN-CC-SR", lw=2.5)

    #axs[0].plot(x_dot_lin_tensor, predicted_lin_F1 + bias_correction, "-", color='violet', label="NN-CC ()", lw=2)
    axs[0].plot(x_dot_lin_tensor, f1_fit(x_dot_lin_tensor)-f1_fit(0) , 'r', label='Poly-CC')
    axs[0].plot(x_dot_lin_tensor.numpy(), f1_sindy(x_dot_lin_tensor.numpy()), 'blue', label='SINDy-CC')
#-0.1    axs[0].plot(x_dot_lin_tensor, f1_sindy(x_dot_lin_tensor) + f2_sindy(0), 'darkgreen', label='Sindy')
    axs[0].plot(x_dot_lin_tensor, F1(x_dot_lin_tensor) , "--", color='black', label="Theoretical", lw=1,dashes=(10,5))
    axs[0].set_xlabel("$\\dot{x}$", fontsize=24)
    axs[0].set_ylabel("$f_3(\\dot{x})$", fontsize=24)
    F1_th_range = np.max(F1_th) - np.min(F1_th)
    axs[0].set_ylim(np.min(F1_th) - F1_th_range * 0.5, np.max(F1_th) + F1_th_range * 0.5)
    axs[0].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1))
    custom_ticks(axs[0], major_x_interval=1, major_y_interval=1, minor_x_interval=0.5, minor_y_interval=0.25)
    #axs[0].tick_params(axis='both', direction='in', which='major', length=8, width=1.5, labelsize=16)
    #axs[0].tick_params(axis='both', direction='in', which='minor', length=4, width=1)
    #axs[0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    #axs[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[0].text(0.96, 0.04, "(a)", transform=axs[0].transAxes, fontsize=24, va='bottom', ha='right')
    # Second subplot: F2
    axs[1].axvspan(np.min(x_data), np.max(x_data), color='lightgray', alpha=0.6, zorder=0)
    #uncomment if required
    #axs[1].scatter(x_data, F2_th,color='blue', label="Training DS", lw=1, s=5,marker='s')
    axs[1].plot(x_extrap_left, f2_extrap_left - bias_correction-correction_extra, "-", color='violet', label="NN-CC (lin-extrap)", lw=2.5)
    axs[1].plot(x_lin_data, predicted_lin_F2_data - bias_correction-correction_extra, "-", color='violet', lw=2.5)
    axs[1].plot(x_extrap_right, f2_extrap_right - bias_correction-correction_extra, "-", color='violet', lw=2.5)

    axs[1].plot(x_lin, (pred_f2SR - bias_correction-correction_extra-correction_SR).flatten(), "-", color='magenta', label='NN-CC-SR', lw=2.5)
    axs[1].plot(x_lin_tensor, predicted_lin_F2_retrain - bias_correction-correction_extra, "-", color='orange', label='NN-CC (retrained lin-extrap)', lw=2.5)
    axs[1].plot(x_lin_tensor, f2_fit(x_lin_tensor) + f1_fit(0), 'r', label='Poly-CC')
    axs[1].plot(x_lin_tensor.numpy(), f2_sindy(x_lin_tensor.numpy()), 'blue', label='SINDy-CC')
#+0.1    axs[1].plot(x_lin_tensor, f2_sindy(x_lin_tensor) - f2_sindy(0), 'darkgreen', label='Sindy')
    axs[1].plot(x_lin_tensor, F2(x_lin_tensor) , "--", color='black', label="Theoretical", lw=1,dashes=(10,5))
    axs[1].set_xlabel("$x$", fontsize=24)
    axs[1].set_ylabel("$f_4(x)$", fontsize=24)
    F2_th_range = np.max(F2_th) - np.min(F2_th)
    axs[1].set_ylim(np.min(F2_th) - F2_th_range * 0.5, np.max(F2_th) + F2_th_range * 0.5)
    axs[1].legend(fontsize=14)
    #axs[1].tick_params( axis='both', direction='in', which='major', length=8, width=1.5, labelsize=16)
    #axs[1].tick_params( axis='both', direction='in', which='minor', length=4, width=1)
    #axs[1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    #axs[1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    custom_ticks(axs[1], major_x_interval=1, major_y_interval=1, minor_x_interval=0.5, minor_y_interval=0.5)
    axs[1].text(0.96, 0.04, "(b)", transform=axs[1].transAxes, fontsize=24, va='bottom', ha='right')
    plt.tight_layout()
    # Save to a specific folder
    folder_path = output_path #"/content/drive/My Drive/Colab Notebooks/Plots"
    os.makedirs(folder_path, exist_ok=True)
    file_name = "CC_duffing.pdf"
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    print(f"Saved to: {file_path}")



    print('bias correction=',bias_correction)
    print('extra correction=',correction_extra)




    fig, axs = plt.subplots(2, 1, figsize=(5, 10), sharex=False)
    # First subplot: F1
    axs[0].axvspan(np.min(x_dot_data), np.max(x_dot_data), color='lightgray', alpha=0.6, zorder=0.5)
    #uncomment if required
    #axs[0].scatter(x_dot_data, F1_th, color='blue', label="Training DS", lw=1, s=5, marker='s')
    axs[0].plot(x_dot_lin_tensor, predicted_lin_F1 + bias_correction, "-", color='red', label="NN-CC (without-ext)", lw=2)
    axs[0].plot(x_dot_lin_tensor, predicted_lin_F1_retrain + bias_correction, "-", color='orange', label="NN-CC (lin-ext)", lw=2)
    axs[0].plot(x_dot_lin_tensor, F1(x_dot_lin_tensor) + bias_correction, "--", color='black', label="Theoretical", lw=1,dashes=(10,5))
    #axs[0].plot(x_dot_lin_tensor, predicted_lin_F1_retrain + bias_correction, "--", color='black', label="Theoretical", lw=2)
    #axs[0].plot(x_dot_lin_tensor, f1_fit(x_dot_lin_tensor) , 'r', label='Poly-CC')
    #axs[0].plot(x_dot_lin_tensor.numpy(), f1_sindy(x_dot_lin_tensor.numpy()), 'darkgreen', label='SINDy')
#-0.1    axs[0].plot(x_dot_lin_tensor, f1_sindy(x_dot_lin_tensor) + f2_sindy(0), 'darkgreen', label='Sindy')
    axs[0].set_xlabel("$\\dot{x}$", fontsize=24)
    axs[0].set_ylabel("$f_3(\\dot{x})$", fontsize=24)
    F1_th_range = np.max(F1_th) - np.min(F1_th)
    axs[0].set_ylim(np.min(F1_th) - F1_th_range * 0.5, np.max(F1_th) + F1_th_range * 0.5)
    axs[0].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1),labelspacing=0.15)
    custom_ticks(axs[0], major_x_interval=1, major_y_interval=1, minor_x_interval=0.5, minor_y_interval=0.25)
    #axs[0].tick_params(axis='both', direction='in', which='major', length=8, width=1.5, labelsize=16)
    #axs[0].tick_params(axis='both', direction='in', which='minor', length=4, width=1)
    #axs[0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    #axs[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[0].text(0.96, 0.04, "(a)", transform=axs[0].transAxes, fontsize=24, va='bottom', ha='right')
    # Second subplot: F2
    axs[1].axvspan(np.min(x_data), np.max(x_data), color='lightgray', alpha=0.6, zorder=0)
    #uncomment if required
    #axs[1].scatter(x_data, F2_th,color='blue', label="Training DS", lw=1, s=5,marker='s')
    axs[1].plot(x_lin_tensor, predicted_lin_F2 - bias_correction, "-", color='red', label='NN-CC (without-ext)', lw=2)
    axs[1].plot(x_lin_tensor, predicted_lin_F2_retrain - bias_correction, "-", color='orange', label='NN-CC (lin-ext)', lw=2)
    axs[1].plot(x_lin_tensor, F2(x_lin_tensor) + bias_correction, "--", color='black', label="Theoretical", lw=1,dashes=(10,5))
    #axs[1].plot(x_lin_tensor, predicted_lin_F2_retrain + bias_correction, "--", color='black', label="Theoretical", lw=2)
    #axs[1].plot(x_lin_tensor, f2_fit(x_lin_tensor) - f2_fit(0), 'r', label='Poly-CC')
    #axs[1].plot(x_lin_tensor.numpy(), f2_sindy(x_lin_tensor.numpy()), 'darkgreen', label='SINDy')
#+0.1    axs[1].plot(x_lin_tensor, f2_sindy(x_lin_tensor) - f2_sindy(0), 'darkgreen', label='Sindy')
    axs[1].set_xlabel("$x$", fontsize=24)
    axs[1].set_ylabel("$f_4(x)$", fontsize=24)
    F2_th_range = np.max(F2_th) - np.min(F2_th)
    axs[1].set_ylim(np.min(F2_th) - F2_th_range * 0.5, np.max(F2_th) + F2_th_range * 0.5)
    #axs[1].legend(fontsize=14)
    axs[1].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1),labelspacing=0.15)
    #axs[1].tick_params( axis='both', direction='in', which='major', length=8, width=1.5, labelsize=16)
    #axs[1].tick_params( axis='both', direction='in', which='minor', length=4, width=1)
    #axs[1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    #axs[1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    custom_ticks(axs[1], major_x_interval=1, major_y_interval=1, minor_x_interval=0.5, minor_y_interval=0.5)
    axs[1].text(0.96, 0.04, "(b)", transform=axs[1].transAxes, fontsize=24, va='bottom', ha='right')
    plt.tight_layout()
    # Save to a specific folder
    folder_path = output_path #"/content/drive/My Drive/Colab Notebooks/Plots"
    os.makedirs(folder_path, exist_ok=True)
    file_name = "CC_duffing_appendix.pdf"
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    print(f"Saved to: {file_path}")




    ##########################################
    # Evaluation of the f1 and f2 fucntions
    ##########################################
    predicted_lin_F1_data_SR=f1_fun_sr(x_dot_lin_data).flatten()
    predicted_lin_F2_data_SR=f2_fun_sr(x_lin_data).flatten()
    predicted_lin_F1_data_parametric=f1_param(x_dot_lin_data).flatten()
    predicted_lin_F2_data_parametric=f2_param(x_lin_data).flatten()
    predicted_lin_F1_data_th=F1(x_dot_lin_data).flatten()
    predicted_lin_F2_data_th=F2(x_lin_data).flatten()
    predicted_lin_F1_data_NN=predicted_lin_F1_data.flatten()
    predicted_lin_F2_data_NN=predicted_lin_F2_data.flatten()
    predicted_lin_F1_data_NN_nosym=predicted_lin_F1_data_nosym.flatten()
    predicted_lin_F2_data_NN_nosym=predicted_lin_F2_data_nosym.flatten()
    predicted_lin_F1_data_NN_retrain=predicted_lin_F1_retrain_data.flatten()
    predicted_lin_F2_data_NN_retrain=predicted_lin_F2_retrain_data.flatten()
    predicted_lin_F1_data_Sindy=f1_sindy(x_dot_lin_data)
    predicted_lin_F2_data_Sindy=f2_sindy(x_lin_data)
    predicted_lin_F1_data_LS=f1_fit(x_dot_lin_data)+f2_fit(0.0)
    predicted_lin_F2_data_LS=f2_fit(x_lin_data)-f2_fit(0.0)
    predicted_lin_F1_data_NN_SR=model_f1SR.predict(x_dot_lin_data.reshape(-1,1))
    predicted_lin_F2_data_NN_SR=model_f2SR.predict(x_lin_data.reshape(-1,1))





    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    rmse_values = {
        "noise_perc_th":noise_percentage_th,
        "noise_perc":noise_percentage,
        "SNR_dB":SNR_dB,
        "NN_f1": rmse(predicted_lin_F1_data_th, predicted_lin_F1_data_NN),
        "NN_f2": rmse(predicted_lin_F2_data_th, predicted_lin_F2_data_NN),
        "NN_retrain_f1": rmse(predicted_lin_F1_data_th, predicted_lin_F1_data_NN_retrain),
        "NN_retrain_f2": rmse(predicted_lin_F2_data_th, predicted_lin_F2_data_NN_retrain),
        "SINDy_f1": rmse(predicted_lin_F1_data_th, predicted_lin_F1_data_Sindy),
        "SINDy_f2": rmse(predicted_lin_F2_data_th, predicted_lin_F2_data_Sindy),
        "LS_f1": rmse(predicted_lin_F1_data_th, predicted_lin_F1_data_LS),
        "LS_f2": rmse(predicted_lin_F2_data_th, predicted_lin_F2_data_LS),
        "NN_SR_f1": rmse(predicted_lin_F1_data_th, predicted_lin_F1_data_NN_SR),
        "NN_SR_f2": rmse(predicted_lin_F2_data_th, predicted_lin_F2_data_NN_SR),
        "Param_f1": rmse(predicted_lin_F1_data_th, predicted_lin_F1_data_parametric),
        "Param_f2": rmse(predicted_lin_F2_data_th, predicted_lin_F2_data_parametric),
        "SR_f1": rmse(predicted_lin_F1_data_th, predicted_lin_F1_data_SR),
        "SR_f2": rmse(predicted_lin_F2_data_th, predicted_lin_F2_data_SR),
    }


    print(f"NN-CC f1 = {rmse(predicted_lin_F1_data_th, predicted_lin_F1_data_NN_nosym)}")
    print(f"NN-CC f2 = { rmse(predicted_lin_F2_data_th, predicted_lin_F2_data_NN_nosym)}")

    print(f"NN-CC+sym f1 = {rmse(predicted_lin_F1_data_th, predicted_lin_F1_data_NN)}")
    print(f"NN-CC+sym f2 = { rmse(predicted_lin_F2_data_th, predicted_lin_F2_data_NN)}")

    print(f"NN-CC+sym+post-SR f1 = {rmse(predicted_lin_F1_data_th, predicted_lin_F1_data_NN_SR)}")
    print(f"NN-CC+sym+post-SR f2 = { rmse(predicted_lin_F2_data_th, predicted_lin_F2_data_NN_SR)}")

    fname = "rmse_results_for_f1_and_f2.txt"
    header = "# " + " ".join(rmse_values.keys())

    #line = " ".join(f"{v:.2e}" for v in rmse_values.values())
    line = " ".join(f"{float(np.ravel(v)[0]):.2e}" for v in rmse_values.values())

    mode = "a" if os.path.exists(fname) else "w"
    with open(fname, mode) as f:
        if mode == "w":
            f.write(header + "\n")
            print(header)
            #printF(header)
        f.write(line + "\n")
        #print(line)

    print(header)
    print(line)
    #printF(line)



    # --- Plot for F1 ---
    plt.figure()
    plt.plot(x_dot_lin_data, predicted_lin_F1_data_th, label="Theoretical", linewidth=2)
    plt.plot(x_dot_lin_data, predicted_lin_F1_data_NN, label="NN",linewidth=3)
    plt.plot(x_dot_lin_data, predicted_lin_F1_data_NN_retrain, label="NN retrain")
    plt.plot(x_dot_lin_data, predicted_lin_F1_data_Sindy, label="SINDy")
    plt.plot(x_dot_lin_data, predicted_lin_F1_data_LS, label="Poly")
    plt.plot(x_dot_lin_data, predicted_lin_F1_data_NN_SR, label="NN+SR")
    plt.plot(x_dot_lin_data, predicted_lin_F1_data_SR, label="SR")

    plt.xlabel("x_dot")
    plt.ylabel("F1")
    plt.title("Comparison of F1 models")
    plt.legend()
    plt.grid(True)

    # --- Plot for F2 ---
    plt.figure()
    plt.plot(x_lin_data, predicted_lin_F2_data_th, label="Theoretical", linewidth=2)
    plt.plot(x_lin_data, predicted_lin_F2_data_NN, label="NN",linewidth=3)
    plt.plot(x_lin_data, predicted_lin_F2_data_NN_retrain, label="NN retrain")
    plt.plot(x_lin_data, predicted_lin_F2_data_Sindy, label="SINDy")
    plt.plot(x_lin_data, predicted_lin_F2_data_LS, label="Poly")
    plt.plot(x_lin_data, predicted_lin_F2_data_NN_SR, label="NN+SR")
    plt.plot(x_lin_data, predicted_lin_F2_data_SR, label="SR")

    plt.xlabel("x")
    plt.ylabel("F2")
    plt.title("Comparison of F2 models")
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


    fig, axs = plt.subplots(2, 1, figsize=(5, 10), sharex=False)
    # First subplot: F1
    axs[0].axvspan(np.min(x_dot_data), np.max(x_dot_data), color='lightgray', alpha=0.6, zorder=0.5)
    #uncomment if required
    #axs[0].scatter(x_dot_data, F1_th, color='blue', label="Training DS", lw=1, s=5, marker='s')
    correction_extra=0.0#-0.1
    correction_SR=0.0#0.1
    axs[0].plot(x_dot_extrap_left, f1_extrap_left + bias_correction+correction_extra, "-", color='violet', label="NN-CC (lin_extrap)", lw=2.5)
    axs[0].plot(x_dot_lin_data, predicted_lin_F1_data + bias_correction+correction_extra, "-", color='violet', lw=2.5)
    axs[0].plot(x_dot_extrap_right, f1_extrap_right + bias_correction+correction_extra, "-", color='violet', lw=2.5)

    axs[0].plot(x_dot_lin_tensor, predicted_lin_F1_retrain + bias_correction+correction_extra, "-", color='orange', label="NN-CC (retrained lin-extrap)", lw=2.5)
    axs[0].plot(x_dot_lin, (pred_f1SR+ bias_correction+correction_extra+correction_SR).flatten(), "-", color='magenta', label="NN-CC-SR", lw=2.5)

    #axs[0].plot(x_dot_lin_tensor, predicted_lin_F1 + bias_correction, "-", color='violet', label="NN-CC ()", lw=2)
    axs[0].plot(x_dot_lin_tensor, f1_fit(x_dot_lin_tensor)-f1_fit(0) , 'r', label='Poly-CC')
    axs[0].plot(x_dot_lin_tensor.numpy(), f1_sindy(x_dot_lin_tensor.numpy()), 'blue', label='SINDy-CC')
#-0.1    axs[0].plot(x_dot_lin_tensor, f1_sindy(x_dot_lin_tensor) + f2_sindy(0), 'darkgreen', label='Sindy')
    axs[0].plot(x_dot_lin_tensor, F1(x_dot_lin_tensor) , "--", color='black', label="Theoretical", lw=1,dashes=(10,5))
    axs[0].set_xlabel("$\\dot{x}$", fontsize=24)
    axs[0].set_ylabel("$f_3(\\dot{x})$", fontsize=24)
    F1_th_range = np.max(F1_th) - np.min(F1_th)
    axs[0].set_ylim(np.min(F1_th) - F1_th_range * 0.5, np.max(F1_th) + F1_th_range * 0.5)
    axs[0].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1))
    custom_ticks(axs[0], major_x_interval=1, major_y_interval=1, minor_x_interval=0.5, minor_y_interval=0.25)
    #axs[0].tick_params(axis='both', direction='in', which='major', length=8, width=1.5, labelsize=16)
    #axs[0].tick_params(axis='both', direction='in', which='minor', length=4, width=1)
    #axs[0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    #axs[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[0].text(0.96, 0.04, "(a)", transform=axs[0].transAxes, fontsize=24, va='bottom', ha='right')
    # Second subplot: F2
    axs[1].axvspan(np.min(x_data), np.max(x_data), color='lightgray', alpha=0.6, zorder=0)
    #uncomment if required
    #axs[1].scatter(x_data, F2_th,color='blue', label="Training DS", lw=1, s=5,marker='s')
    axs[1].plot(x_extrap_left, f2_extrap_left - bias_correction-correction_extra, "-", color='violet', label="NN-CC (lin-extrap)", lw=2.5)
    axs[1].plot(x_lin_data, predicted_lin_F2_data - bias_correction-correction_extra, "-", color='violet', lw=2.5)
    axs[1].plot(x_extrap_right, f2_extrap_right - bias_correction-correction_extra, "-", color='violet', lw=2.5)

    axs[1].plot(x_lin, (pred_f2SR - bias_correction-correction_extra-correction_SR).flatten(), "-", color='magenta', label='NN-CC-SR', lw=2.5)
    axs[1].plot(x_lin_tensor, predicted_lin_F2_retrain - bias_correction-correction_extra, "-", color='orange', label='NN-CC (retrained lin-extrap)', lw=2.5)
    axs[1].plot(x_lin_tensor, f2_fit(x_lin_tensor) + f1_fit(0), 'r', label='Poly-CC')
    axs[1].plot(x_lin_tensor.numpy(), f2_sindy(x_lin_tensor.numpy()), 'blue', label='SINDy-CC')
#+0.1    axs[1].plot(x_lin_tensor, f2_sindy(x_lin_tensor) - f2_sindy(0), 'darkgreen', label='Sindy')
    axs[1].plot(x_lin_tensor, F2(x_lin_tensor) , "--", color='black', label="Theoretical", lw=1,dashes=(10,5))
    axs[1].set_xlabel("$x$", fontsize=24)
    axs[1].set_ylabel("$f_4(x)$", fontsize=24)
    F2_th_range = np.max(F2_th) - np.min(F2_th)
    axs[1].set_ylim(np.min(F2_th) - F2_th_range * 0.5, np.max(F2_th) + F2_th_range * 0.5)
    axs[1].legend(fontsize=14)
    #axs[1].tick_params( axis='both', direction='in', which='major', length=8, width=1.5, labelsize=16)
    #axs[1].tick_params( axis='both', direction='in', which='minor', length=4, width=1)
    #axs[1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    #axs[1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    custom_ticks(axs[1], major_x_interval=1, major_y_interval=1, minor_x_interval=0.5, minor_y_interval=0.5)
    axs[1].text(0.96, 0.04, "(b)", transform=axs[1].transAxes, fontsize=24, va='bottom', ha='right')
    plt.tight_layout()
    # Save to a specific folder
    folder_path = output_path #"/content/drive/My Drive/Colab Notebooks/Plots"
    os.makedirs(folder_path, exist_ok=True)
    file_name = "CC_duffing.pdf"
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    print(f"Saved to: {file_path}")



    print('bias correction=',bias_correction)
    print('extra correction=',correction_extra)




    fig, axs = plt.subplots(2, 1, figsize=(5, 10), sharex=False)
    # First subplot: F1
    axs[0].axvspan(np.min(x_dot_data), np.max(x_dot_data), color='lightgray', alpha=0.6, zorder=0.5)
    #uncomment if required
    #axs[0].scatter(x_dot_data, F1_th, color='blue', label="Training DS", lw=1, s=5, marker='s')
    axs[0].plot(x_dot_lin_tensor, predicted_lin_F1 + bias_correction, "-", color='red', label="NN-CC (without-ext)", lw=2)
    axs[0].plot(x_dot_lin_tensor, predicted_lin_F1_retrain + bias_correction, "-", color='orange', label="NN-CC (lin-ext)", lw=2)
    axs[0].plot(x_dot_lin_tensor, F1(x_dot_lin_tensor) + bias_correction, "--", color='black', label="Theoretical", lw=1,dashes=(10,5))
    #axs[0].plot(x_dot_lin_tensor, predicted_lin_F1_retrain + bias_correction, "--", color='black', label="Theoretical", lw=2)
    #axs[0].plot(x_dot_lin_tensor, f1_fit(x_dot_lin_tensor) , 'r', label='Poly-CC')
    #axs[0].plot(x_dot_lin_tensor.numpy(), f1_sindy(x_dot_lin_tensor.numpy()), 'darkgreen', label='SINDy')
#-0.1    axs[0].plot(x_dot_lin_tensor, f1_sindy(x_dot_lin_tensor) + f2_sindy(0), 'darkgreen', label='Sindy')
    axs[0].set_xlabel("$\\dot{x}$", fontsize=24)
    axs[0].set_ylabel("$f_3(\\dot{x})$", fontsize=24)
    F1_th_range = np.max(F1_th) - np.min(F1_th)
    axs[0].set_ylim(np.min(F1_th) - F1_th_range * 0.5, np.max(F1_th) + F1_th_range * 0.5)
    axs[0].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1),labelspacing=0.15)
    custom_ticks(axs[0], major_x_interval=1, major_y_interval=1, minor_x_interval=0.5, minor_y_interval=0.25)
    #axs[0].tick_params(axis='both', direction='in', which='major', length=8, width=1.5, labelsize=16)
    #axs[0].tick_params(axis='both', direction='in', which='minor', length=4, width=1)
    #axs[0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    #axs[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs[0].text(0.96, 0.04, "(a)", transform=axs[0].transAxes, fontsize=24, va='bottom', ha='right')
    # Second subplot: F2
    axs[1].axvspan(np.min(x_data), np.max(x_data), color='lightgray', alpha=0.6, zorder=0)
    #uncomment if required
    #axs[1].scatter(x_data, F2_th,color='blue', label="Training DS", lw=1, s=5,marker='s')
    axs[1].plot(x_lin_tensor, predicted_lin_F2 - bias_correction, "-", color='red', label='NN-CC (without-ext)', lw=2)
    axs[1].plot(x_lin_tensor, predicted_lin_F2_retrain - bias_correction, "-", color='orange', label='NN-CC (lin-ext)', lw=2)
    axs[1].plot(x_lin_tensor, F2(x_lin_tensor) + bias_correction, "--", color='black', label="Theoretical", lw=1,dashes=(10,5))
    #axs[1].plot(x_lin_tensor, predicted_lin_F2_retrain + bias_correction, "--", color='black', label="Theoretical", lw=2)
    #axs[1].plot(x_lin_tensor, f2_fit(x_lin_tensor) - f2_fit(0), 'r', label='Poly-CC')
    #axs[1].plot(x_lin_tensor.numpy(), f2_sindy(x_lin_tensor.numpy()), 'darkgreen', label='SINDy')
#+0.1    axs[1].plot(x_lin_tensor, f2_sindy(x_lin_tensor) - f2_sindy(0), 'darkgreen', label='Sindy')
    axs[1].set_xlabel("$x$", fontsize=24)
    axs[1].set_ylabel("$f_4(x)$", fontsize=24)
    F2_th_range = np.max(F2_th) - np.min(F2_th)
    axs[1].set_ylim(np.min(F2_th) - F2_th_range * 0.5, np.max(F2_th) + F2_th_range * 0.5)
    #axs[1].legend(fontsize=14)
    axs[1].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1),labelspacing=0.15)
    #axs[1].tick_params( axis='both', direction='in', which='major', length=8, width=1.5, labelsize=16)
    #axs[1].tick_params( axis='both', direction='in', which='minor', length=4, width=1)
    #axs[1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    #axs[1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    custom_ticks(axs[1], major_x_interval=1, major_y_interval=1, minor_x_interval=0.5, minor_y_interval=0.5)
    axs[1].text(0.96, 0.04, "(b)", transform=axs[1].transAxes, fontsize=24, va='bottom', ha='right')
    plt.tight_layout()
    # Save to a specific folder
    folder_path = output_path #"/content/drive/My Drive/Colab Notebooks/Plots"
    os.makedirs(folder_path, exist_ok=True)
    file_name = "CC_duffing_appendix.pdf"
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    print(f"Saved to: {file_path}")







    # Simulation of the model with Neural Network
    # Define simulation parameters
    #Nval=4
    #Tval=50
    #Nval=5000

    ## Initialize variables for the simulation
    #x_simulated = [y0[0]]  # Start with initial position x(0)
    #x_dot_simulated = [y0[1]]  # Start with initial velocity x'(0)
    #
    ## Run the simulation with EULER integration method
    #for i in range(1, Nval):
    #    # Prepare input tensor with current x and x_dot
    #    x_current = torch.tensor([[x_simulated[-1]]], dtype=torch.float32)
    #    x_dot_current = torch.tensor([[x_dot_simulated[-1]]], dtype=torch.float32)
    #    t_current = torch.tensor([[t_eval[i-1]]], dtype=torch.float32)
    #    #F_ext_tensor_current = torch.tensor([[F_ext(t_eval[i-1])]], dtype=torch.float32)
    #    # Predict acceleration using the neural network
    #    #x_ddot_pred = (model3(x_current) -model1(x_current) * x_dot_current - model2(x_current)).item()
    #    x_ddot_pred = ( F_ext(t_current) -model1(x_current) * x_dot_current - model2(x_current)).item()
    #
    #    # Update position and velocity using Euler's method
    #    x_dot_next = x_dot_simulated[-1] + x_ddot_pred * dt
    #    x_next = x_simulated[-1] + x_dot_next * dt
    #
    #    # Append the results
    #    x_dot_simulated.append(x_dot_next)
    #    x_simulated.append(x_next)


    ## Run the simulation with RK4 integration method
    ## Initialize variables for the simulation
    #x_simulated = [y0[0]]  # Start with initial position x(0)
    #x_dot_simulated = [y0[1]]  # Start with initial velocity x'(0)
    ## Run the simulation with RK4 integration method
    #for i in range(1, Nval):
    #    # Prepare input tensors for the current values
    #    x_current = torch.tensor([[x_simulated[-1]]], dtype=torch.float32)
    #    x_dot_current = torch.tensor([[x_dot_simulated[-1]]], dtype=torch.float32)
    #    t_current = torch.tensor([[t_val[i-1]]], dtype=torch.float32)
    #    #F_ext_tensor_current = torch.tensor([[F_ext(t_eval[i-1])]], dtype=torch.float32)
    #    # Runge-Kutta calculations for x_ddot using neural network model
    #    # Step 1
    #    k1_v = (F_ext(t_current)-model1(x_dot_current) - model2(x_current)).item()
    #    k1_x = x_dot_current.item()
    #    # Step 2
    #    x_mid = x_current + 0.5 * k1_x * dt
    #    x_dot_mid = x_dot_current + 0.5 * k1_v * dt
    #    k2_v = (F_ext(t_current)-model1(x_dot_mid)  - model2(x_mid)).item()
    #    #k2_v = (model3(t_current)-model1(x_mid) * x_dot_mid - model2(x_mid)).item()
    #    k2_x = x_dot_mid.item()
    #    # Step 3
    #    x_mid = x_current + 0.5 * k2_x * dt
    #    x_dot_mid = x_dot_current + 0.5 * k2_v * dt
    #    k3_v = (F_ext(t_current)-model1(x_dot_mid)  - model2(x_mid)).item()
    #    k3_x = x_dot_mid.item()
    #    # Step 4
    #    x_end = x_current + k3_x * dt
    #    x_dot_end = x_dot_current + k3_v * dt
    #    k4_v = (F_ext(t_current)-model1(x_dot_end)  - model2(x_end)).item()
    #    k4_x = x_dot_end.item()
    #    # Combine the increments to get the next values
    #    x_dot_next = x_dot_simulated[-1] + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    #    x_next = x_simulated[-1] + (dt / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
    #    # Append the results
    #    x_dot_simulated.append(x_dot_next)
    #    x_simulated.append(x_next)



    ################################################################################
    ################################################################################
    ################################   VALIDATION   ################################
    ################################################################################
    ################################################################################



    print('Validating the system')
    printF('Validating the system')
    print('Integration of model EDOs')
    printF('Integration of model EDOs')


    n_trials = 1  # number of random initial conditions
    #rmse_x_NN_list = []
    #rmse_x_dot_NN_list = []
    #rmse_x_Sindy_list = []
    #rmse_x_dot_Sindy_list = []
    #rmse_x_LS_list = []
    #rmse_x_dot_LS_list = []

    for i in range(n_trials):

        # Random initial conditions uncomment
        #x0_val = np.round(np.random.uniform(-0.5, 0.5),3)
        #v0_val = np.round(np.random.uniform(-0.5, 0.5),3)
        #y0_val = [x0_val, v0_val]
        #A = np.round(np.random.uniform(1.0, 1.5),3)
        #Omega = np.round(np.random.uniform(0.2, 0.4),3)
        #x0_val=x0-0.05
        #v0_val=v0+0.01
        
        x0_val=-0.8 #x0
        v0_val=0.7 #v0
        y0_val = [x0_val,v0_val]
        y0_val = y0
        
        
        #kval = np.random.uniform(0.5, 1.)
        #cval = np.random.uniform(0.1, 0.5)
        #m = 1.0
        #mu_N = np.random.uniform(0.5, 1.0)
        #print(f"Trial {i+1} : x0={x0_val:.4f} ; v0={v0_val:.4f}")
        #print(r"$\A$="+f"{A:.4f} ; "+r"$\Omega$"+f"={Omega:.4f}")
        print(f"Trial {i+1} :")
        #print(f"alpha={alpha}, c={cval}")
        #print(f"$\Omega$={Omega}, $\mu$*N={mu_N}, $x_0$={x0}, $v_0$={v0}")
        #print(f"Omega={Omega}, A={A}, $x_0$={x0_val:.4f}, $v_0$={v0_val:.4f}")
        printF(f"Trial {i+1} :")
        #printF(f"k={kval}, c={cval}")
        #printF(f"Omega={Omega}, A={A}, $x_0$={x0_val:.4f}, $v_0$={v0_val:.4f}")


        #kval = np.round(np.random.uniform(1, 1.5),3)
        #cval = np.round(np.random.uniform(0.1, 0.5),3)
        #m = 1.0
        #mu_N = np.round(np.random.uniform(0.5, 1.0),3)
        #Omega = np.round(np.random.uniform(0.2, 0.5),3)
        #x0 = np.round(np.random.uniform(-0.5, 0.5),3)
        #v0 = np.round(np.random.uniform(-0.5, 0.5),3)

        #uncomment definitions only first #
        #def smooth_sign(x, alpha=500):
        #    return np.tanh(alpha * x)
        #def Ff_dr(x_dot):
        #    abs_v = np.abs(x_dot) + epsilon
        #    #abs_v = np.clip(np.abs(x_dot), 1e-6, 1e2)  # Prevent too small or too big
        #    abs_v = np.maximum(np.abs(x_dot), 1e-10)  # Prevent division by zero
        #    term1 = a * np.log(abs_v / Vf)
        #    term2 = b * np.log(c + Vf / abs_v)
        #    #return (Ff + term1 + term2) * np.sign(x_dot)
        #    return (Ff + term1 + term2) * smooth_sign(x_dot)
        #def Ff_coul(x_dot):
        # #   return mu_N * np.sign(x_dot)
        #    return mu_N * smooth_sign(x_dot)
        #def F1(x_dot):
        #    return cval* x_dot + Ff_coul(x_dot) # + 0.0005 * x_dot**2 #+ Ff_coul(x_dot) #r(x_dot) Ff_coul Ff_dr
        #def F2(x):
        #    return kval*x
        #def F_ext(t):
        #    return A*np.cos(Omega*t)
        #def eq_2nd_ord_veloc(t,y):
        #    x, x_dot = y  # y=[x, x']
        #    x_ddot = (F_ext(t) - F1(x_dot) - F2(x))
        #    return [x_dot, x_ddot]
        #
        #
        #def Ff_coul_anderson2009(x_dot,F_ext):
        #    #abs_x_dot = np.abs(x_dot)
        #    #if abs_x_dot > 0: #1e-8:
        #    #    return mu_N * np.sign(x_dot)
        #    #    #return mu_N * smooth_sign(x_dot)
        #    #else:
        #    #    return np.minimum(np.abs(F_ext), mu_N) * np.sign(F_ext)
        #    abs_x_dot = np.abs(x_dot)
        #    abs_fext = np.abs(F_ext)
        #    sign_fext = np.sign(F_ext)
        #
        #    # If scalar: return scalar
        #    if np.isscalar(x_dot):
        #        if abs_x_dot > 1e-8:
        #            return mu_N * np.sign(x_dot) # smooth_sign(x_dot)  #
        #        else:
        #            return min(abs_fext, mu_N) * sign_fext
        #
        #    # If array: return array
        #    result = np.where(
        #        abs_x_dot > 1e-8,
        #        mu_N * smooth_sign(x_dot),
        #        np.minimum(abs_fext, mu_N) * sign_fext
        #    )
        #    return result
        ##   return mu_N * np.sign(x_dot)
        ##   return mu_N * smooth_sign(x_dot)
        ##    return mu_N * np.sign(x_dot) if |x_dot>0? else min(|Fext|,mu_N)*np.sign(Fext)
        #def F1_anderson2009(x_dot,F_ext):
        #    return c* x_dot + Ff_coul_anderson2009(x_dot,F_ext) #r(x_dot) Ff_coul Ff_dr
        #def F2_anderson2009(x):
        #    return kval*x/m
        #def eq_2nd_ord_veloc_anderson2009(t,y):
        #    x, x_dot = y  # y=[x, x']
        #    fext_val = F_ext(t)
        ##    friction = Ff_coul_anderson2009(x_dot, fext_val)
        #    x_ddot = (fext_val - F1_anderson2009(x_dot,fext_val) - F2_anderson2009(x)) / m
        #    return [x_dot, x_ddot]



        ################ Theoretical Eq ###### validation of the model
        print("Integrating Th")
        printF("Integrating Th")
       
        A=0.5
        alpha=-1.0
        beta=1.0
        delta=0.3
        Omega=1.2
        x0=0.5
        v0=-0.5
        x0_val=x0
        y0_val=y0
        y0=[x0,v0]
        y0_val=y0
        
        start = time.time()  
        sol_val = solve_ivp(eq_2nd_ord_veloc, t_span_val, y0_val, t_eval=t_val,method='LSODA')
        
        
        end = time.time()  
        elapsed = end - start
        print(f"Solve_ivp finished in {elapsed:.3f} seconds")
        printF(f"Solve_ivp finished in {elapsed:.3f} seconds")

        t_simulated_th = sol_val.t
        x_simulated_th = sol_val.y[0]       # Posición
        x_dot_simulated_th = sol_val.y[1]   # Velocidad
        #time_steps = Nval
        dt = t_val[1] - t_val[0]  # Assuming uniform time steps
        plt.figure()
        plt.plot(t_simulated_th, x_simulated_th, label="Validation")
        #plt.plot(time_data, x_data, label="true")
        plt.plot(time_data, x_data, label="Training data")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("x(t)")
        plt.show()

        if np.max(x_simulated_th)>np.max(x_data):
            print("Extrapolation! max(x_sim)>max(x_train) :",np.max(x_simulated_th),">",np.max(x_data))
            printF("Extrapolation! max(x_sim)>max(x_train) :",np.max(x_simulated_th),">",np.max(x_data))
        if np.min(x_simulated_th)<np.min(x_data):
            print("Extrapolation! min(x_sim)<min(x_train) :",np.min(x_simulated_th),"<",np.min(x_data))
            printF("Extrapolation! min(x_sim)<min(x_train) :",np.min(x_simulated_th),"<",np.min(x_data))



        ### SR is failing for stick-slip
        ################ SR ###### validation of the model
        # Integrate
        #t_span = (0, 10)
        #t_val = np.linspace(t_span[0], t_span[1], 500)
        print("Integrating SR")
        printF("Integrating SR")
        start = time.time()  
        x_simulated_SR=[]
        x_dot_simulated_SR=[]
        try:
            sol_sr = solve_ivp(ode_sr, t_span_val, y0_val, t_eval=t_val , method='LSODA') #,rtol=1e-7,atol=1e-7) #,  method='DOP853', rtol=1e-9, atol=1e-12)
            t_SR = sol_sr.t
            x_simulated_SR = sol_sr.y[0]
            x_dot_simulated_NN = sol_sr.y[1]
            # check for NaNs or infs just in case
            if np.any(np.isnan(x_simulated_NN)) or np.any(np.isinf(x_simulated_NN)):
                print(f"Skipping trial {i+1}: SR simulation returned NaNs or infs.")
                printF(f"Skipping trial {i+1}: SR simulation returned NaNs or infs.")
        except Exception as e:
            print(f"Skipping trial {i+1}: Exception during SR simulation -> {e}")
            printF(f"Skipping trial {i+1}: Exception during SR simulation -> {e}")
        end = time.time()  
        elapsed = end - start
        print(f"Solve_ivp finished in {elapsed:.3f} seconds")
        printF(f"Solve_ivp finished in {elapsed:.3f} seconds")
##        sol_sr = solve_ivp(ode_sr, t_span_val, [x0_val, v0_val], t_eval=t_val,method='LSODA')
##        # Results
##        t_SR = sol_sr.t
##        x_simulated_SR = sol_sr.y[0]
##        x_dot_simulated_SR = sol_sr.y[1]
#
        plt.figure()
        plt.plot(t_SR, x_simulated_SR, label="SR")
        #plt.plot(time_data, x_data, label="true")
        plt.plot(time_data, x_data, label="true")
        plt.legend(t_simulated_th,x_simulated_th)
        plt.xlabel("Time")
        plt.ylabel("x(t)")
        plt.show()




        ################ parametric ###### validation of the model
        sol_parametric = solve_ivp(ode_param, t_span_val, [x0_val, v0_val], t_eval=t_val,method='LSODA')
                        #rtol=1e-9, atol=1e-12, max_step=(t_eval[1] - t_eval[0]))   
        t_parametric= sol_parametric.t
        x_simulated_parametric = sol_parametric.y[0]
        x_dot_simulated_parametric = sol_parametric.y[1]
        
        # Plot comparison
        plt.figure(figsize=(8,5))
        plt.plot(time_data, x_data, label="True x(t)", linewidth=2)
        plt.plot(t_parametric, x_simulated_parametric, "--", label="Simulated x(t)", linewidth=2)
        plt.xlabel("Time")
        plt.ylabel("x(t)")
        plt.legend()
        plt.title("Parametric model simulation")
        plt.show()



        ################ NN ###### 
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
#        sol = solve_ivp(learned_dynamics, t_span_val, y0_val, t_eval=t_val , method='LSODA') #,rtol=1e-7,atol=1e-7) #,  method='DOP853', rtol=1e-9, atol=1e-12)
#        #sol = solve_ivp(learned_dynamics, t_span, y0_val, t_eval=t_val, method='DOP853', rtol=1e-9, atol=1e-12)
#        # Extract results
#        x_simulated_NN = sol.y[0]
#        x_dot_simulated_NN = sol.y[1]
        print("Integrating NN")
        printF("Integrating NN")
        start = time.time()  
        x_simulated_NN=[]
        x_dot_simulated_NN=[]
        try:
            sol = solve_ivp(learned_dynamics, t_span_val, y0_val, t_eval=t_val , method='LSODA') #,rtol=1e-7,atol=1e-7) #,  method='DOP853', rtol=1e-9, atol=1e-12)
            x_simulated_NN = sol.y[0]
            x_dot_simulated_NN = sol.y[1]            # check for NaNs or infs just in case
            if np.any(np.isnan(x_simulated_NN)) or np.any(np.isinf(x_simulated_NN)):
                print(f"Skipping trial {i+1}: NN simulation returned NaNs or infs.")
                printF(f"Skipping trial {i+1}: NN simulation returned NaNs or infs.")
        except Exception as e:
            print(f"Skipping trial {i+1}: Exception during NN simulation -> {e}")
            printF(f"Skipping trial {i+1}: Exception during NN simulation -> {e}")
        end = time.time()  
        elapsed = end - start
        print(f"Solve_ivp finished in {elapsed:.3f} seconds")
        printF(f"Solve_ivp finished in {elapsed:.3f} seconds")

        #x_simulated_NN = sol.y[0]
        #x_dot_simulated_NN = sol.y[1]




        ################ NN retrain extrapolation ###### validation of the model
        def learned_dynamics(t, y):
            x = torch.tensor([[y[0]]], dtype=torch.float32)
            x_dot = torch.tensor([[y[1]]], dtype=torch.float32)
            t_tensor = torch.tensor([[t]], dtype=torch.float32)
            F_ext_tensor = torch.tensor([[F_ext(t)]], dtype=torch.float32)
            model1_retrain.eval()
            model2_retrain.eval()
            with torch.no_grad(): # Neural net-based force computation
                force = F_ext_tensor - model1_retrain(x_dot) - model2_retrain(x)
            x_ddot = force.item()
            return [y[1], x_ddot]  # dx/dt = x_dot, dx_dot/dt = x_ddot
#        sol = solve_ivp(learned_dynamics, t_span_val, y0_val, t_eval=t_val , method='LSODA') #,rtol=1e-7,atol=1e-7) #,  method='DOP853', rtol=1e-9, atol=1e-12)
#        #sol = solve_ivp(learned_dynamics, t_span, y0_val, t_eval=t_val, method='DOP853', rtol=1e-9, atol=1e-12)
#        # Extract results
#        x_simulated_NN_retrain = sol.y[0]
#        x_dot_simulated_NN_retrain = sol.y[1]

        print("Integrating NN retrain")
        printF("Integrating NN retrain")
        start = time.time()  
        x_simulated_NN_retrain =[]
        x_dot_simulated_NN_retrain = []
        try:
            sol = solve_ivp(learned_dynamics, t_span_val, y0_val, t_eval=t_val , method='LSODA') #,rtol=1e-7,atol=1e-7) #,  method='DOP853', rtol=1e-9, atol=1e-12)
            x_simulated_NN_retrain = sol.y[0]
            x_dot_simulated_NN_retrain = sol.y[1]
            # check for NaNs or infs just in case
            if np.any(np.isnan(x_simulated_NN)) or np.any(np.isinf(x_simulated_NN)):
                print(f"Skipping trial {i+1}: NN retrain simulation returned NaNs or infs.")
                printF(f"Skipping trial {i+1}: NN retrain simulation returned NaNs or infs.")
        except Exception as e:
            print(f"Skipping trial {i+1}: Exception during NN retrain simulation -> {e}")
            printF(f"Skipping trial {i+1}: Exception during NN retrain simulation -> {e}")
        end = time.time()  
        elapsed = end - start
        print(f"Solve_ivp finished in {elapsed:.3f} seconds")
        printF(f"Solve_ivp finished in {elapsed:.3f} seconds")

        #x_simulated_NN_retrain = sol.y[0]
        #x_dot_simulated_NN_retrain = sol.y[1]


#        # working but improved below for computational speed
#        ################ NN+SR ###### validation of the model
#        # Define learned dynamics using PySR models
#        def learned_dynamics_SR(t, y):
#            x_val = y[0]
#            x_dot_val = y[1]
#            F_ext_val = F_ext(t)
#            # f1(x_dot) and f2(x) using SR (PySR) models
#            f1_val = model_f1SR.predict(np.array([[x_dot_val]]))[0]
#            f2_val = model_f2SR.predict(np.array([[x_val]]))[0]
#            # Newton's law: m x_ddot = F_ext - f1(x_dot) - f2(x)
#            x_ddot = F_ext_val - f1_val - f2_val
#            return [x_dot_val, x_ddot]
#        # Now simulate
#        sol = solve_ivp(
#            learned_dynamics_SR,
#            t_span_val,
#            y0_val,
#            t_eval=t_val,
#            method='LSODA'  # or 'DOP853' depending on your stiffness needs
#        )
#        # Extract results
#        x_simulated_NN_SR = sol.y[0]
#        x_dot_simulated_NN_SR = sol.y[1]



        ################ NN+SR ###### validation of the model
        print("Integrating NN-SR")
        printF("Integrating NN-SR")
#        # Define learned dynamics using PySR models
#        def learned_dynamics_SR(t, y):
#            x_val = y[0]
#            x_dot_val = y[1]
#            F_ext_val = F_ext(t)
#            # f1(x_dot) and f2(x) using SR (PySR) models
#            f1_val = model_f1SR.predict(np.array([[x_dot_val]]))[0]
#            f2_val = model_f2SR.predict(np.array([[x_val]]))[0]
#            # Newton's law: m x_ddot = F_ext - f1(x_dot) - f2(x)
#            x_ddot = F_ext_val - f1_val - f2_val
#            return [x_dot_val, x_ddot]
#        # Now simulate
##        sol = solve_ivp(learned_dynamics_SR,t_span_val,y0_val,t_eval=t_val,method='LSODA')  # or 'DOP853' depending on your stiffness needs
##        x_simulated_NN_SR = sol.y[0]
##        x_dot_simulated_NN_SR = sol.y[1]
#        x_simulated_NN_SR = []
#        x_dot_simulated_NN_SR = []
#        try:
#            sol_nn_sr = solve_ivp(learned_dynamics_SR,t_span_val,y0_val,t_eval=t_val,method='Radau')  # or 'DOP853' depending on your stiffness needs
#            #x_simulated_NN_SR = sol.y[0]
#            # check for NaNs or infs just in case
#            x_simulated_NN_SR = sol_nn_sr.y[0]
#            x_dot_simulated_NN_SR = sol_nn_sr.y[1]
#            if np.any(np.isnan(x_simulated_NN_SR)) or np.any(np.isinf(x_simulated_NN_SR)):
#                print(f"Skipping trial {i+1}: NN-SR simulation returned NaNs or infs.")
#                printF(f"Skipping trial {i+1}: NN-SR simulation returned NaNs or infs.")
#
#        except Exception as e:
#            print(f"Skipping trial {i+1}: Exception during NN-SR simulation -> {e}")
#            printF(f"Skipping trial {i+1}: Exception during NN-SR simulation -> {e}")
#  

        #x_simulated_NN_SR = sol.y[0]
        #x_dot_simulated_NN_SR = sol.y[1]
        #This approach seems to be faster
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
                print(f"Skipping trial {i+1}: NN-CC-SR simulation returned NaNs or infs.")
                printF(f"Skipping trial {i+1}: NN-CC-SR simulation returned NaNs or infs.")
        except Exception as e:
            print(f"Skipping trial {i+1}: Exception during NN-CC-SR simulation -> {e}")
            printF(f"Skipping trial {i+1}: Exception during NN-CC-SR simulation -> {e}")

        
        end = time.time()  
        elapsed = end - start
        print(f"Solve_ivp finished in {elapsed:.3f} seconds")
        printF(f"Solve_ivp finished in {elapsed:.3f} seconds")
        

        plt.figure()
        plt.plot(t_simulated_th, x_simulated_th, label="Th")
        plt.plot(t_NN_SR, x_simulated_NN_SR, label="NN-SR")
        #plt.plot(time_data, x_data, label="true")
        plt.legend
        plt.xlabel("Time")
        plt.ylabel("x(t)")
        plt.show()


        #############################################################        
        ################ SINDY ###### validation of the model
        #y0new=[0.8,0.0] # Condiciones iniciales: x(0) = 2, x'(0) = 0
        # Simulación con el modelo SINDy
        #u_val_sindy = F_ext(t_val)
        #x_val_sindy = model.simulate(y0_val,t_val, u=u_val_sindy)
        bool_print_sindy=True #True
        x_simulated_Sindy = [] #np.zeros(len(t_val))
        x_dot_simulated_Sindy = [] # np.zeros(len(t_val))
        print("Integrating Sindy-CC")
        printF("Integrating Sindy-CC")
        start = time.time()  
        try:
            u_val_sindy = F_ext(t_val)
            x_val_sindy = model.simulate(y0_val, t_val, u=u_val_sindy)
            # check for NaNs or infs just in case
            if np.any(np.isnan(x_val_sindy)) or np.any(np.isinf(x_val_sindy)):
                print(f"Skipping trial {i+1}: SINDy simulation returned NaNs or infs.")
                printF(f"Skipping trial {i+1}: SINDy simulation returned NaNs or infs.")
            x_simulated_Sindy = x_val_sindy[:, 0]
            x_dot_simulated_Sindy = x_val_sindy[:, 1]
             #   bool_print_sindy=False
             #   continue
            # continue your RMSE calculations and plots here...
        except Exception as e:
            print(f"Skipping trial {i+1}: Exception during SINDy simulation -> {e}")
            printF(f"Skipping trial {i+1}: Exception during SINDy simulation -> {e}")
            #bool_print_sindy=False
        end = time.time()  
        elapsed = end - start
        print(f"Solve_ivp finished in {elapsed:.3f} seconds")
        printF(f"Solve_ivp finished in {elapsed:.3f} seconds")


        # continue
        # Visualización de resultados
        plt.figure(figsize=(12, 4))
        plt.plot(t_simulated_th, x_simulated_th, label="Posición (datos reales)", lw=2)
        plt.plot(t_val, x_val_sindy[:, 0], "--", label="Posición (predicción SINDy)", lw=2)
        plt.xlabel("Tiempo")
        plt.ylabel("Posición")
        plt.legend()
        plt.title("Comparación entre datos reales y predicción SINDy (sistema no lineal)")
        plt.show()





        ################ SINDY without restrictions ###### validation of the model
        print("Simulating Sindy without restrictions")
        printF("Simulating Sindy without restrictions")
        x_simulated_Sindy_ku0 = []
        x_dot_simulated_Sindy_ku0 = []
        start = time.time()
        try:
            u_val_sindy = F_ext(t_val)
            x_val_sindy_ku0 = model_sindy_ku0.simulate(y0_val, t_val, u=u_val_sindy)
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
        printF(f"Solve_ivp finished in {elapsed:.3f} seconds")
                     


        ################ LS ###### validation of the model
        print("Integrating LS-CC")
        printF("Integrating LS-CC")

#        Maybe faster integration
#        def f1_fit(x_dot):
#            arg = (x_dot - A0_xd)/A1_xd
#            arg = np.clip(arg, -5e1, 5e1)  # avoid huge powers
#            return sum(c * (arg**i) for i, c in enumerate(c_f1))
#        def f2_fit(x_dot):
#            arg = (x_dot - A0_x)/A1_x
#            arg = np.clip(arg, -5e1, 5e1)  # avoid huge powers
#            return sum(c * (arg**i) for i, c in enumerate(c_f2))
#
#        def fitted_model_LS(t, y):
#            x, x_dot = y
#            x_ddot = F_ext(t) - f1_fit(x_dot) - f2_fit(x)
#            return [x_dot, x_ddot]
#
#        sol_fit = solve_ivp(fitted_model_LS, t_span_val, y0_val, t_eval=t_val,method='LSODA')
#        x_simulated_LS = sol_fit.y[0]       # Posición
#        x_dot_simulated_LS = sol_fit.y[1]   # Velocidad
#        end = time.time()  
#        elapsed = end - start
#        print(f"Solve_ivp finished in {elapsed:.3f} seconds")
#        printF(f"Solve_ivp finished in {elapsed:.3f} seconds")
#             

        start = time.time()  
        x_simulated_LS = [] #np.zeros(len(t_val))
        x_dot_simulated_LS = [] # np.zeros(len(t_val))
        try:
            sol_fit = solve_ivp(fitted_model_LS, t_span_val, y0_val, t_eval=t_val,method='LSODA')
            x_simulated_LS = sol_fit.y[0]       # Posición
            x_dot_simulated_LS = sol_fit.y[1]   # Velocidad
            # check for NaNs or infs just in case
            if np.any(np.isnan(x_simulated_LS)) or np.any(np.isinf(x_simulated_LS)):
                print(f"Skipping trial {i+1}: LS simulation returned NaNs or infs.")
                printF(f"Skipping trial {i+1}: LS simulation returned NaNs or infs.")
            # continue your RMSE calculations and plots here...
        except Exception as e:
            print(f"Skipping trial {i+1}: Exception during LS simulation -> {e}")
            printF(f"Skipping trial {i+1}: Exception during LS simulation -> {e}")
        end = time.time()  
        elapsed = end - start
        print(f"Solve_ivp finished in {elapsed:.3f} seconds")
        printF(f"Solve_ivp finished in {elapsed:.3f} seconds")


#        try:
#            sol_fit = solve_ivp(
#                fitted_model_LS,
#                t_span_val,
#                y0_val,
#                t_eval=t_val,
#                method='Radau'
#                #method='LSODA'
#            )
#
#            #if not sol_fit.success:
#            #    print(f"Skipping trial {i+1}: LS integration failed -> {sol_fit.message}")
#            #    printF(f"Skipping trial {i+1}: LS integration failed -> {sol_fit.message}")
#            #    continue
#
#            # Extract results
#            #x_simulated_LS = sol_fit.y[0]
#            #x_dot_simulated_LS = sol_fit.y[1]
#
#            # Check length match
#            #if len(x_simulated_LS) != len(t_val):
#            #    print(f"Skipping trial {i+1}: solver returned {len(x_simulated_LS)} points, expected {len(t_val)}.")
#            #    continue
#
#            # Check for NaNs/Infs
#            if (np.any(np.isnan(x_simulated_LS)) or np.any(np.isinf(x_simulated_LS)) or
#                np.any(np.isnan(x_dot_simulated_LS)) or np.any(np.isinf(x_dot_simulated_LS))):
#                print(f"Skipping trial {i+1}: LS integration returned NaNs or Infs.")
#                printF(f"Skipping trial {i+1}: LS integration returned NaNs or Infs.")
#                #continue
#            x_simulated_LS = sol_fit.y[0]
#            x_dot_simulated_LS = sol_fit.y[1]
#
#        except Exception as e:
#            print(f"Skipping trial {i+1}: Exception during LS simulation -> {e}")
#            printF(f"Skipping trial {i+1}: Exception during LS simulation -> {e}")
#            #continue

#        # Plot the simulation results
#        plt.figure(figsize=(15, 5))
#        plt.subplot(1, 2, 1)
#        plt.plot(t_val, x_simulated_th,'-',color='blue', label="$x_{th}$",linewidth='2')
#        plt.plot(t_val, x_simulated_NN,"--",color='orange', label="$x_{val} NN$",linewidth='3')
#        plt.plot(t_val[0:len(x_simulated_LS)], x_simulated_LS,"--",color='red', label="$x_{val} LS$",linewidth='3')
#        if(bool_print_sindy):
#            plt.plot(t_val[0:len(x_simulated_Sindy)], x_simulated_Sindy,"--",color='darkgreen', label="$x_{val} Sindy$",linewidth='3')
#        plt.ylim(np.min(x_simulated_th)-0.2,np.max(x_simulated_th)+0.2)
#        plt.xlabel("Time $t$")
#        plt.ylabel("Position")
#        plt.title("Validation Test of Position over Time")
#        plt.legend()
#        plt.grid(True)
#        plt.subplot(1, 2, 2)
#        plt.plot(t_val, x_dot_simulated_th,color='blue', label="$\\dot{x}_{th}$")
#        plt.plot(t_val, x_dot_simulated_NN,"--",color='orange', label="$\\dot{x}_{val} NN$", linestyle="dashed",linewidth='3')
#        plt.plot(t_val[0:len(x_dot_simulated_LS)], x_dot_simulated_LS,"--",color='red', label="$\\dot{x}_{val} LS$", linestyle="dashed",linewidth='3')
#        if(bool_print_sindy):
#            plt.plot(t_val[0:len(x_dot_simulated_Sindy)], x_dot_simulated_Sindy,"--",color='darkgreen', label="$\\dot{x}_{val} Sindy$",linewidth='3')
##        plt.plot(t_val[0:-1], x_val_sindy[:,1],"--",color='darkgreen', label="$\\dot{x}_{val} Sindy$",linewidth='3')
#        plt.ylim(np.min(x_dot_simulated_th)-0.2,np.max(x_dot_simulated_th)+0.2)
#        #plt.ylim(-0.75,0.75)
#        plt.xlabel("Time $t$")
#        plt.ylabel("Velocity")
#        plt.title("Neural Network Simulation of Velocity over Time")
#        plt.legend()
#        plt.grid(True)
#        plt.show()


        # CHECK if some integration failed
        if len(x_simulated_NN) != len(t_val):
            print("Warning:")
            print("NN-CC finished integration before maximum simulation time")
            print("time:",t_val[len(x_simulated_NN)-1])
            printF("Warning:")
            printF("NN-CC finished integration before maximum simulation time")
            printF("time:",t_val[len(x_simulated_NN)-1])
        if len(x_simulated_LS) != len(t_val):
            print("Warning:")
            print("LS-CC finished integration before maximum simulation time")
            print("time:",t_val[len(x_simulated_LS)-1])
            printF("Warning:")
            printF("LS-CC finished integration before maximum simulation time")
            printF("time:",t_val[len(x_simulated_LS)-1])
        if len(x_simulated_Sindy) != len(t_val):
            print("Warning:")
            print("Sindy-CC finished integration before maximum simulation time")
            print("time:",t_val[len(x_simulated_Sindy)-1])
            printF("Warning:")
            printF("Sindy-CC finished integration before maximum simulation time")
            printF("time:",t_val[len(x_simulated_Sindy)-1])
        if len(x_simulated_NN_retrain) != len(t_val):
            print("Warning:")
            print("NN-CC retrained finished integration before maximum simulation time")
            print("time:",t_val[len(x_simulated_NN_retrain)-1])
            printF("Warning:")
            printF("NN-CC retrained finished integration before maximum simulation time")
            printF("time:",t_val[len(x_simulated_NN_retrain)-1])
        if len(x_simulated_NN_SR) != len(t_val):
            print("Warning:")
            print("NN-CC-SR finished integration before maximum simulation time")
            print("time:",t_val[len(x_simulated_NN_SR)-1])
            printF("Warning:")
            printF("NN-CC-SR finished integration before maximum simulation time")
            printF("time:",t_val[len(x_simulated_NN_SR)-1])
        if len(x_simulated_SR) != len(t_val):
            print("Warning:")
            print("SR finished integration before maximum simulation time")
            print("time:",t_val[len(x_simulated_SR)-1])
            printF("Warning:")
            printF("SR finished integration before maximum simulation time")
            printF("time:",t_val[len(x_simulated_SR)-1])


        # Plot the simulation results
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(t_val, x_simulated_th,'-',color='blue', label="$x_{th}$",linewidth='2')
        plt.plot(t_val, x_simulated_NN,"--",color='orange', label="$x_{val} NN$",linewidth='3')
        plt.plot(t_val[0:len(x_simulated_LS)], x_simulated_LS,"--",color='red', label="$x_{val} LS$",linewidth='3')
        plt.plot(t_val[0:len(x_simulated_Sindy)], x_simulated_Sindy,"--",color='violet', label="$x_{val} Sindy-CC$",linewidth='3')
        
        if(bool_print_sindy):
            plt.plot(t_val[0:len(x_simulated_Sindy)], x_simulated_Sindy,"--",color='darkgreen', label="$x_{val} Sindy$",linewidth='3')
        plt.ylim(np.min(x_simulated_th)-0.2,np.max(x_simulated_th)+0.2)
        plt.xlabel("Time $t$")
        plt.ylabel("Position")
        plt.title("Validation Test of Position over Time")
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(t_val, x_dot_simulated_th,color='blue', label="$\\dot{x}_{th}$")
        plt.plot(t_val, x_dot_simulated_NN,"--",color='orange', label="$\\dot{x}_{val} NN$", linestyle="dashed",linewidth='3')
        plt.plot(t_val[0:len(x_dot_simulated_LS)], x_dot_simulated_LS,"--",color='red', label="$\\dot{x}_{val} LS$", linestyle="dashed",linewidth='3')
        if(bool_print_sindy):
            plt.plot(t_val[0:len(x_dot_simulated_Sindy)], x_dot_simulated_Sindy,"--",color='darkgreen', label="$\\dot{x}_{val} Sindy$",linewidth='3')
#        plt.plot(t_val[0:-1], x_val_sindy[:,1],"--",color='darkgreen', label="$\\dot{x}_{val} Sindy$",linewidth='3')
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
        axs[0, 0].plot(t_val, x_simulated_NN, "--", color='orange', label="NN-CC", linewidth=3)
        axs[0, 0].plot(t_val, x_simulated_NN_retrain, "--", color='violet', label="NN-CC retrained", linewidth=3)
        axs[0, 0].plot(t_val, x_simulated_NN_SR, "--", color='magenta', label="NN-CC-SR", linewidth=3)
        axs[0, 0].plot(t_val[:len(x_simulated_LS)], x_simulated_LS, "--", color='red', label="Poly-CC", linewidth=3)
        axs[0, 0].plot(t_val[:-1], x_simulated_Sindy, "--", color='darkgreen', label="SINDy-CC", linewidth=3)
        axs[0, 0].set_ylim(np.min(x_simulated_LS)-0.2, np.max(x_simulated_LS)+0.2)
        axs[0, 0].set_ylabel("$x$", fontsize=24)
        axs[0, 0].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1))
        custom_ticks(axs[0, 0], 20, 0.5 , 5 , 0.25)
        axs[0, 0].text(0.98, 0.04, "(a)", transform=axs[0, 0].transAxes, fontsize=24, va='bottom', ha='right')

        # ---- (b) x_dot(t) values ----
        axs[0, 1].plot(t_val, x_dot_simulated_th, '-', color='blue', label="Theor.", linewidth=3)
        axs[0, 1].plot(t_val, x_dot_simulated_NN, "--", color='orange', label="NN-CC", linewidth=3)
        axs[0, 1].plot(t_val, x_dot_simulated_NN_retrain, "--", color='violet', label="NN-CC retrained", linewidth=3)
        axs[0, 1].plot(t_val, x_dot_simulated_NN_SR, "--", color='magenta', label="NN-CC-SR", linewidth=3)
        axs[0, 1].plot(t_val[:len(x_dot_simulated_LS)], x_dot_simulated_LS, "--", color='red', label="Poly-CC", linewidth=3)
        axs[0, 1].plot(t_val[:-1], x_dot_simulated_Sindy, "--", color='darkgreen', label="SINDy-CC", linewidth=3)
        axs[0, 1].set_ylim(np.min(x_dot_simulated_LS)-0.2, np.max(x_dot_simulated_LS)+0.2)
        axs[0, 1].set_ylabel("$\dot{x}$", fontsize=24)
        axs[0, 1].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1))
        custom_ticks(axs[0, 1], 20, 0.5, 5, 0.25)
        axs[0, 1].text(0.98, 0.04, "(b)", transform=axs[0, 1].transAxes, fontsize=24, va='bottom', ha='right')

        # ---- (c) Residuals: x_model - x_theor ----
        axs[1, 0].plot(t_val, x_simulated_NN - x_simulated_th, '--', color='orange', label="NN-CC", linewidth=3)
        axs[1, 0].plot(t_val[:len(x_simulated_LS)], x_simulated_LS - x_simulated_th[:len(x_simulated_LS)], '--', color='red', label="Poly-CC", linewidth=3)
        axs[1, 0].plot(t_val[:-1], x_simulated_Sindy - x_simulated_th[:-1], '--', color='darkgreen', label="SINDy-CC", linewidth=3)
        axs[1, 0].axhline(0, color='black', linewidth=1)
        axs[1, 0].set_ylabel("$x-x_{th.}$", fontsize=22)
        axs[1, 0].set_xlabel("$t$", fontsize=24)
        axs[1, 0].legend(fontsize=14, loc='upper left', bbox_to_anchor=(0, 1))
        custom_ticks(axs[1, 0], 20, 0.2, 5, 0.1)
        axs[1, 0].text(0.9, 0.04, "(c)", transform=axs[1, 0].transAxes, fontsize=24, va='bottom', ha='right')

        # ---- (d) Residuals: x_dot_model - x_dot_theor ----
        axs[1, 1].plot(t_val, x_dot_simulated_NN - x_dot_simulated_th, '--', color='orange', label="NN-CC", linewidth=3)
        axs[1, 1].plot(t_val[:len(x_dot_simulated_LS)], x_dot_simulated_LS - x_dot_simulated_th[:len(x_dot_simulated_LS)], '--', color='red', label="Poly-CC", linewidth=3)
        axs[1, 1].plot(t_val[:-1], x_dot_simulated_Sindy - x_dot_simulated_th[:-1], '--', color='darkgreen', label="SINDy-CC", linewidth=3)
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

#        time_chaos_x_SR_list.append(t_val[-1])
        time_chaos_x_SR=t_val[len(x_simulated_SR)-1]
        for i in range(len(x_simulated_SR)):
            diff = abs(x_simulated_SR[i] - x_simulated_th[i])
            if diff > threshold_chaos:
#                time_chaos_x_SR_list.append(t_val[i])
                time_chaos_x_SR=t_val[i]
                break
        time_chaos_x_SR_list.append(time_chaos_x_SR)
#        time_chaos_x_parametric_list.append(t_val[-1])

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
        #time_chaos_x_NN_retrain_list.append(t_val[-1])

        time_chaos_x_NN_retrain=t_val[len(x_simulated_NN_retrain)-1]
        for i in range(len(x_simulated_NN_retrain)-1):
            diff = abs(x_simulated_NN_retrain[i] - x_simulated_th[i])
            if diff > threshold_chaos:
                #time_chaos_x_NN_retrain_list.append(t_val[i])
                time_chaos_x_NN_retrain=t_val[i]
                break
        time_chaos_x_NN_retrain_list.append(time_chaos_x_NN_retrain)
        #time_chaos_x_NN_SR_list.append(t_val[-1])

        time_chaos_x_NN_SR=t_val[len(x_simulated_NN_SR)-1]
        for i in range(len(x_simulated_NN_SR)):
            diff = abs(x_simulated_NN_SR[i] - x_simulated_th[i])
            if diff > threshold_chaos:
                #time_chaos_x_NN_SR_list.append(t_val[i])
                time_chaos_x_NN_SR=t_val[i]
                break
        time_chaos_x_NN_SR_list.append(time_chaos_x_NN_SR)
        #time_chaos_x_LS_list.append(t_val[-1])

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
            time_chaos_x_NN,
            time_chaos_x_Sindy,
            time_chaos_x_LS,
            time_chaos_x_NN_retrain,
            time_chaos_x_NN_SR,
            time_chaos_x_Sindy_ku0,
            time_chaos_x_SR,
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
                       header="nois_th nois nois_db  NN  Sindy LS NN_retrain NN_SR  Sindy_ku0  SR  Parametric" if not file_exists else '',
                       #header="# noise_th noise noise_db t_SR t_parametric" if not file_exists else '',
                       fmt="%.2f", delimiter=" ", comments='')



        # Calculation of RMSE values
        #  print(f"Trial {i+1} : x0={x0_val:.4f} ; v0={v0_val:.4f}")
        print("RMSE values:")
        printF("RMSE values:")
        rmse_x_NN = np.sqrt(np.mean((x_simulated_th[0:len(x_simulated_NN)] - x_simulated_NN) ** 2))
        rmse_x_dot_NN = np.sqrt(np.mean((x_dot_simulated_th[0:len(x_dot_simulated_NN)] - x_dot_simulated_NN) ** 2))
        #rmse_x_ddot = np.sqrt(np.mean((x_ddot_data - x_ddot_pred) ** 2))
        # Print the results
        print(f"NN results   \t - x: {rmse_x_NN:.6f}, x': {rmse_x_dot_NN:.6f}")
        printF(f"NN results   \t - x: {rmse_x_NN:.6f}, x': {rmse_x_dot_NN:.6f}")
        #print(f"RMSE for Position (x): {rmse_x_NN:.6f}")
        #print(f"RMSE for Velocity (x'): {rmse_x_dot_NN:.6f}")
        #print(f"RMSE for Acceleration (x''): {rmse_x_ddot:.6f}")
        rmse_x_NN_list.append(rmse_x_NN)
        rmse_x_dot_NN_list.append(rmse_x_dot_NN)

        rmse_x_NN_retrain = np.sqrt(np.mean((x_simulated_th[0:len(x_simulated_NN_retrain)] - x_simulated_NN_retrain) ** 2))
        rmse_x_dot_NN_retrain = np.sqrt(np.mean((x_dot_simulated_th[0:len(x_dot_simulated_NN_retrain)] - x_dot_simulated_NN_retrain) ** 2))
        print(f"NN results retrained \t - x: {rmse_x_NN_retrain:.6f}, x': {rmse_x_dot_NN_retrain:.6f}")
        printF(f"NN results retrained \t - x: {rmse_x_NN_retrain:.6f}, x': {rmse_x_dot_NN_retrain:.6f}")
        rmse_x_NN_retrain_list.append(rmse_x_NN_retrain)
        rmse_x_dot_NN_retrain_list.append(rmse_x_dot_NN_retrain)
        rmse_x_NN_SR = np.sqrt(np.mean((x_simulated_th[0:len(x_simulated_NN_SR)] - x_simulated_NN_SR) ** 2))
        rmse_x_dot_NN_SR = np.sqrt(np.mean((x_dot_simulated_th[0:len(x_dot_simulated_NN_SR)] - x_dot_simulated_NN_SR) ** 2))
        print(f"NN-SR \t - x: {rmse_x_NN_SR:.6f}, x': {rmse_x_dot_NN_SR:.6f}")
        printF(f"NN-SR \t - x: {rmse_x_NN_SR:.6f}, x': {rmse_x_dot_NN_SR:.6f}")
        rmse_x_NN_SR_list.append(rmse_x_NN_SR)
        rmse_x_dot_NN_SR_list.append(rmse_x_dot_NN_SR)


    #    print("Sindy results")
        if(bool_print_sindy):
            rmse_x_Sindy = np.sqrt(np.mean((x_simulated_th[0:len(x_simulated_Sindy)] - x_simulated_Sindy) ** 2))
            rmse_x_dot_Sindy = np.sqrt(np.mean((x_dot_simulated_th[0:len(x_dot_simulated_Sindy)] - x_dot_simulated_Sindy) ** 2))
            #rmse_x_Sindy = np.sqrt(np.mean((x_simulated_th[0:-1] - x_val_sindy[:,0]) ** 2))
            #rmse_x_dot_Sindy = np.sqrt(np.mean((x_dot_simulated_th[0:-1] - x_val_sindy[:,1]) ** 2))
        else:
            rmse_x_Sindy = 1e6
            rmse_x_dot_Sindy = 1e6
        print(f"Sindy results\t - x: {rmse_x_Sindy:.6f}, x': {rmse_x_dot_Sindy:.6f}")
        printF(f"Sindy results\t - x: {rmse_x_Sindy:.6f}, x': {rmse_x_dot_Sindy:.6f}")
        #print(f"RMSE for Position (x): {rmse_x_sindy:.6f}")
        #print(f"RMSE for Velocity (x'): {rmse_x_dot_sindy:.6f}")
        rmse_x_Sindy_list.append(rmse_x_Sindy)
        rmse_x_dot_Sindy_list.append(rmse_x_dot_Sindy)


     #   print("LS results")
        rmse_x_LS = np.sqrt(np.mean((x_simulated_th[0:len(x_simulated_LS)] - x_simulated_LS) ** 2))
        rmse_x_dot_LS = np.sqrt(np.mean((x_dot_simulated_th[0:len(x_dot_simulated_LS)] - x_dot_simulated_LS) ** 2))
        print(f"LS results   \t - x: {rmse_x_LS:.6f}, x': {rmse_x_dot_LS:.6f}")
        printF(f"LS results   \t - x: {rmse_x_LS:.6f}, x': {rmse_x_dot_LS:.6f}")
        #print(f"RMSE for Position (x): {rmse_x_LS:.6f}")
        #print(f"RMSE for Velocity (x'): {rmse_x_dot_LS:.6f}")
        rmse_x_LS_list.append(rmse_x_LS)
        rmse_x_dot_LS_list.append(rmse_x_dot_LS)

        rmse_matrix_append = np.column_stack([
            noise_percentage_th,
            noise_percentage,
            SNR_dB,
            rmse_x_NN,
            rmse_x_dot_NN,
            rmse_x_Sindy,
            rmse_x_dot_Sindy,
            rmse_x_LS,
            rmse_x_dot_LS,
            rmse_x_NN_retrain,
            rmse_x_dot_NN_retrain,
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
                       header="noise_th noise noise_db rmse_x_NN rmse_x_dot_NN rmse_x_Sindy rmse_x_dot_Sindy rmse_x_LS rmse_x_dot_LS rmse_x_NN_retrain rmse_x_dot_NN_retrain rmse_x_NN_SR rmse_x_dot_NN_SR" if not file_exists else '',
                       fmt="%.8f", delimiter=" ", comments='')


    # Compute overall RMSE and standard deviation
    total_rmse_x_NN = np.mean(rmse_x_NN_list)
    std_rmse_x_NN = np.std(rmse_x_NN_list)
    total_rmse_x_dot_NN = np.mean(rmse_x_dot_NN_list)
    std_rmse_x_dot_NN = np.std(rmse_x_dot_NN_list)
    total_rmse_x_Sindy = np.mean(rmse_x_Sindy_list)
    std_rmse_x_Sindy = np.std(rmse_x_Sindy_list)
    total_rmse_x_dot_Sindy = np.mean(rmse_x_dot_Sindy_list)
    std_rmse_x_dot_Sindy = np.std(rmse_x_dot_Sindy_list)
    total_rmse_x_LS = np.mean(rmse_x_LS_list)
    std_rmse_x_LS = np.std(rmse_x_LS_list)
    total_rmse_x_dot_LS = np.mean(rmse_x_dot_LS_list)
    std_rmse_x_dot_LS = np.std(rmse_x_dot_LS_list)
    total_rmse_x_NN_retrain = np.mean(rmse_x_NN_retrain_list)
    std_rmse_x_NN_retrain = np.std(rmse_x_NN_retrain_list)
    total_rmse_x_dot_NN_retrain = np.mean(rmse_x_dot_NN_retrain_list)
    std_rmse_x_dot_NN_retrain = np.std(rmse_x_dot_NN_retrain_list)
    total_rmse_x_NN_SR = np.mean(rmse_x_NN_SR_list)
    std_rmse_x_NN_SR = np.std(rmse_x_NN_SR_list)
    total_rmse_x_dot_NN_SR = np.mean(rmse_x_dot_NN_SR_list)
    std_rmse_x_dot_NN_SR = np.std(rmse_x_dot_NN_SR_list)

    # Print results
    print("\n======= Total RMSE over all trials (mean ± std, % std) =======")
    print("NN results")
    print(f"Position (x):     {total_rmse_x_NN:.6f} ± {std_rmse_x_NN:.6f}  ({std_rmse_x_NN/total_rmse_x_NN*100:.6f}%)")
    print(f"Velocity (x'):    {total_rmse_x_dot_NN:.6f} ± {std_rmse_x_dot_NN:.6f}  ({std_rmse_x_dot_NN/total_rmse_x_dot_NN*100:.6f}%)")
    print("SINDy results")
    print(f"Position (x):     {total_rmse_x_Sindy:.6f} ± {std_rmse_x_Sindy:.6f}  ({std_rmse_x_Sindy/total_rmse_x_Sindy*100:.6f}%)")
    print(f"Velocity (x'):    {total_rmse_x_dot_Sindy:.6f} ± {std_rmse_x_dot_Sindy:.6f}  ({std_rmse_x_dot_Sindy/total_rmse_x_dot_Sindy*100:.6f}%)")
    print("LS results")
    print(f"Position (x):     {total_rmse_x_LS:.6f} ± {std_rmse_x_LS:.6f}  ({std_rmse_x_LS/total_rmse_x_LS*100:.6f}%)")
    print(f"Velocity (x'):    {total_rmse_x_dot_LS:.6f} ± {std_rmse_x_dot_LS:.6f}  ({std_rmse_x_dot_LS/total_rmse_x_dot_LS*100:.6f}%)")
    print("NN-retrain results")
    print(f"Position (x):     {total_rmse_x_NN_retrain:.6f} ± {std_rmse_x_NN_retrain:.6f}  ({std_rmse_x_NN_retrain/total_rmse_x_NN_retrain*100:.6f}%)")
    print(f"Velocity (x'):    {total_rmse_x_dot_NN_retrain:.6f} ± {std_rmse_x_dot_NN_retrain:.6f}  ({std_rmse_x_dot_NN_retrain/total_rmse_x_dot_NN_retrain*100:.6f}%)")
    print("NN-SR results")
    print(f"Position (x):     {total_rmse_x_NN_SR:.6f} ± {std_rmse_x_NN_SR:.6f}  ({std_rmse_x_NN_retrain/total_rmse_x_NN_SR*100:.6f}%)")
    print(f"Velocity (x'):    {total_rmse_x_dot_NN_SR:.6f} ± {std_rmse_x_dot_NN_SR:.6f}  ({std_rmse_x_dot_NN_retrain/total_rmse_x_dot_NN_SR*100:.6f}%)")
    printF("\n======= Total RMSE over all trials (mean ± std, % std) =======")
    printF("NN results")
    printF(f"Position (x):     {total_rmse_x_NN:.6f} ± {std_rmse_x_NN:.6f}  ({std_rmse_x_NN/total_rmse_x_NN*100:.6f}%)")
    printF(f"Velocity (x'):    {total_rmse_x_dot_NN:.6f} ± {std_rmse_x_dot_NN:.6f}  ({std_rmse_x_dot_NN/total_rmse_x_dot_NN*100:.6f}%)")
    printF("SINDy results")
    printF(f"Position (x):     {total_rmse_x_Sindy:.6f} ± {std_rmse_x_Sindy:.6f}  ({std_rmse_x_Sindy/total_rmse_x_Sindy*100:.6f}%)")
    printF(f"Velocity (x'):    {total_rmse_x_dot_Sindy:.6f} ± {std_rmse_x_dot_Sindy:.6f}  ({std_rmse_x_dot_Sindy/total_rmse_x_dot_Sindy*100:.6f}%)")
    printF("LS results")
    printF(f"Position (x):     {total_rmse_x_LS:.6f} ± {std_rmse_x_LS:.6f}  ({std_rmse_x_LS/total_rmse_x_LS*100:.6f}%)")
    printF(f"Velocity (x'):    {total_rmse_x_dot_LS:.6f} ± {std_rmse_x_dot_LS:.6f}  ({std_rmse_x_dot_LS/total_rmse_x_dot_LS*100:.6f}%)")
    printF("NN-retrain results")
    printF(f"Position (x):     {total_rmse_x_NN_retrain:.6f} ± {std_rmse_x_NN_retrain:.6f}  ({std_rmse_x_NN_retrain/total_rmse_x_NN_retrain*100:.6f}%)")
    printF(f"Velocity (x'):    {total_rmse_x_dot_NN_retrain:.6f} ± {std_rmse_x_dot_NN_retrain:.6f}  ({std_rmse_x_dot_NN_retrain/total_rmse_x_dot_NN_retrain*100:.6f}%)")
    printF("NN-SR results")
    printF(f"Position (x):     {total_rmse_x_NN_SR:.6f} ± {std_rmse_x_NN_SR:.6f}  ({std_rmse_x_NN_retrain/total_rmse_x_NN_SR*100:.6f}%)")
    printF(f"Velocity (x'):    {total_rmse_x_dot_NN_SR:.6f} ± {std_rmse_x_dot_NN_SR:.6f}  ({std_rmse_x_dot_NN_retrain/total_rmse_x_dot_NN_SR*100:.6f}%)")


    # These lists must already be defined
    # Each one should contain RMSE values over multiple trials
    # e.g., rmse_x_NN_list = [rmse_trial1, rmse_trial2, ..., rmse_trialN]
    # same for other methods
    rmse_data = [
        rmse_x_NN_list,
        rmse_x_Sindy_list,
        rmse_x_LS_list,
        rmse_x_NN_retrain_list,
        rmse_x_NN_SR_list
    ]
    labels = ['NN-CC', 'SINDy', 'Poly-CC', 'NN-CC retrain','NN-CC-SR']
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
        rmse_x_dot_NN_list,
        rmse_x_dot_Sindy_list,
        rmse_x_dot_LS_list,
        rmse_x_dot_NN_retrain_list,
        rmse_x_dot_NN_SR_list
    ]
    labels = ['NN-CC', 'SINDy', 'Poly-CC','NN-CC retrain','NN-CC-SR']
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
        rmse_x_NN_list,
        rmse_x_dot_NN_list,
        rmse_x_Sindy_list,
        rmse_x_dot_Sindy_list,
        rmse_x_LS_list,
        rmse_x_dot_LS_list,
        rmse_x_NN_retrain_list,
        rmse_x_dot_NN_retrain_list,
        rmse_x_NN_SR_list,
        rmse_x_dot_NN_SR_list
    ])
    folder_path = output_path
    os.makedirs(folder_path, exist_ok=True)
    file_name = "rmse_results_duffing_NN_without_symmetry.txt"
    file_path = os.path.join(folder_path, file_name)
    # Save with header and space as delimiter
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a') as f:
        np.savetxt(f, rmse_matrix,
                   header="noise_th noise_meas noise_db rmse_x_NN rmse_x_dot_NN rmse_x_Sindy rmse_x_dot_Sindy rmse_x_LS rmse_x_dot_LS rmse_x_NN_retrain rmse_x_dot_NN_retrain rmse_x_NN_SR rmse_x_dot_NN_SR" if not file_exists else '',
                   fmt="%.8f", delimiter=" ", comments='')
    #np.savetxt(file_path, rmse_matrix,
    #           header="rmse_x_NN rmse_x_dot_NN rmse_x_Sindy rmse_x_dot_Sindy rmse_x_LS rmse_x_dot_LS",
    #           fmt="%.8f", delimiter=" ")



#test_predictions = model(test_inputs).cpu().numpy()  # Move predictions to CPU for plotting



