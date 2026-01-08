import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy.linalg import norm

# Duffing parameters
delta = 0.3   # damping
alpha = -1.0  # linear stiffness
beta = 1.0    # cubic stiffness
F0 = 0.5      # forcing amplitude
Omega = 1.2   # forcing frequency
x0=0.5
v0=-0.5

def duffing(t, y):
    x, x_dot = y
    x_ddot = -delta * x_dot - alpha * x - beta * x**3 + F0 * np.cos(Omega * t)
    return [x_dot, x_ddot]

# Parameters for integration
t_span = (0, 50)
t_eval = np.linspace(*t_span, 50000)
y0 = [x0, v0]  # initial condition

# Integrate
sol = solve_ivp(duffing, t_span, y0, t_eval=t_eval, method='LSODA')
x = sol.y[0]  # shape: (n_steps, 3)
# Plot
plt.plot(sol.t, sol.y[0])
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.title("Duffing Oscillator")
plt.grid(True)
plt.show()

# -------- Lyapunov exponent calculation -------- #
def lyapunov_exponent(f, t_span, y0, dt, perturb=1e-8):
    y1 = np.array(y0, dtype=float)
    y2 = y1 + perturb*np.random.normal(size=len(y0))  # nearby initial condition
    t0, tf = t_span
    t = t0
    sum_log = 0
    count = 0

    while t < tf:
        # Step both trajectories
        sol1 = solve_ivp(f, (t, t+dt), y1, t_eval=[t+dt], method='LSODA')
        sol2 = solve_ivp(f, (t, t+dt), y2, t_eval=[t+dt], method='LSODA')
        y1 = sol1.y[:, -1]
        y2 = sol2.y[:, -1]

        # Distance
        dist = np.linalg.norm(y2 - y1)
        if dist == 0:
            dist = 1e-16

        # Renormalize separation
        sum_log += np.log(dist / perturb)
        count += 1
        y2 = y1 + perturb * (y2 - y1) / dist

        t += dt

    return sum_log / (count * dt)

# Compute largest Lyapunov exponent
dt = 0.1
lambda_max = lyapunov_exponent(duffing, (0, 200), y0, dt)
lyapunov_time = 1 / lambda_max if lambda_max > 0 else np.inf

print(f"Largest Lyapunov exponent: {lambda_max:.5f}")
print(f"Lyapunov time: {lyapunov_time:.5f}")




print("Now computing multiple trayectories and average of Lyapunov")



def lyap_max_two_traj(f, y0, burn_in=200.0, T=400.0, tau=0.5,
                      delta0=1e-7, rtol=1e-9, atol=1e-12, method='LSODA',
                      n_segments=1, reseed=True):
    """
    Largest Lyapunov exponent via two-trajectory shadow + periodic renormalization.
    Returns mean and std across segments, and per-segment values.
    """
    rng = np.random.default_rng(0)
    # 1) Burn-in to reach attractor
    sol_burn = solve_ivp(f, (0, burn_in), y0, method=method, rtol=rtol, atol=atol)
    y_base = sol_burn.y[:, -1]

    vals = []
    for s in range(n_segments):
        # Optionally reseed the perturbation direction each segment
        d0 = rng.normal(size=len(y_base)) if reseed else np.array([1.0, 0.0])
        d0 = d0 / np.linalg.norm(d0) * delta0
        y1 = y_base.copy()
        y2 = y1 + d0

        t = 0.0
        t_end = T
        sum_log = 0.0
        k = 0

        while t < t_end:
            # Advance both by exactly tau
            sol1 = solve_ivp(f, (t, min(t+tau, t_end)), y1, t_eval=[min(t+tau, t_end)],
                             method=method, rtol=rtol, atol=atol)
            sol2 = solve_ivp(f, (t, min(t+tau, t_end)), y2, t_eval=[min(t+tau, t_end)],
                             method=method, rtol=rtol, atol=atol)

            y1 = sol1.y[:, -1]
            y2 = sol2.y[:, -1]

            # Measure separation and renormalize
            d = y2 - y1
            dist = np.linalg.norm(d)
            if dist <= 0:
                dist = 1e-300  # avoid log(0)

            sum_log += np.log(dist / delta0)
            k += 1
            d = (d / dist) * delta0
            y2 = y1 + d

            t += tau

        # λ ≈ (1/(k * τ)) Σ log(||δ_k|| / δ0)
        lam = sum_log / (k * tau)
        vals.append(lam)

        # Move base point forward a bit between segments to decorrelate
        if s < n_segments - 1:
            sol_step = solve_ivp(f, (0, 5.0), y1, method=method, rtol=rtol, atol=atol)
            y_base = sol_step.y[:, -1]

    vals = np.array(vals)
    return vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0, vals



y0 = [x0, v0]
mean_lam, std_lam, per_seg = lyap_max_two_traj(
    duffing, y0,
    burn_in=200.0,   # discard transient
    T=400.0,         # measuring time per segment
    tau=0.5,         # renormalization interval (a few periods works well)
    delta0=1e-7,
    rtol=1e-9, atol=1e-12,
    n_segments=10,    # average across a few segments
    reseed=True
)
lyap_time = 1/mean_lam if mean_lam > 0 else np.inf
print(f"λ_max = {mean_lam:.6f} ± {std_lam:.6f}  (Lyapunov time ≈ {lyap_time:.3f})")
print("per-segment:", per_seg)







print("Now computing the Lyapunov time from some standard method")

print("Wolf method")


# ---------------- Wolf-like Lyapunov (two trajectories) ----------------
def lyap_wolf_continuous(f, y0, t0=0.0, t_transient=10.0, t_meas=30.0,
                         tau=0.1, delta0=1e-8, solver_method='LSODA',
                         rtol=1e-9, atol=1e-12, max_iters=100000):
    """
    Wolf-style estimate of largest Lyapunov exponent by integrating two nearby trajectories.
    - f: function(t, y) returning dy/dt
    - y0: initial condition (array-like, length n)
    - t_transient: time to integrate and discard to reach attractor
    - t_meas: time over which to measure (after transient)
    - tau: renormalization interval (seconds)
    - delta0: initial perturbation size (Euclidean)
    Returns: lambda_est (1/time), lyap_time, diagnostics dict
    """
    y0 = np.asarray(y0, dtype=float)
    # 1) integrate transient to reach attractor
    sol_tr = solve_ivp(f, (t0, t0 + t_transient), y0, method=solver_method, rtol=rtol, atol=atol, t_eval=[t0 + t_transient])
    y_ref = sol_tr.y[:, -1]

    # 2) create perturbed initial condition
    rng = np.random.default_rng(12345)
    dvec = rng.normal(size=y_ref.shape)
    dvec = dvec / np.linalg.norm(dvec) * delta0
    y_pert = y_ref + dvec

    t_curr = t0 + t_transient
    t_end = t_curr + t_meas

    sum_log = 0.0
    n_renorm = 0

    # main loop: advance both by tau and renormalize
    while t_curr < t_end and n_renorm < max_iters:
        t_next = min(t_curr + tau, t_end)

        # advance both trajectories to t_next
        sol1 = solve_ivp(f, (t_curr, t_next), y_ref, t_eval=[t_next], method=solver_method, rtol=rtol, atol=atol)
        sol2 = solve_ivp(f, (t_curr, t_next), y_pert, t_eval=[t_next], method=solver_method, rtol=rtol, atol=atol)

        y_ref = sol1.y[:, -1]
        y_pert = sol2.y[:, -1]

        # separation
        d = y_pert - y_ref
        dist = np.linalg.norm(d)
        if dist <= 0:
            # extremely unlikely due to numerical noise but guard anyway
            dist = 1e-300

        # accumulate log growth relative to delta0
        sum_log += np.log(dist / delta0)
        n_renorm += 1

        # renormalize perturbed trajectory to distance delta0 (keep direction)
        d = (d / dist) * delta0
        y_pert = y_ref + d

        t_curr = t_next

    if n_renorm == 0:
        raise RuntimeError("No renormalization events — increase t_meas or decrease tau.")

    # lambda = (sum log(dist/d0)) / (n_renorm * tau)
    lambda_est = sum_log / (n_renorm * tau)
    lyap_time = np.inf if lambda_est <= 0 else 1.0 / lambda_est

    diagnostics = dict(n_events=n_renorm, total_time=n_renorm * tau, sum_log=sum_log)
    return lambda_est, lyap_time, diagnostics

# ---------------- Run Wolf on the Duffing system ----------------
lambda_est, T_L, info = lyap_wolf_continuous(
    duffing, y0=y0,
    t_transient=10.0,   # discard initial 10 s
    t_meas=30.0,        # measure for 30 s
    tau=0.05,           # renormalize every 0.05 s
    delta0=1e-8,        # small initial perturbation
    solver_method='LSODA'
)

print("Wolf-style estimate:")
print(f"  lambda = {lambda_est:.6f}  (1/time unit)")
print(f"  Lyapunov time T_L = {T_L:.6f} time units")
print("  diagnostics:", info)



print(" ")
print("Now we compute Wolf method but averaging")

def lyap_wolf_average(f, y0, n_traj=5, t_transient=10.0, t_meas=30.0,
                      tau=0.05, delta0=1e-8, solver_method='LSODA',
                      rtol=1e-9, atol=1e-12):
    """
    Wolf method with multiple perturbed trajectories averaged.

    Parameters
    ----------
    f : callable
        Function f(t, y) returning dy/dt.
    y0 : array_like
        Initial condition.
    n_traj : int
        Number of perturbed trajectories to average.
    t_transient : float
        Integration time to reach attractor.
    t_meas : float
        Measurement time for divergence.
    tau : float
        Renormalization interval.
    delta0 : float
        Initial perturbation magnitude.
    solver_method : str
        Method for solve_ivp.
    rtol, atol : float
        Solver tolerances.

    Returns
    -------
    lambda_avg : float
        Estimated largest Lyapunov exponent.
    T_L_avg : float
        Corresponding Lyapunov time.
    """
    y0 = np.asarray(y0, dtype=float)

    # Integrate transient to reach attractor
    sol_tr = solve_ivp(f, (0, t_transient), y0, t_eval=[t_transient],
                       method=solver_method, rtol=rtol, atol=atol)
    y_ref = sol_tr.y[:, -1]

    # Initialize perturbed trajectories
    rng = np.random.default_rng(12345)
    y_pert_all = []
    for _ in range(n_traj):
        dvec = rng.normal(size=y_ref.shape)
        dvec = dvec / np.linalg.norm(dvec) * delta0
        y_pert_all.append(y_ref + dvec)
    y_pert_all = np.array(y_pert_all)

    n_steps = int(np.ceil(t_meas / tau))
    sum_logs = np.zeros(n_traj)

    y_ref_curr = y_ref.copy()

    for step in range(n_steps):
        t_start = t_transient + step * tau
        t_end = t_start + tau

        # Integrate reference trajectory
        sol_ref = solve_ivp(f, (t_start, t_end), y_ref_curr, t_eval=[t_end],
                            method=solver_method, rtol=rtol, atol=atol)
        y_ref_curr = sol_ref.y[:, -1]

        # Integrate all perturbed trajectories
        for i in range(n_traj):
            sol_pert = solve_ivp(f, (t_start, t_end), y_pert_all[i], t_eval=[t_end],
                                 method=solver_method, rtol=rtol, atol=atol)
            y_pert_all[i] = sol_pert.y[:, -1]

            # Compute distance and accumulate log
            dvec = y_pert_all[i] - y_ref_curr
            dist = np.linalg.norm(dvec)
            dist = max(dist, 1e-300)
            sum_logs[i] += np.log(dist / delta0)

            # Renormalize
            y_pert_all[i] = y_ref_curr + dvec / dist * delta0

    # Average over trajectories and steps
    lambda_avg = np.mean(sum_logs) / (n_steps * tau)
    T_L_avg = np.inf if lambda_avg <= 0 else 1.0 / lambda_avg

    return lambda_avg, T_L_avg
# Define Duffing as before
def duffing(t, y):
    x, v = y
    return [v, -0.3*v + 1.0*x - 1.0*x**3 + 0.5*np.cos(1.2*t)]

y0 = [0.5, -0.5]

lambda_avg, T_L_avg = lyap_wolf_average(duffing, y0, n_traj=10,
                                        t_transient=10, t_meas=50,
                                        tau=0.05, delta0=1e-8)
print(f"Wolf-average: lambda = {lambda_avg:.6f}, T_L = {T_L_avg:.6f} s")


# ---------- Rosenstein (Physica D, 1993) ----------

def time_delay_embed(x, m, tau):
    x = np.asarray(x, dtype=float).flatten()
    N = x.size
    N_eff = N - (m - 1) * tau
    if N_eff <= 0:
        raise ValueError(f"Time series too short for m={m}, tau={tau}. Need length > {(m-1)*tau}.")
    # Build matrix with shape (N_eff, m)
    X = np.empty((N_eff, m), dtype=float)
    for i in range(m):
        X[:, i] = x[i * tau : i * tau + N_eff]
    return X

def theiler_mask(i, N, W):
    mask = np.ones(N, dtype=bool)
    lo, hi = max(0, i - W), min(N, i + W + 1)
    mask[lo:hi] = False
    return mask

def lyap_rosenstein(x, dt, m=10, tau=2, W=50, kmax=200, kfit=(5, 30)):
    """
    Rosenstein et al. largest Lyapunov estimator from a scalar time series x sampled at interval dt.
    Returns lam (1/time) and (t_k, mean_log_d) for plotting.
    """
    x = np.asarray(x).flatten()
    N = x.size

    # auto-adjust m,tau if too long for data
    max_m_possible = max(1, (N - 1) // tau)
    if m > max_m_possible:
        m = max(1, max_m_possible)
        print(f"[Rosenstein] Warning: reduced embedding m to {m} due to series length.")

    X = time_delay_embed(x, m, tau)
    M = X.shape[0]

    # find nearest neighbor index for each reference point (excluding Theiler window)
    nn_index = np.empty(M, dtype=int)
    for i in range(M):
        mask = theiler_mask(i, M, W)
        if not np.any(mask):
            raise RuntimeError("Theiler window too large or data too short.")
        d = norm(X[mask] - X[i], axis=1)
        j_rel = np.argmin(d)
        j = np.arange(M)[mask][j_rel]
        nn_index[i] = j

    # determine maximum k such that i+k and j+k are valid
    max_k_possible = np.min([M - 1 - np.arange(M), M - 1 - nn_index])
    kk = int(np.median(max_k_possible))
    kk = min(kk, kmax)
    if kk < kfit[1]:
        # try to extend kk to fit range if possible
        kk = min(kmax, M//3)
    if kk <= 1:
        raise RuntimeError("Not enough forward steps available for Rosenstein calculation.")

    log_d = []
    valid_counts = []
    for k in range(kk + 1):
        valid_mask = (np.arange(M) + k < M) & (nn_index + k < M)
        idxs = np.arange(M)[valid_mask]
        if idxs.size == 0:
            log_d.append(np.nan)
            valid_counts.append(0)
            continue
        di = norm(X[idxs + k] - X[nn_index[valid_mask] + k], axis=1)
        di = np.where(di <= 1e-300, 1e-300, di)
        log_d.append(np.mean(np.log(di)))
        valid_counts.append(idxs.size)

    log_d = np.array(log_d)
    t_k = np.arange(kk + 1) * dt

    # fit linear region t_k[k0:k1]
    k0, k1 = kfit
    if k1 > kk:
        k1 = kk
        print(f"[Rosenstein] Warning: fit upper bound reduced to {k1} (kk={kk}).")
    if k0 >= k1:
        k0 = 1
        k1 = min(5, kk)

    # linear fit (slope vs time)
    # we fit log_d[k0:k1+1] vs t_k[k0:k1+1]
    valid_slice = ~np.isnan(log_d[k0:k1+1])
    if valid_slice.sum() < 2:
        raise RuntimeError("Not enough valid points in fit range for Rosenstein method.")
    p = np.polyfit(t_k[k0:k1+1][valid_slice], log_d[k0:k1+1][valid_slice], 1)
    lam = p[0]  # slope d/dt of mean log distance
    return lam, (t_k, log_d), (k0, k1), p

# choose embedding params (basic defaults)
# reasonable choice: tau roughly a fraction of forcing period; here choose quarter period in samples
period = 2 * np.pi / Omega
tau_samples = max(1, int(round((period / 4.0) / dt)))
m = 10
W = int(round(period / dt))  # Theiler window: exclude at least one forcing period
kfit = (5, 40)  # fit region in steps (adjust if needed)

print(f"Rosenstein embedding: m={m}, tau={tau_samples}, Theiler W={W}, dt={dt:.5e}")

# compute Rosenstein
try:
    lam_ros, (t_k, log_d), (k0, k1), poly = lyap_rosenstein(x, dt, m=m, tau=tau_samples, W=W, kmax=200, kfit=kfit)
    T_L_ros = np.inf if lam_ros <= 0 else 1.0 / lam_ros
    print("\nRosenstein estimate (from time series):")
    print(f"  lambda = {lam_ros:.6f}  (1/time units)")
    print(f"  Lyapunov time T_L = {T_L_ros:.6f} time units")
    print(f"  fit k-range = {k0} .. {k1}, poly slope = {poly[0]:.6e}, intercept = {poly[1]:.6e}")

    # Plot mean log divergence and fitted line
    plt.figure(figsize=(6,4))
    plt.plot(t_k, log_d, 'o-', markersize=3, label='mean ln(d(k))')
    # fitted line over fit window
    t_fit = t_k[k0:k1+1]
    y_fit = np.polyval(poly, t_fit)
    plt.plot(t_fit, y_fit, 'r--', lw=2, label=f'linear fit (slope={poly[0]:.4e})')
    plt.xlabel('time (s)')
    plt.ylabel('mean ln distance')
    plt.title('Rosenstein divergence curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print("Rosenstein method failed:", e)
    lam_ros = None
    T_L_ros = None




