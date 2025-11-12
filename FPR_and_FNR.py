import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from joblib import Parallel, delayed
import itertools
import time
import os
import matplotlib.pyplot as plt

np.random.seed(42)

# --- 1. Simulation Constants ---

# Parameters for the simulation itself
N_SIMULATIONS = 500_000  # 500k for final accuracy
ALPHA = 0.05 # This is now just the *default* for the main plots
RHO_STEPS = 50           # Number of correlation points to sweep (from 0.01 to 1.0)


# "In-vivo" assumptions for the model
MU_BIO = 2.1             # Mean of the true biological signal (set to 0 for simplicity)
SIGMA_BIO = 1.0          # Variance of the true signal (set to 1 as our unit variance)
MU_CONTAM = 0.0          # Mean of the 'common' contamination (cancels out, set to 0)
FIXED_VR = 1.0           # Fix VR=1.0 to avoid truncation and test other params

# --- 2. Grid Parameters ---
N_SAMPLES_LIST = [10, 25, 100]
R_SIGMA_VALUES = [0.5, 1.0, 2.0]

# List of effect sizes to sweep for Delta (FNR) and delta (FPR)
EFFECT_SIZE_LIST = np.linspace(0.2, 2.2, 11)

# NEW: Alpha sweep for the final plot
ALPHA_SWEEP_LIST = np.logspace(-4, -0.7, 50) # Sweep from 0.0001 to ~0.2

# --- 3. Output Paths ---
OUTPUT_PATH = r'C:\ColumbiaWork\PapersOnGo\MEGAvsNon\MEGAvsNon\Figures'
OUTPUT_DATA_PATH = "simulation_results_final.csv"
OUTPUT_FIGURE_PATH_FPR = "simulation_FPR_final.png"
OUTPUT_FIGURE_PATH_FNR = "simulation_FNR_final.png"
OUTPUT_FIGURE_PATH_ALPHA = "simulation_alpha_tradeoff.png" # Path for the new plot

# Check if path exists, otherwise save to local directory
if OUTPUT_PATH is not None and os.path.exists(OUTPUT_PATH):
    OUTPUT_DATA_PATH = os.path.join(OUTPUT_PATH, OUTPUT_DATA_PATH)
    OUTPUT_FIGURE_PATH_FPR = os.path.join(OUTPUT_PATH, OUTPUT_FIGURE_PATH_FPR)
    OUTPUT_FIGURE_PATH_FNR = os.path.join(OUTPUT_PATH, OUTPUT_FIGURE_PATH_FNR)
    OUTPUT_FIGURE_PATH_ALPHA = os.path.join(OUTPUT_PATH, OUTPUT_FIGURE_PATH_ALPHA)
else:
    print(f"Warning: OUTPUT_PATH '{OUTPUT_PATH}' not found. Saving to current directory.")
    OUTPUT_DATA_PATH = "simulation_results_final.csv"
    OUTPUT_FIGURE_PATH_FPR = "simulation_FPR_final.png"
    OUTPUT_FIGURE_PATH_FNR = "simulation_FNR_final.png"
    OUTPUT_FIGURE_PATH_ALPHA = "simulation_alpha_tradeoff.png"


def run_main_simulation(n_samples, r_sigma, rho):
    
    """
    Runs a full, vectorized Monte Carlo simulation for a single (n_samples, r_sigma, rho) point.
    VR is fixed to 1.0 inside this function.
    
    INTERNALLY, it loops over all delta and Delta values from EFFECT_SIZE_LIST.
    
    Returns a LIST of (n_samples, r_sigma, rho, rate_type, effect_size, rate) tuples.
    """

    np.random.seed(int((n_samples * 1000) + (r_sigma * 100) + (rho * 10000)))

    # --- A. Calculate Model Coefficients (with VR=1.0) ---
    vr = FIXED_VR # Hardcode VR=1.0
    
    if not (0.0 <= rho <= 1.0):
         return []

    x1 = rho * np.sqrt(vr)
    x2_squared_arg = vr * (1 - rho**2) / r_sigma
    x2 = 0.0 if x2_squared_arg < 0 else np.sqrt(x2_squared_arg)
    sigma_contam = np.sqrt(r_sigma * (SIGMA_BIO**2))

    results_list = []

    # --- B. Run False Positive Rate (FPR) Simulations (looping over delta) ---
    
    S_A_fpr = np.random.normal(loc=MU_BIO, scale=SIGMA_BIO, 
                               size=(n_samples, N_SIMULATIONS))
    zeta_A = np.random.normal(loc=MU_CONTAM, scale=sigma_contam, 
                              size=(n_samples, N_SIMULATIONS))
    P_A_fpr = x1 * S_A_fpr + x2 * zeta_A

    for delta in EFFECT_SIZE_LIST:
        zeta_B_fpr = np.random.normal(loc=MU_CONTAM + delta, scale=sigma_contam, 
                                      size=(n_samples, N_SIMULATIONS))
        P_B_fpr = x1 * S_A_fpr + x2 * zeta_B_fpr
        
        _, p_values_fpr = ttest_ind(P_A_fpr, P_B_fpr, axis=0, equal_var=False)
        fpr = np.sum(p_values_fpr < ALPHA) / N_SIMULATIONS
        
        results_list.append((n_samples, r_sigma, rho, "FPR", delta, fpr))

    
    # --- C. Run False Negative Rate (FNR) Simulations (looping over Delta) ---
    
    S_A_fnr = np.random.normal(loc=MU_BIO, scale=SIGMA_BIO, 
                               size=(n_samples, N_SIMULATIONS))
    zeta_A_fnr = np.random.normal(loc=MU_CONTAM, scale=sigma_contam, 
                                  size=(n_samples, N_SIMULATIONS))
    zeta_B_fnr = np.random.normal(loc=MU_CONTAM, scale=sigma_contam, 
                                  size=(n_samples, N_SIMULATIONS))
    P_A_fnr = x1 * S_A_fnr + x2 * zeta_A_fnr

    for Delta in EFFECT_SIZE_LIST:
        S_B_fnr = np.random.normal(loc=MU_BIO + Delta, scale=SIGMA_BIO, 
                                   size=(n_samples, N_SIMULATIONS))
        P_B_fnr = x1 * S_B_fnr + x2 * zeta_B_fnr
        
        _, p_values_fnr = ttest_ind(P_A_fnr, P_B_fnr, axis=0, equal_var=False)
        fnr = np.sum(p_values_fnr >= ALPHA) / N_SIMULATIONS

        results_list.append((n_samples, r_sigma, rho, "FNR", Delta, fnr))

    return results_list


def run_alpha_tradeoff_simulation(n_samples, r_sigma, rho, fixed_Delta, fixed_delta):
    """
    Runs a single, large simulation for a fixed scenario and returns the
    raw p-value arrays, from which we can calculate the alpha trade-off.
    """
    vr = FIXED_VR
    x1 = rho * np.sqrt(vr)
    x2_squared_arg = vr * (1 - rho**2) / r_sigma
    x2 = 0.0 if x2_squared_arg < 0 else np.sqrt(x2_squared_arg)
    sigma_contam = np.sqrt(r_sigma * (SIGMA_BIO**2))

    # --- FPR P-Value Generation ---
    S_A_fpr = np.random.normal(loc=MU_BIO, scale=SIGMA_BIO, 
                               size=(n_samples, N_SIMULATIONS))
    zeta_A_fpr = np.random.normal(loc=MU_CONTAM, scale=sigma_contam, 
                                  size=(n_samples, N_SIMULATIONS))
    zeta_B_fpr = np.random.normal(loc=MU_CONTAM + fixed_delta, scale=sigma_contam, 
                                  size=(n_samples, N_SIMULATIONS))
    P_A_fpr = x1 * S_A_fpr + x2 * zeta_A_fpr
    P_B_fpr = x1 * S_A_fpr + x2 * zeta_B_fpr
    _, p_values_fpr = ttest_ind(P_A_fpr, P_B_fpr, axis=0, equal_var=False)
    
    # --- FNR P-Value Generation ---
    S_A_fnr = np.random.normal(loc=MU_BIO, scale=SIGMA_BIO, 
                               size=(n_samples, N_SIMULATIONS))
    S_B_fnr = np.random.normal(loc=MU_BIO + fixed_Delta, scale=SIGMA_BIO, 
                               size=(n_samples, N_SIMULATIONS))
    zeta_A_fnr = np.random.normal(loc=MU_CONTAM, scale=sigma_contam, 
                                  size=(n_samples, N_SIMULATIONS))
    zeta_B_fnr = np.random.normal(loc=MU_CONTAM, scale=sigma_contam, 
                                  size=(n_samples, N_SIMULATIONS))
    P_A_fnr = x1 * S_A_fnr + x2 * zeta_A_fnr
    P_B_fnr = x1 * S_B_fnr + x2 * zeta_B_fnr
    _, p_values_fnr = ttest_ind(P_A_fnr, P_B_fnr, axis=0, equal_var=False)

    return p_values_fpr, p_values_fnr


# --- 5. Main Execution ---
if __name__ == "__main__":

    print("--- Running Main 3x3 Grid Simulations ---")
    print(f"CPUs available: {os.cpu_count()}")
    start_time = time.time()

    # --- A. Define the Grid Parameters ---
    linear_part = np.linspace(0, 0.9, int(RHO_STEPS * 0.8))
    denser_part = np.linspace(0.9, 1.0, int(RHO_STEPS * 0.2) + 1)
    RHO_VALUES = np.unique(np.concatenate((linear_part, denser_part)))

    tasks = list(itertools.product(N_SAMPLES_LIST, R_SIGMA_VALUES, RHO_VALUES))
    print(f"Total simulation tasks to run: {len(tasks)}")
    print(f"Sweeping N_samples = {N_SAMPLES_LIST}, R_sigma = {R_SIGMA_VALUES}")
    print(f"VR fixed at {FIXED_VR}, ALPHA fixed at {ALPHA}")

    # --- B. Run Simulations in Parallel ---
    results = Parallel(n_jobs=-2, verbose=10)(
        delayed(run_main_simulation)(n_samples, r_sigma, rho) for n_samples, r_sigma, rho in tasks
    )

    # --- C. Process and Save Results ---
    flat_results = [item for sublist in results for item in sublist]
    
    df = pd.DataFrame(flat_results, 
                      columns=['N_samples', 'R_sigma', 'rho', 'Rate_Type', 'Effect_Size', 'Rate'])
    
    df = df.dropna()
    df = df.sort_values(by=['N_samples', 'R_sigma', 'Rate_Type', 'Effect_Size', 'rho'])
    df.to_csv(OUTPUT_DATA_PATH, index=False)

    end_time = time.time()
    print("\n--- All Simulations Complete ---")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to {OUTPUT_DATA_PATH}")

    # --- D. Create and Save Main Plots ---
    print("Generating main plots...")
    
    # --- FPR PLOT (3 Rows, 3 Columns) ---
    fig_fpr, axes_fpr = plt.subplots(nrows=3, ncols=3, figsize=(15, 12), sharex=False, sharey=False)
    
    for i, r_sigma in enumerate(R_SIGMA_VALUES):
        for j, n_samples in enumerate(N_SAMPLES_LIST):
            ax = axes_fpr[i, j]
            data_subset = df[(df['N_samples'] == n_samples) & 
                             (df['R_sigma'] == r_sigma) & 
                             (df['Rate_Type'] == 'FPR')]
            
            for k, delta in enumerate(EFFECT_SIZE_LIST):
                line_data = data_subset[data_subset['Effect_Size'] == delta]
                ax.plot(line_data['rho'], line_data['Rate'], label=f"δ={delta:.1f}")

            ax.set_title(f'$N_{{{{samples}}}} = {n_samples}$, $R_\\sigma = {r_sigma}$')
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(-0.02, 1.02)
            
            if (i == 2) and (j == 1):
                ax.set_xlabel('Correlation (ρ) (unitless)', fontsize=12)
            if (j == 0) and (i == 1):
                ax.set_ylabel('False Positive Rate (FPR) (unitless)', fontsize=12)

    legend_string = [f'δ = {x:.1f}' for x in EFFECT_SIZE_LIST]
    axes_fpr[2, 2].legend(legend_string, loc='center left', bbox_to_anchor=(1.02, 1.7), fancybox=True)
    # plt.tight_layout(rect=[0.05, 0.05, 0.9, 0.95])
    
    if OUTPUT_FIGURE_PATH_FPR:
        plt.savefig(OUTPUT_FIGURE_PATH_FPR, dpi=300, bbox_inches='tight')
        print(f"FPR plot saved to {OUTPUT_FIGURE_PATH_FPR}")
    else:
        plt.show()

    # --- FNR PLOT (1 Row, 3 Columns) ---
    fig_fnr, axes_fnr = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharex=False, sharey=False)

    if len(N_SAMPLES_LIST) == 1:
        axes_fnr_flat = [axes_fnr]
    else:
        axes_fnr_flat = axes_fnr.flatten()

    for j, n_samples in enumerate(N_SAMPLES_LIST):
        ax = axes_fnr_flat[j]
        data_subset = df[(df['N_samples'] == n_samples) & 
                         (df['R_sigma'] == 1.0) &
                         (df['Rate_Type'] == 'FNR')]
        
        for k, Delta in enumerate(EFFECT_SIZE_LIST):
            line_data = data_subset[data_subset['Effect_Size'] == Delta]
            ax.plot(line_data['rho'], line_data['Rate'], label=f"Δ={Delta:.1f}")

        ax.set_title(f'$N_{{{{samples}}}} = {n_samples}$')
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.02, 1.02)

        if j == 1:
            ax.set_xlabel('Correlation (ρ) (unitless)', fontsize=12)
        if j == 0:
            ax.set_ylabel('False Negative Rate (FNR) (unitless)', fontsize=12)

    legend_string = [f'Δ = {x:.1f}' for x in EFFECT_SIZE_LIST]
    axes_fnr_flat[-1].legend(legend_string, loc='center left', bbox_to_anchor=(1.02, 0.5), fancybox=True)
    plt.tight_layout(rect=[0.05, 0.05, 0.85, 0.9])
    
    if OUTPUT_FIGURE_PATH_FNR:
        plt.savefig(OUTPUT_FIGURE_PATH_FNR, dpi=300, bbox_inches='tight')
        print(f"FNR plot saved to {OUTPUT_FIGURE_PATH_FNR}")
    else:
        plt.show()

    # --- 5. NEW: Alpha Trade-off Plot ---
    print("\n--- Running Alpha Trade-off Simulation ---")
    start_time_alpha = time.time()
    
    # Define the "middle panel" scenario
    FIXED_N_SAMPLES = 25
    FIXED_R_SIGMA_ALPHA = 1.0
    FIXED_RHO_ALPHA = 0.72  # Empirical correlation at 7T
    FIXED_DELTA_ALPHA = 1.0 # A reasonable true effect
    FIXED_delta_ALPHA = 1.0 # A reasonable bias
    
    print(f"Parameters: N={FIXED_N_SAMPLES}, R_sigma={FIXED_R_SIGMA_ALPHA}, rho={FIXED_RHO_ALPHA}, Delta={FIXED_DELTA_ALPHA}, delta={FIXED_delta_ALPHA}")
    
    # Run the expensive simulation once
    p_values_fpr, p_values_fnr = run_alpha_tradeoff_simulation(
        FIXED_N_SAMPLES, 
        FIXED_R_SIGMA_ALPHA, 
        FIXED_RHO_ALPHA, 
        FIXED_DELTA_ALPHA, 
        FIXED_delta_ALPHA
    )
    
    # Now, calculate rates by looping over alpha (this is fast)
    alpha_results = []
    for alpha_val in ALPHA_SWEEP_LIST:
        fpr = np.sum(p_values_fpr < alpha_val) / N_SIMULATIONS
        fnr = np.sum(p_values_fnr >= alpha_val) / N_SIMULATIONS
        alpha_results.append((alpha_val, fpr, fnr))
        
    df_alpha = pd.DataFrame(alpha_results, columns=['alpha', 'FPR', 'FNR'])
    
    print(f"Alpha simulation complete in {time.time() - start_time_alpha:.2f} seconds")

    # --- Plot the alpha trade-off ---
    fig_alpha, ax_alpha = plt.subplots(figsize=(8, 6))
    
    # Use colorblind-safe colors: Orange (#E69F00) and Sky Blue (#56B4E9)
    ax_alpha.plot(df_alpha['alpha'], df_alpha['FPR'], color='#E69F00', linestyle='-', label='False Positive Rate (FPR)', lw=2)
    ax_alpha.plot(df_alpha['alpha'], df_alpha['FNR'], color='#56B4E9', linestyle='-', label='False Negative Rate (FNR)', lw=2)
    
    ax_alpha.set_xscale('log')
    ax_alpha.set_xlabel('Nominal false-positive rate, (α) (unitless)', fontsize=12)
    ax_alpha.set_ylabel('Rate (unitless)', fontsize=12)
    ax_alpha.grid(True, which='both', linestyle=':', alpha=0.7)
    ax_alpha.legend()
    ax_alpha.set_ylim(-0.05, 1.05)
    

    plt.tight_layout()
    
    if OUTPUT_FIGURE_PATH_ALPHA:
        plt.savefig(OUTPUT_FIGURE_PATH_ALPHA, dpi=300, bbox_inches='tight')
        print(f"Alpha trade-off plot saved to {OUTPUT_FIGURE_PATH_ALPHA}")
    else:
        plt.show()

    print("\nScript finished.")