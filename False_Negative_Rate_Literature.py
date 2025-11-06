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
N_SIMULATIONS = 500 # Increased to ensure accurate FNR curves
ALPHA = 0.05

# Literature values from your script
N_SAMPLES_CONTROL = 10
N_SAMPLES_CASE = 10
DELTA_MANUSCRIPT = 0.312 * (0.8352941176470589 - 1) # True effect from gold-standard paper

# --- 2. Grid Parameters ---
VR_VALUES = [0.5, 1.0, 2.0]
FIXED_R_SIGMA = 1.0 # R_sigma is fixed for the FNR plot, doesnt matter in teh end

SIGMA_P_LIT = 0.06

# Sweep parameters
CORR_VALUES = [0.58, 1.0 - 1e-9] # Empirical 3T value and "perfect" correlation
DELTA_SWEEP = np.linspace(-0.2, 0, 71) # The range of Delta values to test

# --- 3. Output Paths ---
OUTPUT_PATH = r'C:\ColumbiaWork\PapersOnGo\MEGAvsNon\MEGAvsNon\Figures'
OUTPUT_DATA_PATH = "simulation_results_literature_FNR.csv"
OUTPUT_FIGURE_PATH_FNR = "simulation_FNR_literature.png"

# Check if path exists, otherwise save to local directory
if OUTPUT_PATH is not None and os.path.exists(OUTPUT_PATH):
    OUTPUT_DATA_PATH = os.path.join(OUTPUT_PATH, OUTPUT_DATA_PATH)
    OUTPUT_FIGURE_PATH_FNR = os.path.join(OUTPUT_PATH, OUTPUT_FIGURE_PATH_FNR)
else:
    print(f"Warning: OUTPUT_PATH '{OUTPUT_PATH}' not found. Saving to current directory.")
    OUTPUT_DATA_PATH = "simulation_results_literature_FNR.csv"
    OUTPUT_FIGURE_PATH_FNR = "simulation_FNR_literature.png"


def run_literature_sim(vr, corr, Delta):
    """
    Runs a vectorized FNR simulation for a single (vr, corr, Delta) point.
    R_sigma is implicitly FIXED at 1.0.
    Implements "Experimental Model" (Path B) where sigma_bio is derived.
    """
    
    np.random.seed(abs(int((vr * 1000) + (corr * 100) + (Delta * 10000))))
    
    
    # --- A. Calculate Model Coefficients (R_sigma=1.0) ---
    r_sigma = FIXED_R_SIGMA # 1.0
    
    # We start with the observable SIGMA_P_LIT and derive sigma_bio.
    # sigma_bio is now a VARIABLE that depends on vr.
    sigma_bio = SIGMA_P_LIT / np.sqrt(vr)
    
    # sigma_contam is also variable, as it depends on sigma_bio
    sigma_contam = sigma_bio * np.sqrt(r_sigma)
    
    # x1 = rho * sqrt(VR)
    x1 = corr * np.sqrt(vr) 
    
    # x2 = sqrt(VR * (1 - corr^2) / R_sigma) = sqrt(VR * (1 - corr^2))
    x2_squared_arg = vr * (1 - corr**2) / r_sigma
    x2 = 0.0 if x2_squared_arg < 0 else np.sqrt(x2_squared_arg)

    # --- B. Run False Negative Rate (FNR) Simulation ---
    S_A_fnr = np.random.normal(loc=0, scale=sigma_bio, 
                                 size=(N_SAMPLES_CONTROL, N_SIMULATIONS))
    S_B_fnr = np.random.normal(loc=0 + Delta, scale=sigma_bio, 
                                 size=(N_SAMPLES_CASE, N_SIMULATIONS))
                                 
    # Generate independent contamination signals for both groups
    zeta_A_fnr = np.random.normal(loc=0.0, scale=sigma_contam, 
                                    size=(N_SAMPLES_CONTROL, N_SIMULATIONS))
    zeta_B_fnr = np.random.normal(loc=0.0, scale=sigma_contam, 
                                    size=(N_SAMPLES_CASE, N_SIMULATIONS))

    # Construct the final proxy measurements
    P_A_fnr = x1 * S_A_fnr + x2 * zeta_A_fnr
    P_B_fnr = x1 * S_B_fnr + x2 * zeta_B_fnr
    
    # Run Welch's t-test
    _, p_values_fnr = ttest_ind(P_A_fnr, P_B_fnr, axis=0, equal_var=False)

    # Calculate FNR: The fraction of p-values >= alpha
    fnr = np.sum(p_values_fnr >= ALPHA) / N_SIMULATIONS

    return (vr, corr, Delta, fnr) 

# --- 4. Main Execution ---
if __name__ == "__main__":

    print("Starting Literature FNR Simulations...")
    print(f"CPUs available: {os.cpu_count()}")
    start_time = time.time()

    # --- A. Define the Grid Parameters ---
    tasks = list(itertools.product(VR_VALUES, CORR_VALUES, DELTA_SWEEP))
    print(f"Total simulation tasks to run: {len(tasks)}")

    # --- B. Run Simulations in Parallel ---
    results = Parallel(n_jobs=-2, verbose=10)(
        delayed(run_literature_sim)(vr, corr, Delta) for vr, corr, Delta in tasks
    )

    # --- C. Process and Save Results ---
    df = pd.DataFrame(results, columns=['VR', 'rho', 'Delta', 'FNR'])
    df = df.sort_values(by=['VR', 'rho', 'Delta'])
    df.to_csv(OUTPUT_DATA_PATH, index=False)

    end_time = time.time()
    print("\n--- All Simulations Complete ---")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to {OUTPUT_DATA_PATH}")

    # --- D. Create and Save Plot ---
    print("Generating plot...")
    
    # Create a 1x3 grid of subplots, sharing the Y-axis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    # Iterate over each VR value and its corresponding subplot axis
    for ax, vr in zip(axes, VR_VALUES):
        legend_labels = []
        
        print(f"\n--- Analysis for VR = {vr} ---")
        
        # Get the data subset for this specific VR
        vr_data_subset = df[df['VR'] == vr]

        for corr in CORR_VALUES:
            data_subset = vr_data_subset[vr_data_subset['rho'] == corr]
            
            # Plot the FNR curve on the current ax
            ax.plot(data_subset['Delta'], data_subset['FNR'])

            
            # 1. Find FNR at the "true" literature Delta
            closest_index = np.argmin(np.abs(data_subset['Delta'] - DELTA_MANUSCRIPT))
            fnr_at_true_delta = data_subset.iloc[closest_index]['FNR']
            print(f'  For corr = {corr:.2f}, at the literature Δ of {DELTA_MANUSCRIPT:.3f}, the FNR is {fnr_at_true_delta:.3f}')
            
            # 2. Find Delta required for FNR < 0.05
            fnr_cd_list = data_subset['FNR'].values
            delta_list = data_subset['Delta'].values
            
            # Find the last index where FNR is < 0.05
            indices_below_0p05 = np.where(fnr_cd_list < 0.05)[0]
            if len(indices_below_0p05) > 0:
                index_fnr_below_0p05 = indices_below_0p05[-1]
                print(f'  For corr = {corr:.2f}, the smallest Δ to get FNR < 0.05 is {delta_list[index_fnr_below_0p05]:.3f}')
            else:
                print(f'  For corr = {corr:.2f}, FNR never dropped below 0.05 in the tested range.')

            # Prepare legend label
            legend_labels.append(f'ρ = {corr:.2f}')

        # --- Format this specific subplot (ax) ---
        ax.set_title(f'VR = {vr}') 
        ax.legend(legend_labels, fancybox=True)
        ax.tick_params(which='both', top=True, right=True, labeltop=False, labelright=False)
        ax.set_xlabel('Difference between means of gold-standard measurements, Δ (unitless)')
        ax.grid(True, linestyle=':', alpha=0.7)
        
        # Add marker for the nominal delta we are testing
        ax.axvline(x=DELTA_MANUSCRIPT, color='k', linestyle='--', linewidth=1.5, 
                   label=f"Nominal Δ={DELTA_MANUSCRIPT:.3f}")

    # --- Format common Y-axis label ---
    axes[0].set_ylabel('False negative rate (unitless)')
    
    # Adjust layout
    plt.tight_layout()

    # --- Save the combined figure ---
    if OUTPUT_FIGURE_PATH_FNR:
        plt.savefig(OUTPUT_FIGURE_PATH_FNR, dpi=300, bbox_inches='tight')
        print(f"\nFNR plot saved to {OUTPUT_FIGURE_PATH_FNR}")
    else:
        plt.show()

    print("\nScript finished.")