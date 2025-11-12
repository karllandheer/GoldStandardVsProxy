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
N_SIMULATIONS = 500_000 # Using 500k for final accuracy
ALPHA = 0.05

# --- 2. Literature Values ---
# Values from Jia et al. manuscript
N_SAMPLES = 25 # n=25 for both groups
FIXED_RHO = 0.72 # Empirical 7T value
SIGMA_P_LIT = 1.550032652662542 

# OBSERVED mean difference from the literature.
DELTA_P_LIT = (7.58575834012 - 6.75780033084)

# CALCULATED: The bias 'delta' required to explain DELTA_P_LIT
# IF Delta=0, R_sigma=1.0, and VR=1.0.
# This is just a reference line for the plot.
DELTA_CONSTANT_PART = DELTA_P_LIT / np.sqrt(1 - FIXED_RHO**2)
R_SIGMA_BASELINE = 1.0 # This is the baseline R_sigma for the reference delta
FIXED_DELTA_TEST = DELTA_CONSTANT_PART / R_SIGMA_BASELINE
print(f"Reference delta (for R_sigma=1.0) calculated as: {FIXED_DELTA_TEST:.3f}")

# --- 3. Grid Parameters ---
# MODIFIED: Calculate VR_MAX based on the constraint rho <= 1 / sqrt(VR)
# This is the theoretical maximum VR allowed for rho=0.72
VR_MAX = (1 / FIXED_RHO)**2 
VR_VALUES = [0.5, 1.0, VR_MAX] # Using VR_MAX instead of 2.0
R_SIGMA_VALUES = [0.5, 1.0, 2]
DELTA_SWEEP = np.linspace(0, 3, 71)

print(f"Running with VR_VALUES: [0.5, 1.0, {VR_MAX:.3f} (VR_max)]")

# --- 4. Output Paths ---
OUTPUT_PATH = r'C:\ColumbiaWork\PapersOnGo\MEGAvsNon\MEGAvsNon\Figures'
OUTPUT_DATA_PATH = "simulation_results_literature_FPR_VR_Rsigma_sweep.csv"
OUTPUT_FIGURE_PATH_FPR = "simulation_FPR_literature_VR_Rsigma_sweep.png"

# Check if path exists, otherwise save to local directory
if OUTPUT_PATH is not None and os.path.exists(OUTPUT_PATH):
    OUTPUT_DATA_PATH = os.path.join(OUTPUT_PATH, OUTPUT_DATA_PATH)
    OUTPUT_FIGURE_PATH_FPR = os.path.join(OUTPUT_PATH, OUTPUT_FIGURE_PATH_FPR)
else:
    print(f"Warning: OUTPUT_PATH '{OUTPUT_PATH}' not found. Saving to current directory.")
    OUTPUT_DATA_PATH = "simulation_results_literature_FPR_VR_Rsigma_sweep.csv"
    OUTPUT_FIGURE_PATH_FPR = "simulation_FPR_literature_VR_Rsigma_sweep.png"


def run_literature_fpr_sim(vr, r_sigma, delta): 
    """
    Runs a vectorized FPR simulation for a single (vr, r_sigma, delta) point.
    Implements the "Experimental Model" (Path B), where sigma_bio is derived
    from the observable sigma_P.
    
    *** CORRECTED VERSION ***
    """
    
    # Use a unique seed for each parameter combination
    np.random.seed(int((vr * 1000) + (r_sigma * 100) + (delta * 10000)))
    
    # --- A. Calculate Model Coefficients ---
    rho = FIXED_RHO
    
    # We start with the observable SIGMA_P_LIT and derive sigma_bio.
    # sigma_bio is now a VARIABLE that depends on vr.
    # (From sigma_p_lit^2 = sigma_bio^2 * VR)
    sigma_bio = SIGMA_P_LIT / np.sqrt(vr) 
    
    # sigma_contam is also variable, as it depends on sigma_bio
    # Note: R_sigma is ratio of VARIANCES
    sigma_contam = np.sqrt(r_sigma * (sigma_bio**2)) 
    
    x1 = rho * np.sqrt(vr)
    
    # Handle potential floating point inaccuracies (arg should not be < 0)
    x2_squared_arg = vr * (1 - rho**2) / r_sigma
    x2 = 0.0 if x2_squared_arg <= 0 else np.sqrt(x2_squared_arg)

    # --- B. Run False Positive Rate (FPR) Simulation ---
    
    # S_A for controls
    S_A_fpr = np.random.normal(loc=0.0, scale=sigma_bio, 
                                 size=(N_SAMPLES, N_SIMULATIONS))
    
    # S_B for cases. For FPR, Delta=0, so it's from the same distribution.
    # This is an INDEPENDENT draw from S_A_fpr.
    S_B_fpr = np.random.normal(loc=0.0, scale=sigma_bio, 
                                 size=(N_SAMPLES, N_SIMULATIONS)) 

    # loc=0.0 because common bias is irrelevant to t-stat
    zeta_A = np.random.normal(loc=0.0, scale=sigma_contam, 
                                 size=(N_SAMPLES, N_SIMULATIONS))
    
    # loc=delta, which is the differential bias we are sweeping
    zeta_B_fpr = np.random.normal(loc=0.0 + delta, scale=sigma_contam, 
                                    size=(N_SAMPLES, N_SIMULATIONS))

    P_A_fpr = x1 * S_A_fpr + x2 * zeta_A
    
    # --- MAJOR FIX: Use the independent S_B_fpr signal ---
    P_B_fpr = x1 * S_B_fpr + x2 * zeta_B_fpr
    
    # Run Welch's t-test (independent samples)
    _, p_values_fpr = ttest_ind(P_A_fpr, P_B_fpr, axis=0, equal_var=False)

    fpr = np.sum(p_values_fpr < ALPHA) / N_SIMULATIONS

    return (vr, r_sigma, delta, fpr)

# --- 5. Main Execution ---
if __name__ == "__main__":

    print("Starting Literature FPR Simulations (sweeping VR and R_sigma)...")
    print(f"CPUs available: {os.cpu_count()}")
    start_time = time.time()

    # --- A. Define the Grid Parameters ---
    tasks = list(itertools.product(VR_VALUES, R_SIGMA_VALUES, DELTA_SWEEP))
    print(f"Total simulation tasks to run: {len(tasks)}")

    # --- B. Run Simulations in Parallel ---
    # n_jobs=-2 leaves one core free
    results = Parallel(n_jobs=-2, verbose=10)(
        delayed(run_literature_fpr_sim)(vr, r_sigma, delta) for vr, r_sigma, delta in tasks
    )

    # --- C. Process and Save Results ---
    df = pd.DataFrame(results, columns=['VR', 'R_sigma', 'delta', 'FPR'])
    df = df.sort_values(by=['VR', 'R_sigma', 'delta']) 
    df.to_csv(OUTPUT_DATA_PATH, index=False)
    
    end_time = time.time()
    print("\n--- All Simulations Complete ---")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to {OUTPUT_DATA_PATH}")

    # --- D. Create and Save Plot ---
    print("Generating plot...")
    
    # Create a 1x3 grid of subplots, sharing the Y-axis
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    
    # --- Print Final Analysis Header ---
    print(f"\n--- Analysis of FPR for FIXED BIAS (δ={FIXED_DELTA_TEST:.3f}) ---")
    print("This shows the vulnerability of the study based on assumed noise properties.")

    # Iterate over each VR value and its corresponding subplot axis
    for ax, vr in zip(axes, VR_VALUES):
        legend_labels = []
        analysis_results = [] # Reset for each VR subplot
        
        # Get the data subset for this specific VR
        vr_data_subset = df[df['VR'] == vr]
        
        # Inner loop: plot one line for each R_sigma
        for r_sigma in R_SIGMA_VALUES:
            data_subset = vr_data_subset[vr_data_subset['R_sigma'] == r_sigma]
            
            # Plot the FPR curve for this R_sigma on the current ax
            ax.plot(data_subset['delta'], data_subset['FPR'])
            
            fpr_list = data_subset['FPR'].values
            delta_list_sim = data_subset['delta'].values
            
            # Find the FPR at the FIXED_DELTA_TEST
            closest_index = np.argmin(np.abs(delta_list_sim - FIXED_DELTA_TEST))
            FPR_at_fixed_delta = fpr_list[closest_index]
            
            analysis_results.append({
                'R_sigma': r_sigma,
                'FPR': FPR_at_fixed_delta
            })
            
            legend_labels.append(f'$R_\\sigma = {r_sigma}$')

        # --- Format this specific subplot (ax) ---
        
        if vr == VR_MAX:
            ax.set_title(f'VR = {vr:.3f} ($VR_{{max}}$)')
        else:
            ax.set_title(f'VR = {vr}')
            
        ax.legend(legend_labels, fancybox=True, title="Noise Ratio ($R_\\sigma$)")
        ax.tick_params(which='both', top=True, right=True, labeltop=False, labelright=False)
        ax.set_ylim([0, 1.02])
        ax.set_xlabel('Differntial bias, δ (unitless)')
        ax.grid(True, linestyle=':', alpha=0.7)

        ax.axvline(x=FIXED_DELTA_TEST, color='k', linestyle='--', linewidth=1.5, 
                   label=f"δ={FIXED_DELTA_TEST:.3f} (Rσ=1.0 Baseline)")
        
        # --- Print Analysis for this VR ---
        print(f"\n--- Analysis for VR = {vr:.3f} ---")
        for res in analysis_results:
            print(f"  R_sigma = {res['R_sigma']:.1f}: FPR = {res['FPR']:.3f}")

    # --- Format common Y-axis label ---
    axes[0].set_ylabel('False positive rate (unitless)')
    
    # Adjust layout
    plt.tight_layout()

    # --- Save the combined figure ---
    if OUTPUT_FIGURE_PATH_FPR:
        plt.savefig(OUTPUT_FIGURE_PATH_FPR, dpi=300, bbox_inches='tight')
        print(f"\nFPR plot saved to {OUTPUT_FIGURE_PATH_FPR}")
    else:
        # Fallback to show the plot if saving fails or path is None
        print("\nDisplaying plot...")
        plt.show()
    
    print("\nScript finished.")