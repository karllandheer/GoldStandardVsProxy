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
N_SIMULATIONS = 500_000 #500k for final accuracy
ALPHA = 0.05
DELTA_MANUSCRIPT = -0.05 # Assumed true effect size from manuscript
N_SAMPLES_CONTROL = 10
N_SAMPLES_CASE = 9
SIGMA_P_LIT = 0.06  # From Bilcher et al. [12]

# --- 2. Grid Parameters ---
EMPIRICAL_CORR = 0.58
# VR_MAX = (1 / EMPIRICAL_CORR)**2 # This was the incorrect calculation
VR_MAX = 1.3130643285972257 # This is the corrected value
VR_VALUES = [0.5, 1.0, VR_MAX]

FIXED_R_SIGMA = 1.0 # R_sigma is fixed, as it doesn't affect FNR when delta=0

# Sweep parameters
CORR_VALUES = [EMPIRICAL_CORR, 1.0 - 1e-9] # Empirical 3T value and "perfect"
DELTA_SWEEP = np.linspace(-0.2, 0, 71) # The range of Delta values to test

print(f"Manuscript Reference Delta (Foerster et al.): {DELTA_MANUSCRIPT:.3f}")
print(f"Running with VR_VALUES: [0.5, 1.0, {VR_MAX:.3f} (VR_max)]")


# --- 3. Output Paths ---
OUTPUT_PATH = r'C:\ColumbiaWork\PapersOnGo\MEGAvsNon\MEGAvsNon\Figures'
# New filenames to reflect the change
OUTPUT_DATA_PATH = "simulation_results_literature_FNR_VRmax_1.313.csv"
OUTPUT_FIGURE_PATH_FNR = "simulation_FNR_literature_VRmax_1.313.png"

# Check if path exists, otherwise save to local directory
if OUTPUT_PATH is not None and os.path.exists(OUTPUT_PATH):
    OUTPUT_DATA_PATH = os.path.join(OUTPUT_PATH, OUTPUT_DATA_PATH)
    OUTPUT_FIGURE_PATH_FNR = os.path.join(OUTPUT_PATH, OUTPUT_FIGURE_PATH_FNR)
else:
    print(f"Warning: OUTPUT_PATH '{OUTPUT_PATH}' not found. Saving to current directory.")
    OUTPUT_DATA_PATH = "simulation_results_literature_FNR_VRmax_1.313.csv"
    OUTPUT_FIGURE_PATH_FNR = "simulation_FNR_literature_VRmax_1.313.png"


def run_literature_sim(vr, corr, Delta):
    """
    Runs a vectorized FNR simulation for a single (vr, corr, Delta) point.
    *** INCLUDES MATHEMATICAL VALIDITY CHECK ***
    """

    # --- Mathematical Validity Check ---
    # Check if rho <= 1/sqrt(VR). Add 1e-9 for float precision.
    if corr > (1 / np.sqrt(vr)) + 1e-9:
        # This combination is physically impossible under the model's constraints.
        return (vr, corr, Delta, np.nan)
    # --- End of Check ---

    np.random.seed(abs(int((vr * 1000) + (corr * 100) + (Delta * 10000))))
    
    # --- A. Calculate Model Coefficients (R_sigma=1.0) ---
    r_sigma = FIXED_R_SIGMA # 1.0
    sigma_bio = SIGMA_P_LIT / np.sqrt(vr)
    sigma_contam = sigma_bio * np.sqrt(r_sigma)
    x1 = corr * np.sqrt(vr) 
    x2_squared_arg = vr * (1 - corr**2) / r_sigma
    x2 = 0.0 if x2_squared_arg <= 0 else np.sqrt(x2_squared_arg)

    # --- B. Run False Negative Rate (FNR) Simulation ---
    S_A_fnr = np.random.normal(loc=0, scale=sigma_bio, 
                                 size=(N_SAMPLES_CONTROL, N_SIMULATIONS))
    S_B_fnr = np.random.normal(loc=0 + Delta, scale=sigma_bio, 
                                 size=(N_SAMPLES_CASE, N_SIMULATIONS))
                                 
    zeta_A_fnr = np.random.normal(loc=0.0, scale=sigma_contam, 
                                     size=(N_SAMPLES_CONTROL, N_SIMULATIONS))
    zeta_B_fnr = np.random.normal(loc=0.0, scale=sigma_contam, 
                                     size=(N_SAMPLES_CASE, N_SIMULATIONS))

    P_A_fnr = x1 * S_A_fnr + x2 * zeta_A_fnr
    P_B_fnr = x1 * S_B_fnr + x2 * zeta_B_fnr
    
    _, p_values_fnr = ttest_ind(P_A_fnr, P_B_fnr, axis=0, equal_var=False)
    fnr = np.sum(p_values_fnr >= ALPHA) / N_SIMULATIONS

    return (vr, corr, Delta, fnr) 

# --- 4. Main Execution ---
if __name__ == "__main__":

    print("Starting Literature FNR Simulations (Corrected)...")
    print(f"CPUs available: {os.cpu_count()}")
    start_time = time.time()

    tasks = list(itertools.product(VR_VALUES, CORR_VALUES, DELTA_SWEEP))
    print(f"Total simulation tasks to run: {len(tasks)}")
    print(f"Note: Invalid (rho, VR) combinations will be skipped.")

    results = Parallel(n_jobs=-2, verbose=10)(
        delayed(run_literature_sim)(vr, corr, Delta) for vr, corr, Delta in tasks
    )

    df = pd.DataFrame(results, columns=['VR', 'rho', 'Delta', 'FNR'])
    
    initial_rows = len(df)
    df = df.dropna()
    final_rows = len(df)
    print(f"\nDropped {initial_rows - final_rows} invalid (rho, VR) combinations.")
    
    df = df.sort_values(by=['VR', 'rho', 'Delta'])
    df.to_csv(OUTPUT_DATA_PATH, index=False)

    end_time = time.time()
    print("\n--- All Simulations Complete ---")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to {OUTPUT_DATA_PATH}")

    # --- D. Create and Save Plot ---
    print("Generating plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    
    for ax, vr in zip(axes, VR_VALUES):
        legend_labels = []
        
        print(f"\n--- Analysis for VR = {vr:.3f} ---")
        
        vr_data_subset = df[df['VR'] == vr]
        valid_corrs_for_this_vr = vr_data_subset['rho'].unique()
        
        for corr in CORR_VALUES:
            if corr not in valid_corrs_for_this_vr:
                print(f"  Skipping corr = {corr:.2f} (Invalid for this VR)")
                continue

            data_subset = vr_data_subset[vr_data_subset['rho'] == corr]
            
            ax.plot(data_subset['Delta'], data_subset['FNR'])
            
            closest_index = np.argmin(np.abs(data_subset['Delta'] - DELTA_MANUSCRIPT))
            fnr_at_true_delta = data_subset.iloc[closest_index]['FNR']
            print(f'  For corr = {corr:.2f}, at Δ={DELTA_MANUSCRIPT:.3f}, FNR is {fnr_at_true_delta:.3f}')
            
            fnr_cd_list = data_subset['FNR'].values
            delta_list = data_subset['Delta'].values
            indices_below_0p05 = np.where(fnr_cd_list < 0.05)[0]
            if len(indices_below_0p05) > 0:
                index_fnr_below_0p05 = indices_below_0p05[-1]
                print(f'  For corr = {corr:.2f}, smallest Δ for FNR<0.05 is {delta_list[index_fnr_below_0p05]:.3f}')
            else:
                print(f'  For corr = {corr:.2f}, FNR never dropped below 0.05.')

            legend_labels.append(f'ρ = {corr:.2f}')

        if vr == VR_MAX:
            ax.set_title(f'VR = {vr:.1f} ($VR_{{max}}$)')
        else:
            ax.set_title(f'VR = {vr}')
            
        ax.legend(legend_labels, fancybox=True)
        ax.tick_params(which='both', top=True, right=True, labeltop=False, labelright=False)
        ax.set_xlabel('True effect size, Δ (unitless)')
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_ylim([0, 1.02])
        
        ax.axvline(x=DELTA_MANUSCRIPT, color='k', linestyle='--', linewidth=1.5, 
                   label=f"Nominal Δ={DELTA_MANUSCRIPT:.3f}")

    axes[0].set_ylabel('False negative rate (unitless)')
    plt.tight_layout()

    if OUTPUT_FIGURE_PATH_FNR:
        plt.savefig(OUTPUT_FIGURE_PATH_FNR, dpi=300, bbox_inches='tight')
        print(f"\nFNR plot saved to {OUTPUT_FIGURE_PATH_FNR}")
    else:
        plt.show()

    print("\nScript finished.")