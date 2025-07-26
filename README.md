# The Consequences of Statistical Tests on Using Proxy Measurements in Place of Gold Standard Measurements: An Application Magnetic Resonance Spectroscopy

## Overview

This repository contains the simulation code accompanying the research paper titled "The Consequences of Statistical Tests on Using Proxy Measurements in Place of Gold Standard Measurements: An Application Magnetic Resonance Spectroscopy" by Treacy, Juchem, and Landheer.

The paper investigates how the use of proxy measurements, which are often employed when gold-standard methods are infeasible or unavailable (e.g., short-TE MRS for GABA quantification), can significantly impact the reliability of statistical test outcomes. Through statistical simulations, we demonstrate that even moderately high correlations between proxy and gold-standard measurements can lead to drastically inflated false positive (FPR) and false negative (FNR) rates, highlighting a critical issue for reproducibility in biomedical science.

## Paper Abstract

*Paste your full abstract here, including the authors and affiliations.*

## Key Findings (Summary of what the code demonstrates)

* **Inflated False Positive Rates (FPR):** Demonstrates how even small, unmeasured biases in proxy measurements (e.g., from macromolecule contamination in short-TE MRS) can lead to a substantial increase in FPRs, far exceeding nominal levels.
* **Increased False Negative Rates (FNR) / Reduced Statistical Power:** Shows that imperfect correlation between proxy and gold-standard measurements can significantly reduce statistical power, resulting in high FNRs, which may explain discrepancies in the literature.
* **Sensitivity to Bias and True Effect Size:** Illustrates the high sensitivity of FPR and FNR to the magnitude of introduced bias ($\delta$) and the underlying true effect size ($\Delta$).

## Repository Contents

* `src/`: (Or `code/`, `scripts/`) Directory containing the Python scripts for the statistical simulations.
    * `simulation_main.py`: The primary script to run the simulations.
    * `analysis_plots.py`: (Optional, if you separate plotting) Script for generating the figures.
    * `utils.py`: (Optional) Helper functions or classes used by the main scripts.
* `data/`: (If applicable) Directory for any input data (e.g., example parameters, if not hardcoded). For your current setup, this might not be strictly necessary if all parameters are defined within the scripts.
* `results/`: (Optional) Directory where generated figures or raw simulation output data will be saved.
* `README.md`: This file.
* `LICENSE`: (Highly recommended) A file specifying the license under which your code is released.

## Getting Started

### Prerequisites

To run the simulations, you will need Python installed. We recommend using `conda` or `pip` for managing dependencies.

* Python 3.x

### Installation

Clone the repository:
```bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
