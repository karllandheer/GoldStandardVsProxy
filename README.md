# The Consequences of Statistical Tests on Using Proxy Measurements in Place of Gold Standard Measurements: An Application Magnetic Resonance Spectroscopy

## Overview

This repository contains the simulation code accompanying the research paper titled "The Consequences of Statistical Tests on Using Proxy Measurements in Place of Gold Standard Measurements: An Application Magnetic Resonance Spectroscopy" by Treacy, Juchem, and Landheer.

The paper investigates how the use of proxy measurements, which are often employed when gold-standard methods are infeasible or unavailable (e.g., short-TE MRS for GABA quantification), can significantly impact the reliability of statistical test outcomes. Through statistical simulations, we demonstrate that even moderately high correlations between proxy and gold-standard measurements can lead to drastically inflated false positive (FPR) and false negative (FNR) rates, highlighting a critical issue for reproducibility in biomedical science.

## Paper Abstract

The use of proxy measurements in biomedical science is ubiquitous, often due to the infeasibility or unavailability of gold-standard (i.e., most precise and/or accurate) methods. For example, in magnetic resonance spectroscopy (MRS), proxy short-TE sequences are frequently employed to estimate difficult-to-measure metabolites such as GABA, despite J-difference editing (e.g., MEGA-edited MRS) being the recommended gold-standard for improved specificity and sensitivity. This work investigates the critical relationship between the correlation of proxy and gold-standard measurements and the associated false positive (FPR) and false negative (FNR) rates of statistical tests performed on proxy measurements. Through statistical simulations, we demonstrate that even moderately high correlations (0.6-0.7), reported in the literature for short-TE vs. MEGA-edited GABA, can lead to drastically inflated FPRs and FNRs. We show that these rates are highly sensitive to the magnitude of any introduced bias in the proxy measurement (δ) and the underlying true effect size (Δ). For instance, a small, unmeasured bias in short-TE GABA, potentially arising from macromolecule contamination, can substantially inflate FPRs. Conversely, imperfect correlation can significantly reduce statistical power, leading to high FNRs, which may explain some discrepancies within the literature. Although this work focuses specifically on the relationship between short-TE and MEGA-edited GABA, the arguments presented here apply more broadly to other difficult to measure metabolites (e.g., glutathione, 2-hydroxyglutarate), or generally to any circumstance where the proxy estimate is readily obtained and analyzed in place of the gold-standard measurement. 



## Getting Started

### Prerequisites

To run the simulations, you will need Python installed. We recommend using `conda` or `pip` for managing dependencies.

* Python 3.8 or higher *

### Installation

Clone the repository:
```bash
git clone [https://github.com/karllandheer/GoldStandardVsProxy)]
cd GoldStandardVsProxy

### Running

You can then run each of the individual scripts, which replicate the analyses in Figures 2-5. Figure 1 was generated via synMARSS, which is licensed by Columbia Tech Ventures. Read about synMARSS here, and how to access it: https://pubmed.ncbi.nlm.nih.gov/39948757/

