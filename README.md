# README.md

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type Checking: mypy](https://img.shields.io/badge/type_checking-mypy-blue)](http://mypy-lang.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%23025596?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-150458.svg?style=flat&logo=python&logoColor=white)](https://www.statsmodels.org/stable/index.html)
[![Joblib](https://img.shields.io/badge/Joblib-013243.svg?style=flat&logo=python&logoColor=white)](https://joblib.readthedocs.io/en/latest/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.17599v1-b31b1b.svg)](https://arxiv.org/abs/2507.17599v1)
[![Research](https://img.shields.io/badge/Research-Empirical%20Asset%20Pricing-green)](https://github.com/chirindaopensource/randomized_test_alpha)
[![Discipline](https://img.shields.io/badge/Discipline-Econometrics-blue)](https://github.com/chirindaopensource/randomized_test_alpha)
[![Methodology](https://img.shields.io/badge/Methodology-Panel%20Data%20%7C%20Monte%20Carlo-orange)](https://github.com/chirindaopensource/randomized_test_alpha)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/randomized_test_alpha)

**Repository:** `https://github.com/chirindaopensource/randomized_test_alpha`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"A General Randomized Test for Alpha"** by:

*   Daniele Massacci
*   Lucio Sarno
*   Lorenzo Trapani
*   Pierluigi Vallarino

The project provides a complete, end-to-end pipeline for testing the joint null hypothesis of zero alpha in high-dimensional linear factor models. It replicates the paper's novel randomized testing procedure, which is robust to common violations of classical statistical assumptions such as non-Gaussianity, conditional heteroskedasticity, and strong cross-sectional dependence. The goal is to provide a transparent, robust, and computationally efficient "truth detector" for asset pricing models.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: run_full_study](#key-callable-run_full_study)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the methodologies presented in the 2025 paper "A General Randomized Test for Alpha." The core of this repository is the iPython Notebook `randomized_test_alpha_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings, from initial data validation and cleansing to the final generation of empirical results and Monte Carlo simulations.

The central question in empirical asset pricing is whether a proposed factor model can fully explain the cross-section of expected returns. A model is considered well-specified if the pricing errors, or "alphas," are jointly indistinguishable from zero. This project provides the tools to rigorously test this hypothesis in challenging, high-dimensional settings (where the number of assets `N` can be larger than the number of time periods `T`).

This codebase enables users to:
-   Rigorously validate, clean, and align large panels of asset and factor return data.
-   Apply the novel randomized alpha test to various factor model specifications in a rolling-window fashion.
-   Conduct comprehensive Monte Carlo simulations to verify the test's statistical properties (size and power) under various data generating processes.
-   Benchmark the test's performance against other methods from the literature.
-   Automatically generate publication-quality tables and visualizations to report the findings.

## Theoretical Background

The implemented methods are grounded in modern panel data econometrics and Extreme Value Theory.

**The Linear Factor Model:** The analysis begins with the standard time-series regression for a panel of `N` assets over `T` periods:
$$
y_{i,t} = \alpha_i + \beta'_i f_t + u_{i,t}
$$
The null hypothesis is that the model is correctly specified, meaning all pricing errors (`α_i`) are jointly zero:
$$
H_0: \max_{1 \le i \le N} |\alpha_i| = 0
$$

**Econometric Challenges:** Traditional tests like the GRS test fail in modern settings due to:
1.  **High Dimensionality:** When `N > T`, the `N x N` residual covariance matrix is singular and cannot be inverted.
2.  **Restrictive Assumptions:** Classical tests often assume normally distributed, homoskedastic, and serially uncorrelated errors, which are frequently violated by financial returns.

**The Randomized Test Methodology:** The paper's key innovation is a multi-step procedure that circumvents these issues:
1.  **OLS Estimation:** Estimate `α̂_i` and residuals `û_{i,t}` for each asset `i`.
2.  **Normalization:** Transform the alphas into a scale-free statistic `ψ_{i,NT}` that converges to zero under `H₀` but diverges under the alternative `Hₐ`.
    $$ \psi_{i,NT} = \left| \frac{T^{1/\nu} \hat{\alpha}_{i,T}}{\hat{s}_{NT}} \right|^{\nu/2} $$
3.  **Randomization:** Perturb the `ψ` statistics with i.i.d. standard normal shocks `ω_i` to create `z_{i,NT} = ψ_{i,NT} + ω_i`. Under `H₀`, the `z` statistics behave like a standard normal sample.
4.  **Test Statistic:** The final test statistic is the maximum of the perturbed series, `Z_{N,T} = max(z_{i,NT})`. By Extreme Value Theory, the distribution of `Z_{N,T}` under `H₀` is asymptotically Gumbel, providing a known distribution for inference without estimating any covariance matrices.
5.  **De-randomization:** To ensure a deterministic result, the randomization is repeated `B` times, and the final decision is based on the fraction of times `Z_{N,T}` falls below its asymptotic critical value.

## Features

The provided iPython Notebook (`randomized_test_alpha_draft.ipynb`) implements the full research pipeline, including:

-   **Data Pipeline:** A robust validation and cleansing module for preparing large panel datasets.
-   **Empirical Analysis Engine:** A high-level orchestrator that automates the rolling-window analysis for multiple user-defined factor models.
-   **Core Test Implementation:** A precise, numerically stable implementation of the complete randomized testing procedure, from OLS estimation to the final de-randomized decision.
-   **Monte Carlo Framework:** A powerful, parallelized simulation engine to evaluate the test's size and power under various DGPs (Gaussian, Student's t, GARCH errors; weak, semi-strong, and strong factors).
-   **Comparative Analysis:** A framework for benchmarking the test against other methods from the literature (e.g., FLY, AS tests).
-   **Automated Reporting:** Functions to automatically generate publication-quality summary tables and visualizations for both empirical and simulation results.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Data Preparation (Tasks 1-2):** The pipeline ingests raw asset and factor returns, validates their structure, handles missing values via constrained imputation, treats outliers with cross-sectional winsorization, and aligns the datasets.
2.  **Empirical Analysis (Tasks 3-9):** It sets up a rolling-window schedule and, for each window and each model, executes the full testing procedure:
    -   Vectorized OLS estimation (Task 4).
    -   `ψ` statistic computation (Task 5).
    -   Randomization and `Z` statistic computation (Task 6).
    -   De-randomization and `Q` statistic computation (Task 7).
    -   Application of the final decision rule (Task 8).
3.  **Monte Carlo Analysis (Tasks 13-17):** It sets up a grid of simulation experiments, generates data from complex DGPs (Task 14), and runs the test `M` times to compute empirical size and power (Task 15), managed by a high-level orchestrator (Task 16).
4.  **Reporting (Tasks 11, 12, 18, 19):** It synthesizes all quantitative outputs into summary tables, power curve plots, and a structured textual interpretation of the findings.

## Core Components (Notebook Structure)

The `randomized_test_alpha_draft.ipynb` notebook is structured as a logical pipeline with modular functions for each task:

-   **Tasks 1-3:** Data Validation, Cleansing, and Empirical Setup.
-   **Tasks 4-8:** The core testing pipeline for a single window (Estimation, Statistic Computation, Randomization, De-randomization, Decision).
-   **Task 9:** `EmpiricalAnalysisOrchestrator` class to run the empirical study.
-   **Tasks 10-12:** Empirical Robustness, Compilation, and Visualization.
-   **Tasks 13-16:** Monte Carlo Setup, DGP, Simulation Engine, and Orchestrator.
-   **Tasks 17-19:** Monte Carlo Robustness, Compilation, and Interpretation.
-   **Task 20:** `run_full_study` master function to execute the entire project.

## Key Callable: run_full_study

The central function in this project is `run_full_study`. It orchestrates the entire analytical workflow from raw data to a final, comprehensive report object.

```python
def run_full_study(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    replication_config: Dict[str, Any],
    run_empirical: bool = True,
    run_monte_carlo: bool = True,
    n_jobs: int = -1,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes the complete end-to-end research pipeline for the randomized alpha test.
    """
    # ... (implementation is in the notebook)
```

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`, `joblib`, `tqdm`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/randomized_test_alpha.git
    cd randomized_test_alpha
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy scipy statsmodels matplotlib joblib tqdm
    ```

## Input Data Structure

The pipeline requires three inputs passed to the `run_full_study` function:

1.  **`asset_returns`**: A `pandas.DataFrame` where the index is a monthly `DatetimeIndex` and columns are individual asset returns.
2.  **`factor_returns`**: A `pandas.DataFrame` with the same monthly `DatetimeIndex` and columns for the factor returns (e.g., 'Mkt-RF', 'SMB', etc.).
3.  **`replication_config`**: A nested Python dictionary that controls every aspect of the empirical and Monte Carlo analyses. A fully specified example is provided in the notebook.

## Usage

The `randomized_test_alpha_draft.ipynb` notebook provides a complete, step-by-step guide. The core workflow is:

1.  **Prepare Inputs:** Load your asset and factor returns into DataFrames and define the `replication_config` dictionary.
2.  **Execute Pipeline:** Call the master orchestrator function:
    ```python
    full_study_results = run_full_study(
        asset_returns=my_asset_returns_df,
        factor_returns=my_factor_returns_df,
        replication_config=my_config
    )
    ```
3.  **Inspect Outputs:** Programmatically access any result from the returned `full_study_results` dictionary. For example, to view the main empirical summary table:
    ```python
    # The output is a pandas Styler object, access the data with .data
    empirical_table = full_study_results['empirical_summary_table'].data
    print(empirical_table)
    ```

## Output Structure

The `run_full_study` function returns a single, comprehensive dictionary with the following top-level keys:

-   `empirical_timeseries_results`: A `pd.DataFrame` with the detailed, window-by-window results of the empirical analysis.
-   `empirical_summary_table`: A styled `pd.DataFrame` summarizing the empirical rejection rates.
-   `empirical_q_statistic_plot`: A tuple containing the `matplotlib` Figure and Axes objects for the main empirical plot.
-   `monte_carlo_raw_results`: A tidy `pd.DataFrame` with the results from every Monte Carlo experimental cell.
-   `monte_carlo_summary_tables`: A dictionary of styled `pd.DataFrame`s, one for each simulation scenario.
-   `final_quantitative_summary`: A nested dictionary containing a quantitative interpretation of the key findings.

## Project Structure

```
randomized_test_alpha/
│
├── randomized_test_alpha_draft.ipynb  # Main implementation notebook
├── requirements.txt                     # Python package dependencies
├── LICENSE                              # MIT license file
└── README.md                            # This documentation file
```

## Customization

The pipeline is highly customizable via the master `replication_config` dictionary. Users can easily modify:
-   The `model_specifications` to test different factor models.
-   The `rolling_window_size_months` and date ranges for the empirical study.
-   The `N_grid`, `T_grid`, `scenarios`, and all DGP parameters for the Monte Carlo simulations.
-   The parameters of the test itself, such as `nu` and `tau`.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{massacci2025general,
  title={A general randomized test for Alpha},
  author={Massacci, Daniele and Sarno, Lucio and Trapani, Lorenzo and Vallarino, Pierluigi},
  journal={arXiv preprint arXiv:2507.17599v1},
  year={2025}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). A Python Implementation of "A general randomized test for Alpha". 
GitHub repository: https://github.com/chirindaopensource/randomized_test_alpha
```

## Acknowledgments

-   Credit to Daniele Massacci, Lucio Sarno, Lorenzo Trapani, and Pierluigi Vallarino for their innovative and rigorous research.
-   Thanks to the developers of the scientific Python ecosystem (`numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `joblib`) that makes this work possible.

--

*This README was generated based on the structure and content of `randomized_test_alpha_draft.ipynb` and follows best practices for research software documentation.*
