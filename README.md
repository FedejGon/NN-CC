# NN-CC
This repository contains the codes for a framework for equation discovery by combining Neural Networks with Characteristic Curves (NN-CC), Symmetry Constraints, and Post-Symbolic Regression (Post-SR).
# Symmetry-guided NN-CC and Post-Symbolic Regression

This repository contains the source code and PyTorch implementations associated with the manuscript:

**"Symmetry-guided neural network system identification using interpretable characteristic curves and post-symbolic regression"**

## Overview

This work introduces a modular framework for identifying nonlinear dynamical systems by decomposing dynamics into interpretable **Characteristic Curves (CCs)**. The code implements the **NN-CC** method and its variants, comparing them against standard approaches like SINDy and Symbolic Regression (SR).

The repository includes scripts to reproduce the results and figures presented in the manuscript for two benchmark systems:
1.  **Chaotic Duffing Oscillator:** A continuous system with polynomial nonlinearities.
2.  **Stick-Slip Friction:** A discontinuous system with dry friction.

## Implemented Methods

The codebase covers the following identification strategies discussed in the paper:

* **NN-CC:** Neural Network-based Characteristic Curves (baseline).
* **NN-CC+sym:** NN-CC with symmetry constraints enforced via the loss function.
* **NN-CC+post-SR:** NN-CC with Symbolic Regression (PySR) as a post-processing step.
* **NN-CC+sym+post-SR:** The combined approach.
* **Benchmarks:** Implementations of SINDy-CC, Poly-CC, and standard Symbolic Regression (SR) for comparison.

## Dependencies

The code is written in Python. Key dependencies include:

* **PyTorch** (for Neural Network training)
* **PySR** (for Symbolic Regression steps)
* **PySINDy** (Compatible with v1.7.5)
* **NumPy** & **SciPy** (for data generation and integration)
* **Matplotlib** (for visualization)

## Usage

The repository is organized into two main directories corresponding to the benchmark systems: `Duffing` and `Stick-slip`. Each folder contains the following scripts:

* **Generation and Training:** A unified script to generate synthetic datasets, train the NN-CC variants (including symmetry constraints), and fit the benchmark models (SINDy, SR, etc.).
* **Simulation and Evaluation:** A script that also perform forward integrations using the identified models and calculate performance metrics (e.g., separation time for Duffing, RMSE for Stick-Slip).

*(Note: Detailed instructions on running specific scripts will be updated upon final publication).*

## Citation

If you use this code or methods in your research, please cite the associated manuscript (citation details to be added upon publication).

## License

[MIT License]
