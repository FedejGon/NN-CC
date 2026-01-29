# NN-CC , SINDy-CC , SR-CC 

This repository contains the source codes associated with the manuscript:

**"Integrating prior knowledge in equation discovery: Interpretable symmetry-informed neural networks and symbolic regression via characteristic curves"**

## 📖 Overview

This work introduces a modular framework for identifying nonlinear dynamical systems by decomposing dynamics into interpretable **Characteristic Curves (CCs)**.  implements the **NN-CC** method and its variants based on PyTorch, and variants of **SINDy** and **Symbolic Regression (SR)**  them against standard approaches like SINDy and Symbolic Regression (SR).


Crucially, this framework allows practitioners to explicitly embed domain expertise, such as geometric symmetries, directly into the learning process. It also features a post-processing stage where the learned curves are converted into analytical expressions using Symbolic Regression.

The repository includes scripts to reproduce the results for two benchmark systems:
1.  **Chaotic Duffing Oscillator:** A continuous system with polynomial nonlinearities.
2.  **Stick-Slip Friction:** A discontinuous system with dry friction.


## Implemented Methods

The codebase covers the following identification strategies discussed in the paper:

### Neural Network Approaches (NN-CC)
* **NN-CC:** Neural Network-based Characteristic Curves (baseline).
* **NN-CC$_{+sym}$:** NN-CC with symmetry constraints enforced via the loss function.
* **NN-CC$_{+post-SR}$:** NN-CC with Symbolic Regression (PySR) as a post-processing step.
* **NN-CC$_{+sym+post-SR}$:** The combined approach using both symmetry priors and symbolic refinement.

### Sparse Regression Approaches (SINDy) and least-squares with Polynomial basis (Poly-CC) 
* **Poly-CC:** Characteristic curves represented by fixed polynomial basis functions.
* **SINDy:** Standard Sparse Identification of Nonlinear Dynamics.
* **SINDy-CC:** SINDy constrained to the specific additive structure of the Characteristic Curves ($k_0=1$).
* **SINDy-CC$_{+sym}$:** SINDy-CC with a candidate library restricted to symmetric terms (e.g., odd polynomials).
* **SINDy-CC$_{+sym+post-SR}$:** A hybrid approach applying symbolic regression to the sparse regression results.

### Symbolic Regression Approaches
* **SR:** Standard Symbolic Regression (using PySR) searching the full bivariate space $f(x, \dot{x})$.
* **SR-CC:** Structured Symbolic Regression that strictly enforces the additive separation $f_1(\dot{x}) + f_2(x)$ via alternating optimization.


## Dependencies

The codes are written in Python. Key dependencies include:

* **PyTorch** (for NN-CC methods)
* **PySR** (for Symbolic Regression)
* **PySINDy** (Compatible with v1.7.5)
* **NumPy** & **SciPy** (for data generation and integration)
* **Matplotlib** (for visualization)

## Installation

You can install the required Python libraries via pip:

```bash
pip install torch numpy scipy matplotlib pysindy==1.7.5 pysr
```





## Citation

If you use this code or methods in your research, please cite the associated manuscript.

## License

[MIT License]
