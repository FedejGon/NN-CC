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
* **NN-CC<sub>+sym</sub>:** NN-CC with symmetry constraints enforced via the loss function.
* **NN-CC<sub>+post-SR</sub>:** NN-CC with Symbolic Regression (PySR) as a post-processing step.
* **NN-CC<sub>+sym+post-SR</sub>:** The combined approach using both symmetry priors and symbolic refinement.

### Sparse Regression Approaches (SINDy) and least-squares with Polynomial basis (Poly-CC) 
* **Poly-CC:** Characteristic curves represented by fixed polynomial basis functions.
* **SINDy:** Standard Sparse Identification of Nonlinear Dynamics.
* **SINDy-CC:** SINDy constrained to the specific additive structure of the Characteristic Curves ($k_0=1$).
* **SINDy-CC<sub>+sym</sub>:** SINDy-CC with a candidate library restricted to symmetric terms (e.g., odd polynomials).
* **SINDy-CC<sub>+sym+post-SR</sub>:** A hybrid approach applying symbolic regression to the sparse regression results.

### Symbolic Regression Approaches
* **SR:** Standard Symbolic Regression (using PySR) searching the full bivariate space $f(x, \dot{x})$.
* **SR-CC:** Structured SR version that strictly enforces the identification of univariate functions $f_1(\dot{x})$ and $f_2(x)$ through a nested outer loop optimization.


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





## 🏛️ Citation

If you use this code or methods in your research, please cite the associated manuscript.




In case of using NN-CC method, additionally cite:
  - Gonzalez, F. J. and Lara, L. P. "[Interpretable neural network system identification method for two families of second-order systems based on characteristic curves](https://doi.org/10.1007/s11071-025-11744-6)." Nonlinear Dyn. (2025)

In case of using Poly-CC method, additionally cite:
  - Gonzalez, F.J. "[Determination of the characteristic curves of a nonlinear first order system from fourier analysis](https://doi.org/10.1038/s41598-023-29151-5)." Sci. Rep., vol. 13, 1955, (2023).
  - Gonzalez, F.J. "[System identification based on characteristic curves: a mathematical connection between power series and Fourier analysis for first-order nonlinear systems](https://doi.org/10.1007/s11071-024-09890-4)." Nonlinear Dyn. 112, 16167–16197 (2024).

In case of using post-SR and/or SymbReg-CC methods, additionally cite:
  - Cranmer, M. "[Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl](https://doi.org/10.48550/arXiv.2305.01582)." arXiv preprint arXiv:2305.01582 (2023).





## Citation BibTex


```bibtex
@article{Gonzalez2023,
  title     = {Determination of the characteristic curves of a nonlinear first order system from Fourier analysis},
  author    = {Gonzalez, Federico J.},
  journal   = {Sci. Rep.},
  publisher = {Springer Science and Business Media LLC},
  volume    =  13,
  number    =  1,
  pages     = {1955},
  month     =  feb,
  year      =  2023,
  doi =   {10.1038/s41598-023-29151-5},
}

@article{Gonzalez2024,
  title = {System identification based on characteristic curves: a mathematical connection between power series and Fourier analysis for first-order nonlinear systems},
  author = {{F. J. Gonzalez}},
  volume = {112},
  issn = {1573-269X},
  doi = {10.1007/s11071-024-09890-4},
  number = {18},
  journal = {Nonlinear Dyn.},
  publisher = {Springer Science and Business Media LLC},
  year = {2024},
  month = jul,
  pages = {16167–16197}
}

@article{Gonzalez2025nody,
  title = {{Interpretable neural network system identification method for two families of second-order systems based on characteristic curves}},
  volume = {113},
  ISSN = {1573-269X},
  DOI = {10.1007/s11071-025-11744-6},
  number = {24},
  journal = {Nonlinear Dyn.},
  publisher = {Springer Science and Business Media LLC},
  author = {Gonzalez,  Federico J. and Lara,  Luis P.},
  year = {2025},
  month = sep,
  pages = {33063–33086}
}


@article{Cranmer2023PySR,
  title={Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl},
  author={Miles Cranmer},
  journal={arXiv preprint arXiv:2305.01582},      
  year={2023},
  eprint={2305.01582},
  url={https://arxiv.org/abs/2305.01582},
}
```


### 🤝 We are open to collaborations and adding new possible features.
Please share your [![Ideas](https://img.shields.io/badge/ideas-github-informational)](https://github.com/FedejGon/pyCC.id/discussions/categories/ideas) or reach out for a possible collaboration to:
 - Federico J. Gonzalez: fgonzalez@ifir-conicet.gov.ar


## License

[MIT License]
