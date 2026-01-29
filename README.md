# NN-CC , SINDy-CC , SR-CC 

This repository contains the source codes associated with the manuscript:

**"Integrating prior knowledge in equation discovery: Interpretable symmetry-informed neural networks and symbolic regression via characteristic curves"**

## 📖 Overview

This work introduces a modular framework for identifying nonlinear dynamical systems by decomposing dynamics into interpretable **Characteristic Curves (CCs)**.  implements the **NN-CC** method and its variants based on PyTorch, and variants of **SINDy** and **Symbolic Regression (SR)**  them against standard approaches like SINDy and Symbolic Regression (SR).


Crucially, this framework allows practitioners to explicitly embed domain expertise, such as geometric symmetries, directly into the learning process. It also features a post-processing stage where the learned curves are converted into analytical expressions using Symbolic Regression.

The repository includes scripts to reproduce the results for two benchmark systems:
1.  **Chaotic Duffing Oscillator:** A continuous system with polynomial nonlinearities.
2.  **Stick-Slip Friction:** A discontinuous system with dry friction.

## Installation

The codes are written in Python. Key dependencies include:

* **PyTorch** (for NN-CC methods)
* **PySR** (for Symbolic Regression)
* **PySINDy** (Compatible with v1.7.5)
* **NumPy** & **SciPy** (for data generation and integration)
* **Matplotlib** (for visualization)

You can install the required Python libraries via pip:

```bash
pip install torch numpy scipy matplotlib pysindy==1.7.5 pysr
```

## Implemented Methods

The codes in the root folder provide basic implementation examples. For full reproducibility of the figures shown in the paper, please refer to the corresponding FigX folders. We implement the following methods:

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

## 🏛️ Citation

In case of using or adapting some of the codes or methods in your research, please cite the associated manuscript.

Additionally:
i) In case of using NN-CC method, please cite:
  - Gonzalez, F. J. and Lara, L. P. "[Interpretable neural network system identification method for two families of second-order systems based on characteristic curves](https://doi.org/10.1007/s11071-025-11744-6)." Nonlinear Dyn. 113, 33063–33086 (2025).

ii) In case of using Poly-CC method, please cite:
  - Gonzalez, F.J. "[Determination of the characteristic curves of a nonlinear first order system from fourier analysis](https://doi.org/10.1038/s41598-023-29151-5)." Sci. Rep., vol. 13, 1955, (2023).
  - Gonzalez, F.J. "[System identification based on characteristic curves: a mathematical connection between power series and Fourier analysis for first-order nonlinear systems](https://doi.org/10.1007/s11071-024-09890-4)." Nonlinear Dyn. 112, 16167–16197 (2024).

iii) In case of using SINDy-CC methods, please cite:
  - Kaptanoglu, A.A.; et al. "[PySINDy: A comprehensive Python package for robust sparse system identification](https://doi.org/10.21105/joss.03994)." J. Open Source Softw. vol. 7, 3994, (2021)
  - de Silva, B.M.; et al. "[PySINDy: A Python package for the sparse identification of nonlinear dynamical systems from data](https://doi.org/10.21105/joss.03994)." J. Open Source Softw. vol. 5, 2104, (2020)

iv) In case of using SR-CC variants or post-SR, please cite:
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

@article{Gonzalez2025,
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


@article{Kaptanoglu2022, 
  doi = {10.21105/joss.03994}, 
  year = {2022}, 
  publisher = {The Open Journal}, 
  volume = {7}, 
  number = {69}, 
  pages = {3994}, 
  author = {Kaptanoglu, Alan A. and de Silva, Brian M. and Fasel, Urban and Kaheman, Kadierdan and Goldschmidt, Andy J. and Callaham, Jared and Delahunt, Charles B. and Nicolaou, Zachary G. and Champion, Kathleen and Loiseau, Jean-Christophe and Kutz, J. Nathan and Brunton, Steven L.}, 
  title = {PySINDy: A comprehensive Python package for robust sparse system identification}, 
  journal = {J. Open Source Softw.},
}


@article{Silva2020, 
  doi = {10.21105/joss.02104}, 
  year = {2020},
  publisher = {The Open Journal}, 
  volume = {5}, 
  number = {49}, 
  pages = {2104}, 
  author = {de Silva, Brian M. and Champion, Kathleen and Quade, Markus and Loiseau, Jean-Christophe and Kutz, J. Nathan and Brunton, Steven L.}, 
  title = {PySINDy: A Python package for the sparse identification of nonlinear dynamical systems from data}, 
  journal = {J. Open Source Softw.} ,
} 

```


### 🤝 We are open to collaborations and adding new possible features.
Reach out for a possible collaboration to:
 - Federico J. Gonzalez: fgonzalez@ifir-conicet.gov.ar


## License

[MIT License]
