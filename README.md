Stochastic Spiral Optimizer (SPO) – User Manual
Author: Ian Brown
Version: 1.0

------------------------------------------------------------

Overview

SPO is a global optimization algorithm designed to find the minimum of continuous functions. It operates by iteratively rotating and contracting a set of points towards the best discovered point, and updating the point used as the center of rotation whenever a new best point is discovered.

This project implements the following papers:
https://ieeexplore.ieee.org/document/6557686
https://ieeexplore.ieee.org/document/7919176
https://ieeexplore.ieee.org/document/8261609

------------------------------------------------------------

Files

- SPO.py – Contains the SpiralOptimizer class.
- Main.ipynb - Contains the definitions of the benchmark functions and tests SPO on different versions of them.
- README.txt – This manual.

------------------------------------------------------------

Installation

SPO requires only Python 3.6+ and NumPy (and Jupyter).

Install with:
pip install numpy jupyter

------------------------------------------------------------

Example Usage

Import and run the optimizer:

from spo_optimizer import SpiralOptimizer

Create an optimizer object:

spo = SpiralOptimizer(
    objective_function=rastrigin,
    dim=10,
    bounds=(-5,5),
    num_points=50,
    r=0.95
    theta=0.1
    rand_theta=False
)

Run the optimizer:

spo.run(max_iterations=1000)
