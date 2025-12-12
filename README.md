# Analysis of Partial Differential Equations using Physics-Informed Neural Networks and Numerical Methods

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project explores solving complex partial differential equations (PDEs) using two complementary approaches: traditional numerical methods and Physics-Informed Neural Networks (PINNs). The work demonstrates how combining physics with machine learning opens new possibilities for solving real-world science and engineering problems.

## Overview

PDEs appear throughout physics and engineering but can be extremely difficult or impossible to solve analytically. This project implements and compares:

- **Physics-Informed Neural Networks (PINNs)**: Neural networks that learn not just from data but also respect the underlying physical laws through specialized loss functions
- **Traditional Numerical Methods**: Central difference schemes for spatial derivatives, Euler method for simple time stepping, and Runge-Kutta 4th order (RK4) for higher accuracy

## Equations Solved

| Equation | Type | Application |
|----------|------|-------------|
| 1st Order ODE (dy/dx = eˣ) | Basic | Foundation for understanding |
| 1D Poisson Equation | 2nd Order Spatial | Diffusion problems |
| 1D Burgers' Equation | Nonlinear PDE | Fluid dynamics, shock waves |
| 3D Burgers' Equation | Coupled Nonlinear PDEs | Turbulent flow modeling |
| 1D Euler Equations | Hyperbolic System | Compressible gas dynamics |

## Key Findings

- **PINNs excel** at smooth problems with continuous solutions
- **Numerical methods** remain more reliable for capturing sharp discontinuities like shock waves
- **Hybrid approaches** combining both methods show promise for complex real-world problems
- PINNs struggle with shock waves due to smooth activation functions (tanh, sigmoid) that inherently smooth out sharp features

## Project Structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_first_order_ode.ipynb
│   ├── 02_poisson_equation.ipynb
│   ├── 03_burgers_1d.ipynb
│   ├── 04_burgers_3d.ipynb
│   └── 05_euler_equations.ipynb
├── src/
│   ├── pinn/
│   │   ├── __init__.py
│   │   ├── network.py          # Neural network architecture
│   │   ├── losses.py           # Physics-informed loss functions
│   │   └── training.py         # Training utilities
│   ├── numerical/
│   │   ├── __init__.py
│   │   ├── central_diff.py     # Central difference methods
│   │   ├── rk4.py              # Runge-Kutta 4th order
│   │   └── euler_solver.py     # Euler equations solver
│   └── utils/
│       ├── plotting.py         # Visualization utilities
│       └── validation.py       # Solution comparison tools
├── results/
│   └── figures/                # Generated plots and figures
└── docs/
    └── paper.pdf               # Full project report
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pinn-pde-solver.git
cd pinn-pde-solver

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.9.0
```

## Usage

### 1D Burgers' Equation (PINN)

```python
import torch
import torch.nn as nn

# Define the PINN architecture
class BurgersPINN(nn.Module):
    def __init__(self, layers=[2, 20, 20, 20, 20, 20, 1]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = nn.Tanh()
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        for layer in self.layers[:-1]:
            inputs = self.activation(layer(inputs))
        return self.layers[-1](inputs)

# Physics-informed loss
def physics_loss(model, x, t, nu=0.01):
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u = model(x, t)
    
    # Compute derivatives using automatic differentiation
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    # Burgers' equation residual: u_t + u*u_x - nu*u_xx = 0
    residual = u_t + u * u_x - nu * u_xx
    
    return torch.mean(residual**2)
```

### Numerical Solution (RK4 + Central Difference)

```python
import numpy as np

def burgers_numerical(nx=100, nt=1000, L=2.0, T=1.0, nu=0.01):
    dx = L / (nx - 1)
    dt = T / nt
    x = np.linspace(0, L, nx)
    
    # Initial condition
    u = np.sin(np.pi * x)
    
    def rhs(u):
        dudx = np.zeros_like(u)
        d2udx2 = np.zeros_like(u)
        
        # Central differences (interior points)
        dudx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
        d2udx2[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        
        return -u * dudx + nu * d2udx2
    
    # RK4 time stepping
    for _ in range(nt):
        k1 = rhs(u)
        k2 = rhs(u + 0.5*dt*k1)
        k3 = rhs(u + 0.5*dt*k2)
        k4 = rhs(u + dt*k3)
        u = u + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return x, u
```

## Results

### 1D Burgers' Equation
The PINN successfully captures the evolution of the velocity field and shock formation, though slight smoothing is observed near discontinuities.

### Euler Equations (Sod Shock Tube)
Both methods capture the shock wave, rarefaction fan, and contact discontinuity. The numerical method shows sharper discontinuities while the PINN provides smoother approximations.

### 3D Burgers' Equation
The PINN qualitatively captures the velocity field decay due to viscosity across all three spatial dimensions.

## PINN Architecture Details

| Component | Configuration |
|-----------|---------------|
| Hidden Layers | 5-10 layers depending on problem complexity |
| Neurons per Layer | 20-64 neurons |
| Activation Function | Tanh (enables learning nonlinear patterns) |
| Optimizer | Adam (learning rate: 1e-3 to 1e-2) |
| Loss Function | MSE combining PDE residual, IC, and BC losses |

### Loss Function Structure

```
Total Loss = λ_pde × L_pde + λ_ic × L_ic + λ_bc × L_bc
```

Where:
- **L_pde**: Residual of the governing PDE at collocation points
- **L_ic**: Mismatch with initial conditions
- **L_bc**: Mismatch with boundary conditions
- **λ**: Weighting factors to balance loss components

## Limitations and Future Work

### Current Limitations
- PINNs struggle with sharp discontinuities due to smooth activation functions
- 3D problems require significantly more computational resources
- Hyperparameter tuning can be challenging

### Potential Improvements
- Implement adaptive activation functions for better shock capturing
- Explore domain decomposition methods for complex geometries
- Investigate hybrid PINN-numerical schemes
- Add support for inverse problems (parameter estimation)

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

2. Burgers, J. M. (1948). A mathematical model illustrating the theory of turbulence. *Advances in Applied Mechanics*, 1, 171-199.

3. Sod, G. A. (1978). A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws. *Journal of Computational Physics*, 27(1), 1-31.

## Authors

- **Ritik Kumar** - Indian Institute of Science, Bangalore
- **Prof. Rishita Das** (Supervisor) - Department of Aerospace, IISc Bangalore

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Indian Institute of Science, Bangalore for computational resources
- PyTorch team for the deep learning framework
- The scientific computing community for foundational numerical methods

---

*This project was completed as part of coursework at the Indian Institute of Science, Bangalore.*
