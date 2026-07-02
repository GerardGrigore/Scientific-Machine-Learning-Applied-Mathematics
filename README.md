# :infinity: Scientific Machine Learning Applied Mathematics :brain:
This repository documents a journey into **Scientific Machine Learning (SciML)** and **Operator Learning**, combining modern Deep Learning with concepts from **Functional Analysis**, **Numerical Analysis**, and **Partial Differential Equations**.

The objective is not only to reproduce existing methods, but also to understand and implement every component from first principles before moving towards state-of-the-art Neural Operators.

# 🚀 Project Roadmap

The project is divided into several successive stages.

## 📖 Stage 1 — Understanding Neural Networks from Scratch

The first part of this project focuses on implementing a fully-connected **Multi-Layer Perceptron (MLP)** without relying on any Deep Learning framework.
The objective is to understand:

- Forward propagation.
- Activation functions.
- Loss functions.
- Backpropagation.
- Stochastic Gradient Descent.
- Weight initialization.
- Decision boundaries.

The network is trained on the classical **Two-Moons** classification dataset. See the file Scientific_Machine_Learning_I and the Python script Two_Moons_NN.py.
This first implementation serves as a mathematical introduction to optimization in neural networks and provides a solid understanding of gradient-based learning.

## 🐍 Stage 2 — Transition to PyTorch

Once every component has been implemented manually, the same architecture is reproduced using **PyTorch**.
The goal is to become familiar with:

- `nn.Module`.
- Automatic differentiation.
- Optimizers.
- DataLoaders.
- GPU computation.
- Modular network design.

This stage validates that the theoretical implementation and the framework implementation produce identical behaviour. See the file Scientific_Machine_Learning_I and the Python script Two_Moons_NN_Classical_Libraries.py.

## 🌊 Stage 3 — Fourier Neural Operators

The next objective is to move from finite-dimensional function approximation to **Operator Learning**.
A complete implementation of the **Fourier Neural Operator (FNO)** is developed from scratch following the original paper.
Topics covered include:

- Fourier Transform.
- Spectral Convolutions.
- Fourier Layers.
- Lifting and Projection operators.
- Complex-valued weights.
- Mode truncation.
- Operator learning.

The model is trained to learn the solution operator of the one-dimensional **Burgers' equation**. See the file Scientific_Machine_Learning_II and the Python script FNO_Burger.py.

## ♾️ Stage 4 — Physics-Informed Neural Operators

After learning the solution operator from data, the repository introduces **Physics-Informed Neural Operators (PINO)**.
The objective is to incorporate physical constraints directly into the optimization process through the governing Partial Differential Equation.
Topics include:

- Physics-informed loss functions.
- PDE residual minimization.
- Automatic differentiation.
- Data loss and PDE loss.
- Fine-tuning strategies.
- Generalization to unseen initial conditions.

See the file Scientific_Machine_Learning_III and the Python script PINO_Burger.py.

# 📚 Mathematical Background

This project combines ideas from several mathematical disciplines:

- Functional Analysis.
- Measure Theory.
- Numerical Analysis.
- Optimization.
- Partial Differential Equations.
- Fourier Analysis.
- Machine Learning.

The long-term objective is to better understand the mathematical foundations of Neural Operators and their approximation properties in infinite-dimensional function spaces.

# 🛠️ Technologies and libraries used

- Python.
- NumPy.
- SciPy.
- PyTorch.
- Matplotlib.

# 📚 References

- Li, Z. et al. — *Fourier Neural Operator for Parametric Partial Differential Equations*
- Li, Z. et al. — *Physics-Informed Neural Operator for Learning Partial Differential Equations*
- Lu, L. et al. — *Learning Nonlinear Operators via DeepONet*

## ⭐ Motivation

This repository was created as part of my journey toward research in **Applied Mathematics** and **Scientific Machine Learning**.
Rather than treating Neural Operators as black-box architectures, the goal is to understand the mathematical principles that make them effective, from classical neural networks to operator learning in infinite-dimensional spaces.
Every implementation is therefore built progressively, emphasizing both mathematical rigor and practical understanding.
