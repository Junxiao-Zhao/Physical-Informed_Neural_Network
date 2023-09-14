# Physical-Informed Neural Networks
Physics-Informed Neural Networks (PINNs) combine the power of neural networks with the physical laws governing a system, allowing for the incorporation of domain knowledge and enforcing physical constraints during training, making them suitable for solving partial differential equations and related problems.

## 1D Curve
Fitting $y=e^x$ with:
- Govern function: $\frac{dy}{dx}=y, x\in[0, 1]$
- Data: $y(0)=1$

![1D curve fitting](./imgs/pinn%201d.png)

## 2D Burgers' Equation
Fitting 2D Burgers' Equation with:
- Govern function: $z_x + zz_y - \frac{0.01}{\pi}z_{yy}=0, x\in [0, 1], y\in [-1, 1]$
- Data: 
    - $z(0, y) = -\sin(\pi y)$
    - $z(x, -1) = z(x, 1) = 0$

![2D Burgers' Equation](./imgs/2d%20burgers.png)