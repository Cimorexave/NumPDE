import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def solve_fem(N, beta):
    # 1. Mesh Generation
    # x_i = (i/N)^beta
    nodes = np.linspace(0, 1, N + 1)
    x = nodes**beta
    if N == 10:
        print(f"Nodes for N={N}, beta={beta}: {x}")
    h = np.diff(x) # Width of each individual element

    # 2. Problem Definitions
    # u(x) = x^(2/3) * (1 - x)
    # f(x) = -u''(x) = (2/9)x^(-4/3) + (10/9)x^(-1/3)
    f = lambda x: (2/9) * x**(-4/3) + (10/9) * x**(-1/3)
    u_exact_prime = lambda x: (2/3) * x**(-1/3) - (5/3) * x**(2/3)

    # 3. Assembly of Stiffness Matrix (K) and Load Vector (F)
    # We solve for internal nodes only (1 to N-1) because u(0)=u(1)=0
    K = np.zeros((N - 1, N - 1))
    F = np.zeros(N - 1)

    for i in range(1, N):
        idx = i - 1 # Array index
        
        # Local Stiffness (Standard 1D FEM for linear elements)
        # Main diagonal: 1/h_i + 1/h_{i+1}
        K[idx, idx] = 1/h[i-1] + 1/h[i]
        # Off-diagonals: -1/h_i
        if idx > 0:
            K[idx, idx-1] = -1/h[i-1]
        if idx < N - 2:
            K[idx, idx+1] = -1/h[i]

        # Local Load Vector (Integral of f * phi_i)
        # phi_i is 1 at x_i, and 0 at x_{i-1} and x_{i+1}
        phi_left = lambda val: (val - x[i-1]) / h[i-1]
        phi_right = lambda val: (x[i+1] - val) / h[i]
        
        term1, _ = quad(lambda val: f(val) * phi_left(val), x[i-1], x[i])
        term2, _ = quad(lambda val: f(val) * phi_right(val), x[i], x[i+1])
        F[idx] = term1 + term2

    # 4. Solve Linear System
    u_internal = np.linalg.solve(K, F)
    u_h = np.concatenate(([0], u_internal, [0]))

    # 5. Energy Norm Error Calculation
    # ||e||_E^2 = sum over elements of integral (u' - u_h')^2
    energy_error_sq = 0
    for i in range(N):
        # u_h' is constant on each element: (u_i+1 - u_i) / h_i
        uh_prime = (u_h[i+1] - u_h[i]) / h[i]
        integrand = lambda val: (u_exact_prime(val) - uh_prime)**2
        err, _ = quad(integrand, x[i], x[i+1])
        energy_error_sq += err

    return np.sqrt(energy_error_sq)

# --- Analysis ---
N_values = [10, 20, 40, 80, 160]
betas = [1.0, 1.5, 2.0, 2.5, 5, 7, 10] # beta=1 is equidistant

plt.figure(figsize=(10, 6))

for b in betas:
    errors = [solve_fem(N, b) for N in N_values]
    plt.loglog(N_values, errors, '-o', label=f'beta = {b}')

# Reference line for O(N^-1) convergence
plt.loglog(N_values, [1/n for n in N_values], '--', color='gray', label='O(1/N) Slope')

plt.xlabel('Number of Elements (N)')
plt.ylabel('Energy Norm Error')
plt.title('FEM Error Convergence with Graded Mesh')
plt.legend()
plt.grid(True, which="both", ls="-")
plt.show()