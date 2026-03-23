import numpy as np
import matplotlib.pyplot as plt

def solve_poisson_fem(N):
    h = 1.0 / N
    x = np.linspace(0, 1, N + 1)
    
    # Stiffness Matrix A and Load Vector f
    # N+1 size to include the boundary nodes
    A = np.zeros((N + 1, N + 1))
    f_vec = np.zeros(N + 1)
    
    # using standard 1D finite element stencil: [-1/h, 2/h, -1/h]
    for i in range(1, N):
        A[i, i-1] = -1/h
        A[i, i]   = 2/h
        A[i, i+1] = -1/h
        
        # load vector contribution (trapezoidal approximation for f(x)=3x)
        f_vec[i] = 3 * x[i] * h
        
    # dirichlet boundary conditions
    # u(0) = 0
    A[0, 0] = 1.0
    f_vec[0] = 0.0
    # u(1) = 1
    A[N, N] = 1.0
    f_vec[N] = 1.0
    
    # solve linear system
    u_h = np.linalg.solve(A, f_vec)
    
    #  calculate maximum error e_N at midpoints
    midpoints = (x[:-1] + x[1:]) / 2
    u_exact_mid = -0.5 * midpoints**3 + 1.5 * midpoints
    u_fem_mid = (u_h[:-1] + u_h[1:]) / 2
    
    error_max = np.max(np.abs(u_exact_mid - u_fem_mid))
    
    return error_max



N_values = [4, 8, 16, 32, 64, 128, 256, 512]
errors = [solve_poisson_fem(N) for N in N_values]

plt.figure(figsize=(8, 6))
plt.loglog(N_values, errors, 'o-', label='measured error $e_N$')

# reference line for O(h^2) convergence (slope of -2 on log-log)
h_ref = 1.0 / np.array(N_values)
plt.loglog(N_values, (h_ref**2) * errors[0]/h_ref[0]**2, '--', label='ref $O(h^2)$')

plt.xlabel('(N)')
plt.ylabel('max midpoint error ($e_N$)')
plt.title('convergence analysis of 1D FEM for $-u\'\' = 3x$')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()

for i in range(1, len(errors)):
    rate = np.log(errors[i-1]/errors[i]) / np.log(2)
    print(f"N: {N_values[i]:4d} | Error: {errors[i]:.2e} | Conv. Rate: {rate:.2f}")