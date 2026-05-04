import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import eval_legendre

# ---------------------------------------------------------
# 1. Basis Functions on Reference Element [-1, 1]
# ---------------------------------------------------------
def phi(k, xi):
    """Evaluates the k-th basis function at reference coordinate xi."""
    if k == 0:
        return (1 - xi) / 2.0
    elif k == 1:
        return (1 + xi) / 2.0
    else:
        # Analytical antiderivative of Legendre polynomial L_{k-1}
        # Integral of P_{k-1}(x) is (P_k(x) - P_{k-2}(x)) / (2k - 1)
        return (eval_legendre(k, xi) - eval_legendre(k-2, xi)) / (2*k - 1)

def dphi(k, xi):
    """Evaluates the derivative of the k-th basis function."""
    if k == 0:
        return -0.5 * np.ones_like(xi)
    elif k == 1:
        return 0.5 * np.ones_like(xi)
    else:
        # Derivative of the integral of L_{k-1} is just L_{k-1}
        return eval_legendre(k-1, xi)

# ---------------------------------------------------------
# 2. Main FEM Solver Function
# ---------------------------------------------------------
def solve_fem_1d(p, N):
    """
    Solves -u'' + u = 2sin(x) on (0, pi) with u(0)=u(pi)=0.
    p: Polynomial degree
    N: Number of uniform elements
    Returns: element size (h) and the energy norm of the error.
    """
    domain = (0, np.pi)
    x_nodes = np.linspace(domain[0], domain[1], N + 1)
    h = x_nodes[1] - x_nodes[0]
    
    # Total Degrees of Freedom (DOFs)
    # N+1 nodal DOFs, plus (p-1) internal bubble DOFs per element
    num_dofs = (N + 1) + N * (p - 1)
    
    A = sp.lil_matrix((num_dofs, num_dofs))
    F = np.zeros(num_dofs)
    
    # Gauss-Legendre Quadrature rules for exact integration of polynomials
    # Stiffness matrix needs degree ~2p, load vector needs a bit more for sin(x)
    quad_deg = max(10, 2*p + 2)
    xi_q, w_q = np.polynomial.legendre.leggauss(quad_deg)
    
    # Pre-evaluate basis functions at quadrature points to save time
    phi_vals = np.array([phi(k, xi_q) for k in range(p + 1)])
    dphi_vals = np.array([dphi(k, xi_q) for k in range(p + 1)])
    
    # ---------------------------------------------------------
    # Assembly Loop
    # ---------------------------------------------------------
    for e in range(N):
        a, b = x_nodes[e], x_nodes[e+1]
        
        # Mapped coordinates x(xi)
        x_mapped = (a + b) / 2.0 + (h / 2.0) * xi_q
        
        # Local to Global DOF mapping
        # [left_node, right_node, bubble_1, bubble_2, ...]
        local_to_global = [e, e+1] + list(range(N + 1 + e*(p-1), N + 1 + (e+1)*(p-1)))
        
        # Calculate local stiffness matrix and load vector
        for i in range(p + 1):
            # Load vector integration: 2 * sin(x) * phi_i * dx
            F[local_to_global[i]] += np.sum(w_q * 2 * np.sin(x_mapped) * phi_vals[i] * (h / 2.0))
            
            for j in range(p + 1):
                # Stiffness matrix integration: ((2/h)^2 * dphi_i * dphi_j + phi_i * phi_j) * dx
                integrand = ((2.0/h)**2 * dphi_vals[i] * dphi_vals[j] + phi_vals[i] * phi_vals[j])
                A[local_to_global[i], local_to_global[j]] += np.sum(w_q * integrand * (h / 2.0))

    # ---------------------------------------------------------
    # Boundary Conditions
    # ---------------------------------------------------------
    # u(0) = 0 (DOF 0) and u(pi) = 0 (DOF N)
    for bc_dof in [0, N]:
        A[bc_dof, :] = 0
        A[bc_dof, bc_dof] = 1.0
        F[bc_dof] = 0.0

    # Solve the linear system
    U = spla.spsolve(A.tocsr(), F)
    
    # ---------------------------------------------------------
    # Error Evaluation (Energy Norm)
    # ---------------------------------------------------------
    # True solution u(x) = sin(x) -> u'(x) = cos(x)
    xi_err, w_err = np.polynomial.legendre.leggauss(15) # High degree for accurate error
    phi_err = np.array([phi(k, xi_err) for k in range(p + 1)])
    dphi_err = np.array([dphi(k, xi_err) for k in range(p + 1)])
    
    error_energy_sq = 0.0
    
    for e in range(N):
        a, b = x_nodes[e], x_nodes[e+1]
        x_mapped = (a + b) / 2.0 + (h / 2.0) * xi_err
        
        local_dofs = [e, e+1] + list(range(N + 1 + e*(p-1), N + 1 + (e+1)*(p-1)))
        U_local = U[local_dofs]
        
        # Calculate numerical solution and its derivative at quadrature points
        u_h = np.dot(U_local, phi_err)
        u_h_prime = np.dot(U_local, dphi_err) * (2.0 / h) # Chain rule!
        
        # Exact solution and derivative
        u_exact = np.sin(x_mapped)
        u_exact_prime = np.cos(x_mapped)
        
        # Integrate (u_exact' - u_h')^2 + (u_exact - u_h)^2
        integrand = (u_exact_prime - u_h_prime)**2 + (u_exact - u_h)**2
        error_energy_sq += np.sum(w_err * integrand * (h / 2.0))

    return h, np.sqrt(error_energy_sq)

# ---------------------------------------------------------
# 3. Execution and Plotting
# ---------------------------------------------------------
if __name__ == "__main__":
    polynomial_degrees = [1, 2, 3]
    elements_array = [4, 8, 16, 32, 64, 128]
    
    plt.figure(figsize=(8, 6))
    
    for p in polynomial_degrees:
        h_values = []
        errors = []
        
        for N in elements_array:
            h, err = solve_fem_1d(p, N)
            h_values.append(h)
            errors.append(err)
            
        h_values = np.array(h_values)
        errors = np.array(errors)
        
        # Plot the error
        plt.loglog(h_values, errors, marker='o', label=f'$\mathcal{{P}}^{p}$ Elements')
        
        # Calculate and print the observed convergence rate (slope of the log-log plot)
        rate = np.polyfit(np.log(h_values), np.log(errors), 1)[0]
        print(f"Polynomial Degree p={p} | Observed Convergence Rate: {rate:.2f}")

    # Add reference slopes
    h_ref = np.array([h_values[0], h_values[-1]])
    plt.loglog(h_ref, 0.1 * h_ref**1, 'k--', alpha=0.5, label='$\mathcal{O}(h)$ reference')
    plt.loglog(h_ref, 0.1 * h_ref**2, 'k-.', alpha=0.5, label='$\mathcal{O}(h^2)$ reference')
    plt.loglog(h_ref, 0.1 * h_ref**3, 'k:', alpha=0.5, label='$\mathcal{O}(h^3)$ reference')

    plt.xlabel('Element Size ($h$)')
    plt.ylabel('Energy Norm Error ($||u - u_h||_E$)')
    plt.title('Convergence of 1D FEM for $-u^{\prime\prime} + u = 2\sin(x)$')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()