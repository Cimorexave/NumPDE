import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def exact_u(x, y):
    return 0.5 * x * (1 - x) * y * (1 - y)

def exact_grad_u(x, y):
    du_dx = 0.5 * (1 - 2*x) * y * (1 - y)
    du_dy = 0.5 * x * (1 - x) * (1 - 2*y)
    return du_dx, du_dy

def source_f(x, y):
    return y * (1 - y) + x * (1 - x)

def solve_fem_poisson(N):
    # 1. Mesh Generation
    h = 1.0 / N
    x_1d = np.linspace(0, 1, N+1)
    y_1d = np.linspace(0, 1, N+1)
    X, Y = np.meshgrid(x_1d, y_1d)
    nodes = np.vstack([X.ravel(), Y.ravel()]).T
    n_nodes = nodes.shape[0]
    
    # Create regular triangulation
    tri = Delaunay(nodes)
    elements = tri.simplices
    n_elements = elements.shape[0]

    # Find boundary nodes (x=0, x=1, y=0, y=1)
    tol = 1e-10
    is_boundary = (nodes[:,0] < tol) | (nodes[:,0] > 1-tol) | \
                  (nodes[:,1] < tol) | (nodes[:,1] > 1-tol)
    boundary_nodes = np.where(is_boundary)[0]

    # 2. Global Assembly Initialization
    A = sp.lil_matrix((n_nodes, n_nodes))
    F = np.zeros(n_nodes)

    # 3. Element-by-Element Assembly
    for i in range(n_elements):
        # Get node indices and coordinates for this triangle
        elem_nodes = elements[i]
        coords = nodes[elem_nodes]
        
        # Area of the triangle and gradients of shape functions
        # For a triangle, basis gradients are constant
        x0, y0 = coords[0]
        x1, y1 = coords[1]
        x2, y2 = coords[2]
        
        # Jacobian determinant (2 * area)
        detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        area = 0.5 * abs(detJ)
        
        # Gradients of the linear shape functions [d/dx, d/dy]
        b = np.array([y1 - y2, y2 - y0, y0 - y1]) / detJ
        c = np.array([x2 - x1, x0 - x2, x1 - x0]) / detJ
        
        # Local stiffness matrix (A_local_ij = area * (grad_phi_i . grad_phi_j))
        A_local = area * (np.outer(b, b) + np.outer(c, c))
        
        # Local load vector (using 1-point quadrature at the centroid)
        xc, yc = np.mean(coords, axis=0)
        F_local = (area / 3.0) * source_f(xc, yc) * np.ones(3)
        
        # Add to global matrices
        for local_i, global_i in enumerate(elem_nodes):
            F[global_i] += F_local[local_i]
            for local_j, global_j in enumerate(elem_nodes):
                A[global_i, global_j] += A_local[local_i, local_j]

    # 4. Apply Dirichlet Boundary Conditions (u = 0)
    # Zero out rows and columns for boundary nodes, set diagonal to 1
    A = A.tocsr()
    for bn in boundary_nodes:
        A.data[A.indptr[bn]:A.indptr[bn+1]] = 0 # zero out row
    A.eliminate_zeros()
    A = A + sp.diags(is_boundary.astype(float)) # set diagonal to 1
    F[boundary_nodes] = 0

    # 5. Solve the linear system
    U = spla.spsolve(A, F)

    # 6. Compute Errors (using midpoint rule over elements)
    L2_error_sq = 0.0
    H1_error_sq = 0.0
    
    for i in range(n_elements):
        elem_nodes = elements[i]
        coords = nodes[elem_nodes]
        U_local = U[elem_nodes]
        
        # Triangle geometry
        x0, y0 = coords[0]
        x1, y1 = coords[1]
        x2, y2 = coords[2]
        detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        area = 0.5 * abs(detJ)
        
        # Centroid (evaluation point for simple quadrature)
        xc, yc = np.mean(coords, axis=0)
        
        # Exact solutions at centroid
        u_ex = exact_u(xc, yc)
        grad_u_ex = np.array(exact_grad_u(xc, yc))
        
        # FEM solutions at centroid
        # u_h is the average of the nodal values at the centroid
        u_h = np.mean(U_local) 
        
        # grad_u_h is constant over the triangle
        b = np.array([y1 - y2, y2 - y0, y0 - y1]) / detJ
        c = np.array([x2 - x1, x0 - x2, x1 - x0]) / detJ
        grad_u_h = np.array([np.dot(b, U_local), np.dot(c, U_local)])
        
        # Accumulate squared errors
        L2_error_sq += area * (u_ex - u_h)**2
        diff_grad = grad_u_ex - grad_u_h
        H1_error_sq += area * np.dot(diff_grad, diff_grad)

    return np.sqrt(L2_error_sq), np.sqrt(H1_error_sq)

# --- Run the convergence study ---
N_values = [4, 8, 16, 32, 64]
L2_errors = []
H1_errors = []

for N in N_values:
    e_L2, e_H1 = solve_fem_poisson(N)
    L2_errors.append(e_L2)
    H1_errors.append(e_H1)

plt.figure(figsize=(8, 6))
plt.loglog(N_values, L2_errors, 'o-', label='$L^2$ Error', linewidth=2)
plt.loglog(N_values, H1_errors, 's-', label='Energy Norm Error', linewidth=2)

# Reference slope lines
# plt.loglog(N_values, [L2_errors[0]*(N_values[0]/N)**2 for N in N_values], 'k--', label='$\mathcal{O}(N^{-2})$ Reference')
# plt.loglog(N_values, [H1_errors[0]*(N_values[0]/N)**1 for N in N_values], 'k:', label='$\mathcal{O}(N^{-1})$ Reference')

plt.xlabel('Number of intervals $N$')
plt.ylabel('Error')
plt.title('FEM Convergence for Poisson Equation')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()