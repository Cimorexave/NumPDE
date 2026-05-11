import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.tri as mtri

def create_l_shape_mesh(N):
    """
    Generates a uniform mesh for the L-shaped domain [-1, 1]^2 \ (0, 1]x[-1, 0).
    N is the number of subdivisions per unit length (h = 1/N).
    """
    # 2N subdivisions across the length of 2 (from -1 to 1)
    x = np.linspace(-1, 1, 2*N + 1)
    y = np.linspace(-1, 1, 2*N + 1)
    X, Y = np.meshgrid(x, y)
    
    # Mask to keep Q1, Q2, Q3 and remove strictly Q4
    # We keep the boundaries of Q4 (x=0 and y=0)
    mask = ~((X > 1e-10) & (Y < -1e-10))
    
    nodes = np.column_stack((X[mask], Y[mask]))
    
    # Map grid indices to node indices
    node_idx = -np.ones(X.shape, dtype=int)
    node_idx[mask] = np.arange(len(nodes))
    
    elements = []
    # Create triangles exactly as diagram
    # Diagonals go from bottom-left to top-right
    for i in range(2*N):
        for j in range(2*N):
            n1 = node_idx[i, j]       # bottom-left
            n2 = node_idx[i, j+1]     # bottom-right
            n3 = node_idx[i+1, j]     # top-left
            n4 = node_idx[i+1, j+1]   # top-right
            
            # If all 4 corners of the square are in the domain, add 2 triangles
            if n1 >= 0 and n2 >= 0 and n3 >= 0 and n4 >= 0:
                elements.append([n1, n2, n4]) # Bottom-right triangle
                elements.append([n1, n4, n3]) # Top-left triangle
                
    return nodes, np.array(elements)

def assemble_system(nodes, elements):
    """Assembles the Stiffness Matrix (A) and Load Vector (b)."""
    num_nodes = len(nodes)
    A_row, A_col, A_data = [], [], []
    b = np.zeros(num_nodes)
    
    for elem in elements:
        coords = nodes[elem]
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        
        # Triangle Area
        area = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
        
        # B matrix (gradients of the 3 linear basis functions)
        B = np.zeros((2, 3))
        B[0,0] = y2 - y3; B[1,0] = x3 - x2
        B[0,1] = y3 - y1; B[1,1] = x1 - x3
        B[0,2] = y1 - y2; B[1,2] = x2 - x1
        B /= (2 * area)
        
        # Local stiffness matrix and load vector
        Ke = area * (B.T @ B)
        be = np.array([area/3, area/3, area/3]) # Integral of 1 * phi_i
        
        # Add to global arrays
        for i in range(3):
            b[elem[i]] += be[i]
            for j in range(3):
                A_row.append(elem[i])
                A_col.append(elem[j])
                A_data.append(Ke[i,j])
                
    A = csr_matrix((A_data, (A_row, A_col)), shape=(num_nodes, num_nodes))
    return A, b

def apply_boundary_conditions(A, b, nodes):
    """Identifies boundary nodes and applies u=0 Dirichlet conditions."""
    x, y = nodes[:,0], nodes[:,1]
    eps = 1e-8
    
    # Boundary edges: outer square edges + re-entrant inner edges
    is_bdry = (np.abs(x - 1) < eps) | (np.abs(x + 1) < eps) | \
              (np.abs(y - 1) < eps) | (np.abs(y + 1) < eps) | \
              ((np.abs(x) < eps) & (y < eps)) | \
              ((np.abs(y) < eps) & (x > -eps))
              
    interior = np.where(~is_bdry)[0]
    return interior

def solve_fem(N):
    nodes, elements = create_l_shape_mesh(N)
    A, b = assemble_system(nodes, elements)
    interior = apply_boundary_conditions(A, b, nodes)
    
    u = np.zeros(len(nodes))
    # Solve only for interior nodes
    u[interior] = spsolve(A[interior, :][:, interior], b[interior])
    
    return nodes, elements, u, A

# --- Reference Solution ---
N_ref = 256 # Use a very fine mesh for reference solution
print(f"Computing reference solution on N={N_ref}...")
nodes_ref, elems_ref, u_ref, A_ref = solve_fem(N_ref)

# --- Compare Solutions and Errors ---
N_values = [4, 8, 16, 32, 64, 128] # Coarser meshes for error analysis
errors = []
h_values = [1/N for N in N_values]

for N in N_values:
    print(f"Solving for N={N}...")
    nodes_c, elems_c, u_c, _ = solve_fem(N)
    
    # Interpolate coarse solution u_c onto the fine reference nodes
    coarse_triang = mtri.Triangulation(nodes_c[:,0], nodes_c[:,1], elems_c)
    interpolator = mtri.LinearTriInterpolator(coarse_triang, u_c)
    u_c_fine = interpolator(nodes_ref[:,0], nodes_ref[:,1])
    
    # Handle minor float precision NaN errors near the boundary
    u_c_fine = np.nan_to_num(u_c_fine, nan=0.0)
    
    # Calculate Energy Norm Error: ||e||_E = sqrt(e^T * A_ref * e)
    error_vec = u_ref - u_c_fine
    # energy_error = np.sqrt(error_vec.T @ A_ref @ error_vec)
    # Use the .dot() method for the sparse matrix product to ensure compatibility
    Ax = A_ref.dot(error_vec)
    energy_error = np.sqrt(np.dot(error_vec, Ax))
    errors.append(energy_error)

# --- Convergence ---
plt.figure(figsize=(8, 6))
plt.loglog(h_values, errors, '-o', label='Calculated Error $\|u_{ref} - u_h\|_E$')

# Reference lines for O(h^1) and O(h^(2/3))
ref_h1 = [errors[0] * (h/h_values[0])**1 for h in h_values]
ref_h23 = [errors[0] * (h/h_values[0])**(2/3) for h in h_values]

plt.loglog(h_values, ref_h1, '--', color='gray', label='$\mathcal{O}(h^1)$ (Optimal)')
plt.loglog(h_values, ref_h23, ':', color='red', label='$\mathcal{O}(h^{2/3})$ (Expected)')

plt.xlabel('Mesh width $h$')
plt.ylabel('Energy Error')
plt.title('FEM Error Convergence on L-Shaped Domain')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()