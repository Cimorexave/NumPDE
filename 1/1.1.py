import numpy as np
import matplotlib.pyplot as plt


def solve_poisson(N, rhs_func):
    h = 1.0 / N
    x = np.linspace(0, 1, N+1)
    
    n_internal = N - 1
    main_diag = 2 * np.ones(n_internal)
    off_diag = -1 * np.ones(n_internal - 1)
    
    A = (np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)) / (h**2)
    
    f = rhs_func(x[1:-1])
    
    u_internal = np.linalg.solve(A, f)
    
    u = np.concatenate(([0], u_internal, [0]))
    return x, u

# a) 
print("------------- a -----------\n")
def f_a(x): return 1 + x
def exact_a(x): return -1/6*x**3 - 1/2*x**2 + 2/3*x

for N in [4, 8, 16, 32]:
    x, u_num = solve_poisson(N, f_a)
    u_exact = exact_a(x)
    error = np.abs(u_exact - u_num)
    print(f"N = {N} | Error: {error}")

# b)
print("------------- b -----------\n")

def f_b(x): return 2*x**2 + 3*x - 4/3
def exact_b(x): return -1/6*x**4 - 1/2*x**3 + 2/3*x**2
errors_b = []

for N in [4, 8, 16, 32]:
    x, u_num = solve_poisson(N, f_b)
    u_exact = exact_b(x)
    error = np.max(np.abs(u_exact - u_num))
    errors_b.append(error)
    print(f"N = {N} | Max Nodal Error: {error:.2e}")

plt.plot([4, 8, 16, 32], errors_b)
plt.xlabel("N")
plt.ylabel("Error: u_exact - u_num")
plt.show()