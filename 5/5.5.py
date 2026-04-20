import numpy as np
import matplotlib.pyplot as plt


def compute_u_hat(N):
    """
    computes exact fourier coefficients for the solution u 
    up to a spectral cutoff frequency N.
    """
    # create frequency grid from -N to N
    k = np.arange(-N, N + 1)
    kx, ky = np.meshgrid(k, k)
    
    # Initialize f_hat with zeros
    f_hat = np.zeros_like(kx, dtype=complex)
    
    # Vectorized computation for better performance
    # Case 1: kx = 0 and ky = 0
    mask_zero = (kx == 0) & (ky == 0)
    f_hat[mask_zero] = 1.0
    
    # Case 2: kx != 0 and ky = 0
    mask_kx_nonzero = (kx != 0) & (ky == 0)
    f_hat[mask_kx_nonzero] = 1j / (2 * np.pi * kx[mask_kx_nonzero])
    
    # Case 3: kx = 0 and ky != 0
    mask_ky_nonzero = (kx == 0) & (ky != 0)
    f_hat[mask_ky_nonzero] = 1j / (2 * np.pi * ky[mask_ky_nonzero])
    
    # Case 4: both non-zero remains 0.0 (already zero)
    
    # Solve the PDE in Fourier space by dividing by the penalty term
    k_squared = kx**2 + ky**2
    u_hat = f_hat / (4 * np.pi**2 * k_squared + 1.0)
    
    return kx, ky, u_hat


def main():
    # PART A: Plotting the Periodic Extension
    print("Generating Part A plots...")
    
    # cutoffs
    J_values = [2, 3, 4, 5]
x_1d = np.linspace(0, 1, spatial_resolution, endpoint=False)
x, y = np.meshgrid(x_1d, x_1d)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Part A: Periodic Extension of Spectral Solutions on [-1, 2]²", fontsize=16)

for idx, N in enumerate(N_values):
    ax = axes[idx // 2, idx % 2]
    
    # get the Fourier coeffs for this cutoff N
    kx, ky, u_hat = compute_u_hat(N)
    
    # shift frequencies for standard FFT layout
    u_hat_shifted = np.fft.ifftshift(u_hat)
    
    # pad the frequency array with zeros to match spatial grid
    # trick called "zero-padding" which perfectly interpolates the Fourier series
    pad_width = (spatial_resolution - (2*N + 1)) // 2
    u_hat_padded = np.pad(u_hat_shifted, pad_width, mode='constant')
    
    # transform back to physical space (multiply by total elements due to FFT scaling)
    u_spatial = np.real(np.fft.ifft2(u_hat_padded)) * (spatial_resolution**2)
    
    # tile the domain to create the periodic extension from [-1, 2]
    u_tiled = np.tile(u_spatial, (3, 3))
    
    # plotting
    im = ax.imshow(u_tiled, extent=[-1, 2, -1, 2], origin='lower', cmap='viridis')
    ax.set_title(f"Cutoff N = {N}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()


# PART B: Error Analysis on a Log-Log Scale
print("Computing Part B errors...")

# Our "True" solution placeholder is N = 2^8 = 256
N_star = 2**8
_, _, u_hat_star = compute_u_hat(N_star)

L2_errors = []
H1_errors = []

# Parseval's Theorem allows to calculate the spatial error directly in Fourier space!
# the error between u_star and u_N is simply the energy of the high frequencies 
# that exist in u_star but were "cut off" in u_N.
for N in N_values:
    # create a mask of the frequencies in u_star that are strictly GREATER than N
    k_star = np.arange(-N_star, N_star + 1)
    kx_star, ky_star = np.meshgrid(k_star, k_star)
    
    # the mask is True where the frequency exceeds our current cutoff N
    cutoff_mask = np.maximum(np.abs(kx_star), np.abs(ky_star)) > N
    
    # extract only the "leftover" coefficients that represent the error
    error_coeffs = u_hat_star[cutoff_mask]
    error_kx = kx_star[cutoff_mask]
    error_ky = ky_star[cutoff_mask]
    
    # calculate the L2 Error (just sum of squared coefficients)
    L2_err = np.sqrt(np.sum(np.abs(error_coeffs)**2))
    L2_errors.append(L2_err)
    
    # calculate the H1 Error (sum of squared coefficients, weighted by the H^1 penalty)
    # H1 penalty = (1 + 4*pi^2*|k|^2)
    k_squared_err = error_kx**2 + error_ky**2
    H1_weight = 1.0 + 4 * np.pi**2 * k_squared_err
    H1_err = np.sqrt(np.sum(H1_weight * np.abs(error_coeffs)**2))
    H1_errors.append(H1_err)

# Plotting the errors
plt.figure(figsize=(8, 6))
plt.loglog(N_values, L2_errors, marker='o', label='$L^2$ Error', linewidth=2)
plt.loglog(N_values, H1_errors, marker='s', label='$H^1$ Error', linewidth=2)

plt.title("Error Convergence of Spectral Galerkin Method", fontsize=14)
plt.xlabel("Spectral Cutoff (N)", fontsize=12)
plt.ylabel("Error Norm", fontsize=12)
plt.xticks(N_values, labels=[str(n) for n in N_values])
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.legend(fontsize=12)
plt.show()