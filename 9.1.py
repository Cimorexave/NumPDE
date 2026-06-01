import numpy as np
import matplotlib.pyplot as plt

def simulate_universality():
    h_exponents = np.arange(2, 20)
    h_values = 2.0 ** (-h_exponents)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    distributions = ['a) uniform in [-1, 1]', 'b) exp(1) - 1', 'c) unfair Coin']

    for i, h in enumerate(h_values):
        n_steps = int(1 / h)
        
        t = np.linspace(0, 1, n_steps + 1)
        
        # Y_i
        Y_a = np.random.uniform(-1, 1, n_steps)
        Y_b = np.random.exponential(1, n_steps) - 1
        Y_c = np.random.choice([10, -1], p=[1/11, 10/11], size=n_steps)
        
        # W_kh
        sqrt_h = np.sqrt(h)
        W_a = np.concatenate(([0], np.cumsum(sqrt_h * Y_a)))
        W_b = np.concatenate(([0], np.cumsum(sqrt_h * Y_b)))
        W_c = np.concatenate(([0], np.cumsum(sqrt_h * Y_c)))

        is_final_step = (h == h_values[-1])
        alpha_val = 1.0 if is_final_step else 0.3
        line_width = 1.5 if is_final_step else 0.8
        color = 'blue' if is_final_step else 'gray'
        
        axes[0].plot(t, W_a, alpha=alpha_val, lw=line_width, color=color)
        axes[1].plot(t, W_b, alpha=alpha_val, lw=line_width, color=color)
        axes[2].plot(t, W_c, alpha=alpha_val, lw=line_width, color=color)

    for ax, title in zip(axes, distributions):
        ax.set_title(title)
        ax.set_xlabel('Time (t)')
        ax.grid(True, linestyle='--', alpha=0.6)

    axes[0].set_ylabel(r'$\tilde{W}_{kh}$')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_universality()