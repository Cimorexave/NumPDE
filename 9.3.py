import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def simulate_fractal_bm():
    h = 2**(-12)
    n_steps = int(1 / h)
    
    t_grid = np.linspace(0, 1, n_steps + 1)
    
    increments = np.random.normal(0, np.sqrt(h), n_steps)
    W = np.concatenate(([0], np.cumsum(increments)))
    
    c = 1.1
    j_values = np.arange(0, 21)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], lw=1.5, color='blue')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-3, 3) 
    ax.set_title("scale invariance: fractial zoom of BM")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$c^{j/2} W_{c^{-j}t}$")
    ax.grid(True, linestyle='--', alpha=0.6)

    text_j = ax.text(0.05, 0.95, '', transform=ax.transAxes, 
                     fontsize=12, verticalalignment='top', 
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def init():
        line.set_data([], [])
        text_j.set_text('')
        return line, text_j

    def animate(j):
        # scaled time indices: t * c^(-j)
        scaled_time = t_grid * (c ** (-j))
        
        # interpolation
        W_scaled_time = np.interp(scaled_time, t_grid, W)
        
        # scale the Y values by c^(j/2)
        W_fractal = (c ** (j / 2)) * W_scaled_time
        
        line.set_data(t_grid, W_fractal)
        text_j.set_text(f'Zoom Level: j = {j}\nTime Window: [0, {c**(-j):.3f}]')
        
        max_val = np.max(np.abs(W_fractal)) * 1.2
        if max_val > 0.1: 
             ax.set_ylim(-max_val, max_val)
             
        return line, text_j

    ani = animation.FuncAnimation(fig, animate, frames=j_values, 
                                  init_func=init, blit=False, interval=1000, repeat_delay=1000)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_fractal_bm()