from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt

def animate_complex_matrix(matrix, x, ax=None, interval=50):
    # If no axes are provided, create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Plot lines for real, imaginary, and modulus parts
    real_line, = ax.plot([], [], lw=2, label="Real Part", color="blue")
    imag_line, = ax.plot([], [], lw=2, label="Imaginary Part", color="orange")
    modulus_line, = ax.plot([], [], lw=2, label="Modulus", color="black")

    # Set x-axis limits based on x values
    ax.set_xlim(min(x), max(x))

    # Calculate y-axis limits based on matrix values
    all_values = np.array(matrix).flatten()
    y_min = min(np.min(all_values.real), np.min(all_values.imag))
    y_max = max(np.max(all_values.real), np.max(all_values.imag))
    ax.set_ylim(y_min, y_max)
    ax.legend()

    def update(frame):
        # Calculate real, imaginary, and modulus for the current frame
        y_real = np.real(matrix[frame])
        y_imag = np.imag(matrix[frame])
        y_modulus = np.abs(matrix[frame])

        # Update lines
        real_line.set_data(x, y_real)
        imag_line.set_data(x, y_imag)
        modulus_line.set_data(x, y_modulus)

        return real_line, imag_line, modulus_line

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(matrix), blit=False, interval=interval)
    
    return ani  # Return the animation object for further customization
