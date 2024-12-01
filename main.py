import numpy as np
import scipy
from scipy.sparse import diags
from scipy.integrate import solve_ivp
from animate import animate_complex_matrix
import matplotlib.pyplot as plt

def gaussian(x, mu, sig):
    return 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2) + 0j

def quadratic(x):
    return 10* x**2

def rectangular_pulse(x, w, x0, A):
	return A*np.heaviside(x-x0, 1) - A*np.heaviside(x-(x0+w), 1)

def sin(x,A,f):
    return A*np.sin(2*np.pi/f*x)

# Parameters
h = 1
m = 1
dx = 1 / 100
dt = 1 / 100
t0 = 0.0
tf = 1.0
t_eval = np.arange(t0, tf, dt)
p0=10

x = np.arange(-10, 25, dx)
psi = np.exp(1j*p0/h * x)*gaussian(x, 0, 1)
V = rectangular_pulse(x, 10, 4, 40)#rectangular_pulse(x,1,-8,1000)+rectangular_pulse(x,1,8,1000) #quadratic(x)

# Define the Laplacian as a sparse matrix for efficiency
laplacian = diags([1, -2, 1], [-1, 0, 1], shape=(x.size, x.size)) / dx**2

def dpsidt(t, psi):
    """Compute the time derivative of the wavefunction."""
    return 1j * (h / (2 * m) * laplacian.dot(psi) - V * psi / h)

# Solve the Schrodinger equation over time
sol = solve_ivp(dpsidt, t_span=[t0, tf], y0=psi, t_eval=t_eval, method="RK23")

# Reshape solution for easy animation
psi_t = sol.y.T  # Each row is the wavefunction at a specific time

# Create a figure and axis, then add custom features
fig, ax = plt.subplots()
ax.plot(x, 1/100* V, label="Potential V", color="green")

# Now call animate_complex_matrix and pass the existing axis
ani = animate_complex_matrix(psi_t, x, ax=ax)

# Display the plot and animation
#plt.show()

ani.save("test.gif")