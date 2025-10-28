import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a loss function (example: a nonlinear surface)
def loss_function(x, y):
    return np.sin(x) * np.cos(y) + 0.1 * (x**2 + y**2)

# Create meshgrid for surface
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)
Z = loss_function(X, Y)

# Simulated SGD path
np.random.seed(0)
path_x = np.linspace(2.5, -2, 50)
path_y = np.sin(path_x) + np.random.normal(scale=0.2, size=path_x.shape)
path_z = loss_function(path_x, path_y)

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Surface
surf = ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.8, edgecolor='k', linewidth=0.3)

# Contours on bottom plane
ax.contour(X, Y, Z, 20, cmap='jet', linestyles="solid", offset=np.min(Z)-2)

# SGD path (black line)
ax.plot(path_x, path_y, path_z, color="black", linewidth=2)

# Start and end points
ax.scatter(path_x[0], path_y[0], path_z[0], color="black", s=50)  # start
ax.scatter(path_x[-1], path_y[-1], path_z[-1], color="black", s=50)  # end

# Labels
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Loss")

plt.show()
