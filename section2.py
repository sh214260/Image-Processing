import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Homogeneous transformation matrices
# -----------------------------

def matrix_translation(a, b):
    return np.array([
        [1, 0, a],
        [0, 1, b],
        [0, 0, 1]
    ], dtype=float)

def matrix_rotation(theta):
    rad = np.deg2rad(theta)
    return np.array([
        [np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad),  np.cos(rad), 0],
        [0,            0,           1]
    ], dtype=float)

def matrix_scale(sx, sy=None):
    if sy is None:
        sy = sx
    return np.array([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1]
    ], dtype=float)

# -----------------------------
# Function to apply a transformation to points
# -----------------------------
def apply_transform(points, M):
    # points: Nx2 array
    homog = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed = homog @ M.T
    return transformed[:, :2]

# -----------------------------
# Build a rectangle of width 2 and height 1 centered at the origin
# -----------------------------
def create_rectangle():
    # Rectangle: width 2 -> x in [-1, 1]
    #            height 1 -> y in [-0.5, 0.5]
    return np.array([
        [-1, -0.5],
        [ 1, -0.5],
        [ 1,  0.5],
        [-1,  0.5],
        [-1, -0.5]   # Close the shape
    ])

# -----------------------------
# Plotting
# -----------------------------
rect = create_rectangle()

# A. Original rectangle
rect_original = rect

# B. 30-degree rotation
M1 = matrix_rotation(30)
rect_rot30 = apply_transform(rect, M1)

# C. 45° rotation, then x-scale by 2
M2 = matrix_scale(2, 1) @ matrix_rotation(45)
rect_rot45_scale = apply_transform(rect, M2)

# D. x-scale by 2, then 45° rotation
M3 = matrix_rotation(45) @ matrix_scale(2, 1)
rect_scale_rot45 = apply_transform(rect, M3)

# -----------------------------
# Display
# -----------------------------
plt.figure(figsize=(8, 8))
plt.axis('equal')

plt.plot(rect_original[:,0], rect_original[:,1], label="Original")
plt.plot(rect_rot30[:,0], rect_rot30[:,1], label="Rotate 30°")
plt.plot(rect_rot45_scale[:,0], rect_rot45_scale[:,1], label="45° then Scale")
plt.plot(rect_scale_rot45[:,0], rect_scale_rot45[:,1], label="Scale then 45°")

plt.legend()
plt.title("Homogeneous Transformations on a Rectangle")
plt.grid(True)
plt.show()
