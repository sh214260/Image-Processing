import numpy as np

def matrix_scale(sx, sy=None):
    """
    מחזירה מטריצת scale הומוגנית 3x3.
    אם sy לא ניתן, מבוצע scale uniform לפי sx.
    """
    if sy is None:
        sy = sx  # scale uniform

    return np.array([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1]
    ], dtype=float)
