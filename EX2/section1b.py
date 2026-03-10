import numpy as np

def matrix_translation(a, b):
    """
    מחזירה מטריצת הזזה 3x3 בקואורדינטות הומוגניות.
    ההזזה היא בשיעור (a, b).
    """
    return np.array([
        [1, 0, a],
        [0, 1, b],
        [0, 0, 1]
    ], dtype=float)
