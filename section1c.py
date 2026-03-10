import numpy as np

def matrix_rotation(theta):
    """
    מחזירה מטריצת סיבוב 3x3 בקואורדינטות הומוגניות.
    theta ניתנת במעלות.
    """
    # המרה מרדיונים למעלות
    rad = np.deg2rad(theta)
    
    return np.array([
        [np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad),  np.cos(rad), 0],
        [0,            0,           1]
    ], dtype=float)
