import numpy as np
import matplotlib.pyplot as plt

#סעיף א
def degrees_to_radians(deg):
    return deg * np.pi / 180
#סעיף ב
degrees_list = [1, 5, 10, 30, 45, 180, 90, 0]

for deg in degrees_list:
    rad = np.deg2rad(deg)        # המרה לרדיאנים
    s = np.sin(rad)              # סינוס
    c = np.cos(rad)              # קוסינוס
    
    print(f"{c:.6f},{s:.6f},{rad:.6f},{deg}")
