import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# א. יצירת מטריצת סיבוב ב־30 מעלות
# -----------------------------
theta = np.deg2rad(30)
R_30 = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])
print("30_r =\n", R_30)

# -----------------------------
# ב. יצירת מטריצת scale פי 2 בציר x
# -----------------------------
Sx_2 = np.array([
    [2, 0],
    [0, 1]
])
print("2_sx =\n", Sx_2)

# -----------------------------
# ג. חישוב rs = 2_sx @ 30_r
# -----------------------------
rs = Sx_2 @ R_30
print("rs =\n", rs)

# -----------------------------
# ד. חישוב sr = 30_r @ 2_sx
# -----------------------------
sr = R_30 @ Sx_2
print("sr =\n", sr)

# -----------------------------
# ה. ציור מלבן ברוחב 2 וגובה 1 שמרכזו בראשית
# -----------------------------
# נקודות המלבן (פוליגון סגור)
rect = np.array([
    [-1, -0.5],
    [ 1, -0.5],
    [ 1,  0.5],
    [-1,  0.5],
    [-1, -0.5]   # חזרה לנקודה הראשונה לסגירה
]).T  # טרנספוזה כדי לקבל 2xN

# -----------------------------
# ו. סיבוב המלבן ב־30 מעלות
# -----------------------------
rect_rot = R_30 @ rect

# -----------------------------
# ז. מתיחה פי 2 בציר x
# -----------------------------
rect_scale = Sx_2 @ rect

# -----------------------------
# ח. הפעלת sr ו־rs על המלבן
# -----------------------------
rect_sr = sr @ rect
rect_rs = rs @ rect

# -----------------------------
# ט. ציור כל 5 המלבנים
# -----------------------------
plt.figure(figsize=(8, 8))
plt.axis('equal')

# המקורי
plt.plot(rect[0], rect[1], label="Original")

# סיבוב
plt.plot(rect_rot[0], rect_rot[1], label="Rotated 30°")

# מתיחה
plt.plot(rect_scale[0], rect_scale[1], label="Scaled x2 (x-axis)")

# sr
plt.plot(rect_sr[0], rect_sr[1], label="sr = R @ Sx")

# rs
plt.plot(rect_rs[0], rect_rs[1], label="rs = Sx @ R")

plt.legend()
plt.grid(True)
plt.title("Rectangle Transformations")
plt.show()
