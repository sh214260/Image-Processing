from section1b import matrix_translation as existing_matrix_translation
from section1c import matrix_rotation as existing_matrix_rotation

def rotation_around_point(theta, px, py):
    # 1. הזזה כך שהנקודה תהיה בראשית
    T1 = existing_matrix_translation(-px, -py)

    # 2. סיבוב סביב הראשית
    R = existing_matrix_rotation(theta)

    # 3. החזרה למיקום המקורי
    T2 = existing_matrix_translation(px, py)

    # הרכבת הפעולות
    return T2 @ R @ T1


# דוגמה: סיבוב של 30° סביב (100, 200)
M = rotation_around_point(30, 100, 200)
print(M)
