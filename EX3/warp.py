import numpy as np

def warp_image(image: np.ndarray,
               angle_deg: float,
               scale_x: float,
               scale_y: float) -> np.ndarray:
    if image is None:
        raise ValueError("image must not be None")

    if image.ndim not in (2, 3):
        raise ValueError("image must be 2D (grayscale) or 3D (color)")

    h, w = image.shape[:2]
    cx = w / 2.0
    cy = h / 2.0

    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    s = np.array([
        [scale_x, 0.0, 0.0],
        [0.0, scale_y, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    r = np.array([
        [cos_t, -sin_t, 0.0],
        [sin_t, cos_t, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    t_pos = np.array([
        [1.0, 0.0, cx],
        [0.0, 1.0, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    t_neg = np.array([
        [1.0, 0.0, -cx],
        [0.0, 1.0, -cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # M = T(cx, cy) * R(theta) * S(sx, sy) * T(-cx, -cy)
    m = t_pos @ r @ s @ t_neg
    m_inv = np.linalg.inv(m)

    ys = np.arange(h, dtype=np.float64) + 0.5
    xs = np.arange(w, dtype=np.float64) + 0.5
    x_out, y_out = np.meshgrid(xs, ys)

    out_coords = np.stack(
        [x_out.ravel(), y_out.ravel(), np.ones(h * w, dtype=np.float64)],
        axis=0,
    )
    in_coords = m_inv @ out_coords

    x_in = in_coords[0].reshape(h, w)
    y_in = in_coords[1].reshape(h, w)

    valid = (
        (x_in >= 0.5)
        & (x_in <= w - 0.5)
        & (y_in >= 0.5)
        & (y_in <= h - 0.5)
    )

    u = x_in - 0.5
    v = y_in - 0.5

    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    a = u - x0
    b = v - y0

    single_channel = image.ndim == 2
    src = image[..., None] if single_channel else image
    src_f = src.astype(np.float32)

    i00 = src_f[y0, x0]
    i01 = src_f[y0, x1]
    i10 = src_f[y1, x0]
    i11 = src_f[y1, x1]

    wa = ((1.0 - a) * (1.0 - b))[..., None]
    wb = (a * (1.0 - b))[..., None]
    wc = ((1.0 - a) * b)[..., None]
    wd = (a * b)[..., None]

    warped_f = wa * i00 + wb * i01 + wc * i10 + wd * i11

    out_f = np.zeros_like(src_f)
    out_f[valid] = warped_f[valid]

    if np.issubdtype(image.dtype, np.integer):
        out = np.clip(out_f, 0, 255).astype(image.dtype)
    else:
        out = out_f.astype(image.dtype)

    return out[..., 0] if single_channel else out