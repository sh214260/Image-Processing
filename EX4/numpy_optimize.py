import numpy as np
import time


def warp_image_numpy(image, M_inv):
    """Vectorized inverse-warp with nearest-neighbor interpolation.

    Pixel center convention:
        destination pixel (i, j) is at (x=j+0.5, y=i+0.5)
    """
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2D (H, W) or 3D (H, W, C)")

    if M_inv.shape != (3, 3):
        raise ValueError("M_inv must have shape (3, 3)")

    H, W = image.shape[:2]

    # Step 0: create grid
    rows, cols = np.indices((H, W))
    x_dst = cols + 0.5
    y_dst = rows + 0.5

    # Step 1: build homogeneous coords
    ones = np.ones_like(x_dst)
    coords = np.stack([x_dst, y_dst, ones], axis=0)  # (3, H, W)
    coords_flat = coords.reshape(3, -1)  # (3, H*W)

    # Step 2: apply transform
    src = M_inv @ coords_flat

    # Step 3: reshape back
    x_src = src[0].reshape(H, W)
    y_src = src[1].reshape(H, W)

    # Step 4: nearest neighbor
    r_nn = np.round(y_src - 0.5).astype(int)
    c_nn = np.round(x_src - 0.5).astype(int)

    # Step 5: boundary mask
    valid = (
        (r_nn >= 0)
        & (r_nn < H)
        & (c_nn >= 0)
        & (c_nn < W)
    )

    # Step 6: build output
    output = np.zeros_like(image)
    output[valid] = image[r_nn[valid], c_nn[valid]]

    return output


def warp_image_bilinear(image, M_inv):
    """Vectorized inverse-warp with bilinear interpolation.

    Pixel center convention:
        destination pixel (i, j) is at (x=j+0.5, y=i+0.5)
    """
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2D (H, W) or 3D (H, W, C)")

    if M_inv.shape != (3, 3):
        raise ValueError("M_inv must have shape (3, 3)")

    H, W = image.shape[:2]

    # Step 0: create grid
    rows, cols = np.indices((H, W))
    x_dst = cols + 0.5
    y_dst = rows + 0.5

    # Step 1: build homogeneous coords
    ones = np.ones_like(x_dst)
    coords = np.stack([x_dst, y_dst, ones], axis=0)  # (3, H, W)
    coords_flat = coords.reshape(3, -1)  # (3, H*W)

    # Step 2: apply transform
    src = M_inv @ coords_flat

    # Step 3: reshape back
    x_src = src[0].reshape(H, W)
    y_src = src[1].reshape(H, W)

    # Continuous source coordinates in array-index space
    u = x_src - 0.5
    v = y_src - 0.5

    # Step 4: bilinear neighbors
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    # Step 5: boundary mask and clipped neighbor indices
    valid = (
        (u >= 0)
        & (u <= (W - 1))
        & (v >= 0)
        & (v <= (H - 1))
    )

    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    # Step 6: bilinear weights
    dx = u - x0
    dy = v - y0
    w00 = (1.0 - dx) * (1.0 - dy)
    w01 = dx * (1.0 - dy)
    w10 = (1.0 - dx) * dy
    w11 = dx * dy

    # Sample four neighbors with advanced indexing
    image_f = image.astype(np.float64, copy=False)
    I00 = image_f[y0c, x0c]
    I01 = image_f[y0c, x1c]
    I10 = image_f[y1c, x0c]
    I11 = image_f[y1c, x1c]

    # Combine via broadcasting for grayscale and color
    if image.ndim == 2:
        interp = w00 * I00 + w01 * I01 + w10 * I10 + w11 * I11
    else:
        interp = (
            w00[..., None] * I00
            + w01[..., None] * I01
            + w10[..., None] * I10
            + w11[..., None] * I11
        )

    output = np.zeros((H, W) if image.ndim == 2 else image.shape, dtype=np.float64)
    output[valid] = interp[valid]
    return output


def warp_image_loop(image, M_inv):
    """Naive inverse-warp with nearest-neighbor interpolation using Python loops."""
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2D (H, W) or 3D (H, W, C)")

    if M_inv.shape != (3, 3):
        raise ValueError("M_inv must have shape (3, 3)")

    H, W = image.shape[:2]
    output = np.zeros_like(image)

    for i in range(H):
        for j in range(W):
            x_dst = j + 0.5
            y_dst = i + 0.5

            x_src, y_src, _ = M_inv @ np.array([x_dst, y_dst, 1.0])

            r_nn = int(np.round(y_src - 0.5))
            c_nn = int(np.round(x_src - 0.5))

            if 0 <= r_nn < H and 0 <= c_nn < W:
                output[i, j] = image[r_nn, c_nn]

    return output


def benchmark_warp_performance():
    """Benchmark loop-based warp vs fully vectorized NumPy warp."""
    sizes = [(100, 100), (300, 300), (800, 800), (1500, 1500)]
    M_inv = np.array([[1, 0, -2], [0, 1, -2], [0, 0, 1]], dtype=float)

    # Fewer repeats for very large images keeps runtime practical.
    repeats = {
        (100, 100): 10,
        (300, 300): 5,
        (800, 800): 2,
        (1500, 1500): 1,
    }

    print("Warp Performance Benchmark (nearest-neighbor)")
    print(f"{'Size':>12} | {'Loop (s)':>10} | {'NumPy (s)':>10} | {'Speedup':>9}")
    print("-" * 52)

    rng = np.random.default_rng(42)

    for H, W in sizes:
        image = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
        n_runs = repeats.get((H, W), 1)

        # Warm-up run (helps avoid one-time overhead skew).
        _ = warp_image_loop(image, M_inv)
        _ = warp_image_numpy(image, M_inv)

        t0 = time.perf_counter()
        for _ in range(n_runs):
            out_loop = warp_image_loop(image, M_inv)
        loop_time = (time.perf_counter() - t0) / n_runs

        t0 = time.perf_counter()
        for _ in range(n_runs):
            out_numpy = warp_image_numpy(image, M_inv)
        numpy_time = (time.perf_counter() - t0) / n_runs

        if out_loop.shape != image.shape or out_numpy.shape != image.shape:
            raise RuntimeError("Output shape mismatch detected during benchmark")

        speedup = loop_time / numpy_time if numpy_time > 0 else float("inf")
        print(f"{H}x{W: <6} | {loop_time:10.6f} | {numpy_time:10.6f} | {speedup:9.2f}x")


if __name__ == "__main__":
    benchmark_warp_performance()