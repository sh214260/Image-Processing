import numpy as np
from scipy.signal import correlate2d


def kernel_initialize():
    """
    Returns the 3x3 kernel as a numpy array (float32)
    """
    kernel = np.array([
        [-1,  2,  1],
        [-2,  1, -3],
        [ 3,  0, -1]
    ], dtype=np.float32)
    return kernel


def image_initialize():
    """
    Returns the 4x4 image as a numpy array (float32)
    """
    image = np.array([
        [103, 102, 101, 100],
        [104, 103, 102, 101],
        [ 53,  52,  51,  50],
        [ 45,  53,  52,  51]
    ], dtype=np.float32)
    return image


def correlation_python(image, kernel):
    """
    Cross-correlation using pure Python loops.
    Output size is 2x2 because kernel fits only in those positions.
    """
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    out_h = img_h - k_h + 1
    out_w = img_w - k_w + 1

    result = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            acc = 0.0
            for ki in range(k_h):
                for kj in range(k_w):
                    acc += image[i + ki, j + kj] * kernel[ki, kj]
            result[i, j] = acc

    return result


def correlation_numpy(image, kernel):
    """
    Cross-correlation using NumPy operations only.
    """
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    out_h = img_h - k_h + 1
    out_w = img_w - k_w + 1

    result = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            patch = image[i:i+k_h, j:j+k_w]
            result[i, j] = np.sum(patch * kernel)

    return result


def correlation_scipy(image, kernel):
    """
    Cross-correlation using SciPy's correlate2d with mode='valid'
    """
    return correlate2d(image, kernel, mode='valid')


def main():
    kernel = kernel_initialize()
    image = image_initialize()

    print("Kernel:\n", kernel)
    print("Image:\n", image)

    print("\n--- Python loops ---")
    print(correlation_python(image, kernel))

    print("\n--- NumPy implementation ---")
    print(correlation_numpy(image, kernel))

    print("\n--- SciPy implementation ---")
    print(correlation_scipy(image, kernel))


if __name__ == "__main__":
    main()
