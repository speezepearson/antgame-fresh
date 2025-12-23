import numpy as np


def perlin(w: int, h: int) -> np.ndarray:
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    # permutation table
    rng = np.random.default_rng()
    p = np.arange(256, dtype=int)
    rng.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)
    # internal coordinates
    xf = x - xi
    yf = y - yi
    # fade factors
    u = fade(xf)
    v = fade(yf)
    # noise components
    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
    return lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here


def lerp(
    a: np.typing.NDArray[np.float64],
    b: np.typing.NDArray[np.float64],
    x: np.typing.NDArray[np.float64],
) -> np.typing.NDArray[np.float64]:
    """linear interpolation"""
    return a + x * (b - a)


def fade(t: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
    """6t^5 - 15t^4 + 10t^3"""
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def gradient(h: int, x: float, y: float) -> np.typing.NDArray[np.float64]:
    """grad converts h to the right gradient vector and return the dot product with (x,y)"""
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.float64)
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y  # type: ignore
