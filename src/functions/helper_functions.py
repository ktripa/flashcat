import numpy as np

def roll_mean(x, window):
    """
    Trailing rolling mean over `window` days.
    - x: 1D array-like
    - window: int (e.g., 28 -> 28-day rolling mean)
    Returns: 1D numpy array (same length), NaN for the first window-1 points.

    Example:
    x = [1,2,3,4,5,6,7,8]
    roll_mean(x, 3)  # -> [nan, nan, 2., 3., 4., 5., 6., 7.]

    """
    x = np.asarray(x, float)
    w = int(window)
    if w <= 1:
        return x.copy()

    kernel = np.ones(w) / w
    y = np.convolve(x, kernel, mode="valid")  # length = len(x) - w + 1

    out = np.full_like(x, np.nan, dtype=float)
    out[w-1:] = y
    return out

