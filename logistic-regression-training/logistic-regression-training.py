import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    X = np.array(X)
    y = np.array(y)
    N, D = X.shape

    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):
        z = np.dot(X, w) + b
        p = _sigmoid(z)

        error = p - y
        grad_w = np.dot(X.T, error)/N
        grad_b = np.mean(error)

        w -= lr * grad_w
        b -= lr * grad_b
    return w, b