def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    # Write code here
    x = x0
    
    # Iterate exactly 'steps' times
    for _ in range(steps):
        # 1. Calculate the gradient: f'(x) = 2ax + b
        gradient = 2 * a * x + b
        
        # 2. Update x by moving in the opposite direction of the gradient
        x = x - lr * gradient
        
    return x