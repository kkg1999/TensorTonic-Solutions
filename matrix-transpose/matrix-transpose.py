import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    row = len(A)
    col = 0
    if row>0:
        col = len(A[0])
    if row == 0 or col == 0:
        return [[]]
    ans = [[0]* row for _ in range(col)]
    #Note: cannot do [[0]*row] * col - due to same reference
    for r in range(row):
        for c in range(col):
            ans[c][r] = A[r][c]
    return np.array(ans)
