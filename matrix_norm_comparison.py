import numpy as np
from numpy.linalg import norm, svd

# Define the original matrix A
A = np.array([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9]], dtype=float)

print("Original Matrix A:")
print(A)
print()

# Perform Singular Value Decomposition
U, S, Vt = svd(A, full_matrices=True)

print("Singular Values:")
print(S)
print()

# Compute rank-k approximation using SVD
# For this example, we'll use k = len(S) to keep all significant components
# This reconstructs the original matrix (or very close to it)
k = len(S)  # Using full rank

# Reconstruct A_k from SVD components
A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

print(f"Rank-{k} Approximation A_k:")
print(A_k)
print()

# ============================================
# Compute Frobenius Norm
# ============================================
frobenius_norm_A = norm(A, 'fro')
frobenius_norm_A_k = norm(A_k, 'fro')

print("=" * 60)
print("FROBENIUS NORM")
print("=" * 60)
print(f"Frobenius norm of A: {frobenius_norm_A}")
print(f"Frobenius norm of A_k: {frobenius_norm_A_k}")
print()

# ============================================
# Compute Spectral Norm (2-norm)
# ============================================
spectral_norm_A = norm(A, 2)
spectral_norm_A_k = norm(A_k, 2)

print("=" * 60)
print("SPECTRAL NORM (2-norm)")
print("=" * 60)
print(f"Spectral norm of A: {spectral_norm_A}")
print(f"Spectral norm of A_k: {spectral_norm_A_k}")
print()

# ============================================
# Additional Analysis
# ============================================
print("=" * 60)
print("ADDITIONAL ANALYSIS")
print("=" * 60)
print(f"Difference in Frobenius norm: {abs(frobenius_norm_A - frobenius_norm_A_k)}")
print(f"Difference in Spectral norm: {abs(spectral_norm_A - spectral_norm_A_k)}")
print()

# Verify using alternative methods
print("=" * 60)
print("VERIFICATION (Alternative Methods)")
print("=" * 60)

# Frobenius norm = sqrt(sum of all squared elements)
frobenius_alt = np.sqrt(np.sum(A**2))
print(f"Frobenius norm (alternative): {frobenius_alt}")

# Spectral norm = largest singular value
spectral_alt = S[0]
print(f"Spectral norm (alternative - largest singular value): {spectral_alt}")
