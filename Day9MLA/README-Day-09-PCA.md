# Day 9: Principal Component Analysis (PCA) --- Dimensionality Reduction

## Overview

Principal Component Analysis (PCA) is an unsupervised dimensionality
reduction technique used to reduce the number of features in a dataset
while preserving as much variance as possible.

In this project, PCA is applied to the Breast Cancer dataset (30
features) and reduced to 2 principal components for visualization and
analysis.

------------------------------------------------------------------------

## Why Dimensionality Reduction?

High-dimensional datasets often:

-   Increase computational cost
-   Contain redundant features
-   Cause multicollinearity
-   Make visualization difficult

PCA solves this by transforming original features into a smaller set of
new variables called Principal Components.

------------------------------------------------------------------------

## What is PCA?

PCA transforms data into a new coordinate system such that:

-   The first principal component captures the maximum variance.
-   The second principal component captures the next highest variance
    (orthogonal to the first).
-   And so on.

The components are linear combinations of original features.

------------------------------------------------------------------------

## Mathematical Foundation

Step 1: Standardization

Z = (X - mean) / standard_deviation

Step 2: Covariance Matrix

Cov(X) = (1 / (n - 1)) \* X\^T X

Step 3: Eigen Decomposition

Cov(X)v = lambda v

Where:

-   v = Eigenvector (Principal Component direction)
-   lambda = Eigenvalue (Variance captured)

Principal Components = Eigenvectors\
Explained Variance = Eigenvalues

------------------------------------------------------------------------

## Optimization Objective

PCA finds direction w that maximizes variance:

Maximize Var(w\^T X)

Subject to:

\|\|w\|\| = 1

------------------------------------------------------------------------

## Explained Variance Ratio

Explained Variance Ratio tells us how much variance each principal
component captures.

Example:

PC1: 44%\
PC2: 19%

------------------------------------------------------------------------

## Implementation Steps

1.  Load dataset
2.  Standardize features
3.  Apply PCA
4.  Transform data
5.  Visualize in 2D
6.  Analyze explained variance

------------------------------------------------------------------------

## Code Snippet

``` python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)
```

------------------------------------------------------------------------

## Real-World Applications

-   Image compression
-   Face recognition
-   Noise reduction
-   Feature extraction
-   Finance risk modeling
-   Genomics

------------------------------------------------------------------------

## Advantages

-   Reduces dimensionality efficiently
-   Removes multicollinearity
-   Speeds up ML algorithms
-   Helps visualization
-   Improves generalization

------------------------------------------------------------------------

## Limitations

-   Loses interpretability of original features
-   Assumes linear relationships
-   Sensitive to scaling
-   May discard useful information

------------------------------------------------------------------------

## Time Complexity

O(min(n\^2d, nd\^2))

Where: - n = number of samples - d = number of features

------------------------------------------------------------------------

## Key Takeaways

-   PCA is variance-maximization based
-   Principal Components are orthogonal
-   Eigenvectors define directions
-   Eigenvalues define importance
-   Scaling is mandatory before applying PCA

------------------------------------------------------------------------

## Tools Used

-   Python
-   NumPy
-   Scikit-learn
-   Matplotlib

------------------------------------------------------------------------

## Author

Likhitha J
