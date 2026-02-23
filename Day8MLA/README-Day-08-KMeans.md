# Day 8: K-Means Clustering --- Customer Segmentation

## Overview

K-Means is an unsupervised machine learning algorithm used to group
similar data points into clusters.

Unlike supervised learning: - There are no labels - There are no
predefined outputs - The algorithm discovers patterns automatically

In this project, we applied K-Means Clustering to perform Customer
Segmentation based on:

-   Annual Income
-   Spending Score

------------------------------------------------------------------------

## Understanding Unsupervised Learning

In supervised learning: - We have inputs (X) - We have labels (Y)

In unsupervised learning: - We only have inputs (X) - The algorithm must
find structure on its own

K-Means is one of the most fundamental unsupervised algorithms.

------------------------------------------------------------------------

## What is K-Means?

K-Means is a distance-based clustering algorithm.

It divides data into K clusters by minimizing the distance between data
points and their assigned cluster centroid.

------------------------------------------------------------------------

## Step-by-Step Algorithm

1.  Choose the number of clusters (K)
2.  Randomly initialize K centroids
3.  Assign each data point to the nearest centroid
4.  Recalculate centroids as the mean of assigned points
5.  Repeat steps 3--4 until centroids stop changing

------------------------------------------------------------------------

## Mathematical Objective Function

K-Means minimizes:

J = sum \|\|x_i - mu_ci\|\|\^2

Where: - x_i = data point - mu_ci = centroid of assigned cluster

This is called Within-Cluster Sum of Squares (WCSS) or Inertia.

------------------------------------------------------------------------

## Why Feature Scaling is Important

K-Means uses Euclidean Distance.

If features are on different scales: - Larger scale dominates
clustering - Smaller scale becomes insignificant

Therefore, we use StandardScaler to normalize features.

------------------------------------------------------------------------

## Project Implementation

Dataset: - Annual Income - Spending Score

Pipeline: 1. Created dataset using pandas 2. Applied StandardScaler 3.
Trained KMeans model 4. Retrieved cluster labels 5. Visualized clusters
using matplotlib 6. Printed cluster centers

------------------------------------------------------------------------

## Choosing Optimal K --- The Elbow Method

1.  Train K-Means for different K values
2.  Plot Inertia vs K
3.  Look for the elbow point

------------------------------------------------------------------------

## Real-World Applications

-   Customer segmentation
-   Market basket analysis
-   Image compression
-   Anomaly detection
-   Recommendation systems
-   Social network analysis

------------------------------------------------------------------------

## Advantages

-   Simple and easy to implement
-   Computationally efficient
-   Scales well to large datasets
-   Fast convergence

------------------------------------------------------------------------

## Limitations

-   Must predefine K
-   Sensitive to initial centroid placement
-   Struggles with non-spherical clusters
-   Sensitive to outliers
-   Assumes equal cluster sizes

------------------------------------------------------------------------

## Time Complexity

Average Case:

O(n \* k \* d \* i)

Where: - n = number of data points - k = number of clusters - d = number
of features - i = number of iterations

------------------------------------------------------------------------

## Tools Used

-   Python
-   NumPy
-   Pandas
-   Matplotlib
-   Scikit-learn

------------------------------------------------------------------------

## Key Takeaways

-   K-Means minimizes within-cluster variance
-   Feature scaling is essential
-   Selecting correct K is critical
-   Works best for spherical clusters

------------------------------------------------------------------------

## Author

Likhitha J
