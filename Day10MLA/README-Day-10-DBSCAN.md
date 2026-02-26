# Day 10: DBSCAN --- Density-Based Spatial Clustering of Applications with Noise

## Overview

DBSCAN is an unsupervised machine learning algorithm used for clustering
based on data density.\
Unlike centroid-based methods like K-Means, DBSCAN groups points that
are closely packed together and marks low-density regions as outliers.

In this project, DBSCAN is applied to a non-linear synthetic dataset to
demonstrate how it detects clusters of arbitrary shapes and identifies
noise automatically.

------------------------------------------------------------------------

## Why DBSCAN?

Traditional clustering algorithms like K-Means:

-   Require predefined number of clusters (K)
-   Assume spherical cluster shapes
-   Struggle with outliers

DBSCAN solves these problems by:

-   Automatically determining number of clusters
-   Detecting arbitrary-shaped clusters
-   Identifying noise points
-   Being robust to outliers

------------------------------------------------------------------------

## Core Idea

DBSCAN defines clusters as dense regions separated by areas of lower
density.

Instead of minimizing variance, it groups points based on neighborhood
density.

------------------------------------------------------------------------

## Key Parameters

1.  eps (ε)\
    The radius that defines the neighborhood around a data point.

2.  min_samples\
    Minimum number of points required within eps radius to form a dense
    region.

These two parameters control cluster formation.

------------------------------------------------------------------------

## Types of Points

DBSCAN categorizes points into:

1.  Core Points\
    A point with at least min_samples neighbors within eps.

2.  Border Points\
    A point that is within eps of a core point but has fewer than
    min_samples neighbors.

3.  Noise Points\
    A point that is neither core nor border. Labeled as -1.

------------------------------------------------------------------------

## Algorithm Steps

1.  Pick an unvisited point.\
2.  Check if it has at least min_samples neighbors within eps.\
3.  If yes, create a cluster and expand it by including all
    density-reachable points.\
4.  If not, mark it as noise.\
5.  Repeat until all points are processed.

------------------------------------------------------------------------

## Mathematical Definition

Let N_eps(p) be the set of points within eps distance from point p.

Core Point Condition:

\|N_eps(p)\| ≥ min_samples

Density Reachability:

Point q is directly density-reachable from p if: - p is a core point - q
belongs to N_eps(p)

Clusters are formed by chaining density-reachable points together.

------------------------------------------------------------------------

## Distance Metric

DBSCAN typically uses Euclidean distance:

d(x, y) = sqrt( sum (x_i - y_i)\^2 )

Scaling is important because distance-based clustering is sensitive to
feature magnitude.

------------------------------------------------------------------------

## Project Implementation

Dataset: - Synthetic non-linear dataset using make_moons()

Pipeline:

1.  Generate dataset\
2.  Apply StandardScaler\
3.  Apply DBSCAN(eps=0.3, min_samples=5)\
4.  Extract cluster labels\
5.  Visualize clusters\
6.  Count noise points

------------------------------------------------------------------------

## Choosing eps

Choosing the right eps is critical.

Common technique:

-   Plot k-distance graph\
-   Look for elbow point\
-   That distance becomes eps

------------------------------------------------------------------------

## Time Complexity

Worst Case:

O(n\^2)

With spatial indexing (KD-tree):

Approximately O(n log n)

------------------------------------------------------------------------

## Real-World Applications

-   Geospatial clustering\
-   Anomaly detection\
-   Fraud detection\
-   Customer segmentation\
-   Image processing\
-   Network traffic analysis

------------------------------------------------------------------------

## Advantages

-   No need to specify number of clusters\
-   Detects arbitrary cluster shapes\
-   Identifies outliers automatically\
-   Works well for spatial data\
-   Robust to noise

------------------------------------------------------------------------

## Limitations

-   Sensitive to eps parameter\
-   Struggles with varying density clusters\
-   Performance decreases in high-dimensional spaces\
-   Computationally expensive for very large datasets

------------------------------------------------------------------------

## Comparison with K-Means

K-Means: - Centroid-based\
- Requires predefined K\
- Assumes spherical clusters\
- Sensitive to outliers

DBSCAN: - Density-based\
- Automatically detects clusters\
- Handles arbitrary shapes\
- Detects noise

------------------------------------------------------------------------

## Key Takeaways

-   DBSCAN clusters based on density, not centroids.\
-   eps and min_samples control clustering behavior.\
-   Automatically identifies noise.\
-   Suitable for non-linear and irregular cluster shapes.\
-   Scaling is important before applying DBSCAN.

------------------------------------------------------------------------

## Tools Used

-   Python\
-   NumPy\
-   Matplotlib\
-   Scikit-learn\
-   StandardScaler

------------------------------------------------------------------------

## Author

Likhitha J\
Machine Learning Learning Journey
