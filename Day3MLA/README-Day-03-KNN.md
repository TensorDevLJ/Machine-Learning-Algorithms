# Day 3: K-Nearest Neighbors (KNN) -- Iris Flower Classification

## What is K-Nearest Neighbors?

K-Nearest Neighbors (KNN) is a simple and intuitive machine learning
algorithm used for classification and regression problems.\
It makes predictions based on the **closest data points** in the feature
space.

KNN is called a **lazy learning algorithm** because it does not build a
model during training. Instead, it stores the training data and makes
decisions at prediction time.

------------------------------------------------------------------------

## How the Algorithm Works (Simple Intuition)

1.  Choose a value for **K** (number of neighbors).
2.  For a new data point, calculate the distance between it and all
    points in the training data.
3.  Select the **K closest points**.
4.  Look at the classes of those neighbors.
5.  Assign the class that appears most frequently.

Example: If K = 3 and among the 3 nearest neighbors: - 2 are Setosa - 1
is Versicolor

Then the new sample is classified as **Setosa**.

------------------------------------------------------------------------

## Project Description

In this project, we built a **flower classification system** using the
famous Iris dataset.

The model predicts the type of flower based on four features: - Sepal
length - Sepal width - Petal length - Petal width

### Output classes:

-   Setosa
-   Versicolor
-   Virginica

------------------------------------------------------------------------

## Steps Performed

1.  Loaded the Iris dataset
2.  Split the data into training and testing sets
3.  Created a KNN classifier
4.  Trained the model
5.  Predicted the test data
6.  Calculated model accuracy
7.  Predicted a new flower sample

------------------------------------------------------------------------

## How to Run This Project

### Step 1: Install dependencies

    pip install numpy pandas scikit-learn

### Step 2: Run the program

    python knn_iris.py

### Step 3: Expected Output

You will see: - Model accuracy in the terminal - Predicted flower type
for a new sample

------------------------------------------------------------------------

## Real-World Applications

1.  Recommendation systems
2.  Image recognition
3.  Medical diagnosis
4.  Fraud detection
5.  Document classification

------------------------------------------------------------------------

## Advantages

-   Simple and easy to understand
-   No training phase required
-   Works well with small datasets
-   Can model complex decision boundaries

------------------------------------------------------------------------

## Limitations

-   Slow for large datasets
-   Sensitive to irrelevant features
-   Requires proper scaling of data
-   Performance depends on the choice of K

------------------------------------------------------------------------

## Tools Used

-   Python
-   NumPy
-   Scikit-learn
-   VS Code

------------------------------------------------------------------------

## Author

Likhitha J\
Machine Learning Learning Journey
