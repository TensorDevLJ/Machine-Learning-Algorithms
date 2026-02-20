# Day 5: Random Forest -- Loan Approval Prediction

# 🏦 Loan Approval Prediction: Random Forest Model

## 📖 Project Overview
This project implements a **Random Forest Classifier** to automate the credit decision-making process. By analyzing an applicant's Income, Credit Score, and Age, the model learns to categorize applications into "Approved" or "Rejected."

This project was developed as part of **Day 5** of the Machine Learning Learning Journey.

---

## 🧠 What is Random Forest?
Random Forest is an **Ensemble Learning** algorithm. Instead of relying on a single decision tree (which can be biased or "overfit"), it builds a "Forest" of many independent trees.

### How it Works (The Wisdom of the Crowd)
1.  **Bootstrap Sampling:** Each tree is trained on a random subset of the data.
2.  **Feature Randomness:** At each split, only a random selection of features is considered. This ensures the model doesn't rely solely on one feature (like Income).
3.  **Aggregation:** * **Classification:** The forest takes a "Majority Vote."
    * **Regression:** The forest calculates the average of all tree results.





Final output is the combined result of all trees.



---

## 🛠️ Project Implementation Steps
1.  **Data Creation:** Developed a synthetic dataset representing real-world loan scenarios.
2.  **Feature Scaling:** Used `StandardScaler` to ensure all features (Income vs. Age) are on a similar mathematical scale.
3.  **Model Training:** Initialized a `RandomForestClassifier` with 100 estimators (trees).
4.  **Validation:** Applied **Cross-Validation** and **OOB (Out-of-Bag) Scoring** to ensure the model generalizes well to new data.
5.  **Inference:** Built a prediction pipeline for new applicants.



---

## 📊 Technical Analysis
### Why Random Forest over a Single Decision Tree?
| Feature | Decision Tree | Random Forest |
| :--- | :--- | :--- |
| **Accuracy** | Lower (Prone to errors) | Higher (More robust) |
| **Overfitting** | High (Memorizes data) | Low (Generalizes patterns) |
| **Stability** | Sensitive to data changes | Very Stable |

### Mathematical Intuition
For a classification task with $N$ trees, the final prediction $\hat{y}$ is:
$$\hat{y} = \text{mode}\{T_1(x), T_2(x), T_3(x), \dots, T_n(x)\}$$

---
## Mathematical Intuition2

If we have N trees:

Final Prediction (Classification) = Mode(T1(x), T2(x), T3(x), ...,
Tn(x))

Final Prediction (Regression) = (1/N) × Sum(Ti(x))

Where: Ti(x) = Prediction from the i-th decision tree.


------------------------------------------------------------------------

## Why Random Forest?

A single Decision Tree can easily overfit the training data. Random
Forest solves this problem by:

-   Training multiple trees
-   Allowing each tree to learn slightly different patterns
-   Using majority voting (for classification)
-   Averaging predictions (for regression)

In simple terms:

One tree might make mistakes. Many trees voting together reduce
mistakes.

------------------------------------------------------------------------

------------------------------------------------------------------------



------------------------------------------------------------------------

## Project Description

In this project, we implemented a Random Forest Classifier to predict
loan approval.

The model determines whether a loan should be: - Approved (1) - Rejected
(0)

Features used: - Income - Credit Score - Age

We trained the model and tested it on unseen data to evaluate
performance.

------------------------------------------------------------------------

## Steps Performed

1.  Created a synthetic dataset
2.  Split dataset into training and testing sets
3.  Initialized RandomForestClassifier
4.  Trained the model
5.  Predicted results on test data
6.  Calculated model accuracy
7.  Tested prediction on a new applicant

------------------------------------------------------------------------

## How to Run This Project

### Step 1: Install Dependencies

pip install numpy pandas scikit-learn

### Step 2: Run the Program

python random_forest.py

Expected Output: - Model Accuracy - Prediction for new applicant

------------------------------------------------------------------------

## Key Hyperparameters

-   n_estimators: Number of trees in the forest
-   max_depth: Maximum depth of each tree
-   random_state: Ensures reproducibility
-   max_features: Number of features considered at each split

------------------------------------------------------------------------

## Real-World Applications

-   Fraud detection systems
-   Credit risk analysis
-   Medical diagnosis
-   Stock market prediction
-   Customer churn prediction
-   Recommendation systems

------------------------------------------------------------------------

## Advantages

-   Reduces overfitting compared to Decision Trees
-   High accuracy in many real-world problems
-   Handles large datasets efficiently
-   Works well with missing values
-   Provides feature importance

------------------------------------------------------------------------

## Limitations

-   Less interpretable compared to a single Decision Tree
-   Computationally more expensive
-   Can be slower with very large forests
-   Requires parameter tuning for optimal performance

------------------------------------------------------------------------

## Difference Between Decision Tree and Random Forest

Decision Tree: - Single model - High variance - Prone to overfitting

Random Forest: - Multiple trees - Reduced variance - Better
generalization

------------------------------------------------------------------------

## Tools Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   VS Code

------------------------------------------------------------------------

## Author

Likhitha J\
Machine Learning Learning Journey
