# Day 4: Decision Tree -- Loan Approval Prediction

## What is a Decision Tree?

A Decision Tree is a supervised machine learning algorithm used for
**classification** and **regression** tasks.\
It works like a **flowchart** where each internal node represents a
decision based on a feature, each branch represents the outcome of that
decision, and each leaf node represents the final prediction.

It is one of the most intuitive and interpretable machine learning
algorithms.

------------------------------------------------------------------------

## Simple Intuition

A Decision Tree works by asking a sequence of questions to reach a final
decision.

Example (Loan Approval): 1. Is income greater than ₹50,000? - Yes → Go
to next question - No → Reject loan 2. Is credit score above 700? - Yes
→ Approve loan - No → Reject loan

This process continues until the model reaches a final decision.

------------------------------------------------------------------------

## How the Algorithm Works

1.  The algorithm looks at all features in the dataset.
2.  It selects the feature that best splits the data into different
    classes.
3.  The dataset is divided into smaller subsets based on that feature.
4.  The process repeats recursively on each subset.
5.  The tree stops growing when:
    -   A stopping condition is met, or
    -   The data is perfectly classified.


    or
    ### How it Works
A Decision Tree splits the data into subsets based on the most significant attribute. It uses a "Top-Down" approach:
1.  **Root Node:** Represents the entire population.
2.  **Decision Node:** When a sub-node splits into further sub-nodes.
3.  **Leaf/Terminal Node:** Nodes that do not split; these provide the final classification (Approved/Rejected).

------------------------------------------------------------------------

## Splitting Criteria (Important Formulas)

### Gini Impurity

Gini = 1 − Σ (pᵢ)²

### Entropy

Entropy = − Σ pᵢ log₂(pᵢ)

### Information Gain

Information Gain = Entropy(parent) − Weighted average entropy(children)


To find the best split, the algorithm evaluates:
* **Entropy:** The measure of impurity in a group of examples.
    $$Entropy = -\sum p_i \log_2(p_i)$$
* **Information Gain:** The decrease in entropy after a dataset is split on an attribute.
    $$Gain(S, A) = Entropy(S) - \sum \frac{|S_v|}{|S|} Entropy(S_v)$$

------------------------------------------------------------------------

## Project Description

This project builds a **Loan Approval Prediction system** using a
Decision Tree classifier.

Output: - Approved (1) - Rejected (0)

Features: - Income - Credit score - Age

------------------------------------------------------------------------

## Steps Performed

1.  Created dataset
2.  Split into training and testing sets
3.  Trained Decision Tree model
4.  Predicted results
5.  Calculated accuracy
6.  Tested on new applicant

------------------------------------------------------------------------

## How to Run This Project

### Install dependencies

    pip install numpy pandas scikit-learn

### Run the program

    python decision_tree.py

------------------------------------------------------------------------

## Real-World Applications

-   Loan approval
-   Medical diagnosis
-   Fraud detection
-   Credit risk analysis
-   Customer segmentation

------------------------------------------------------------------------

## Advantages

-   Easy to interpret
-   Handles non-linear data
-   Requires little preprocessing
-   Works with numerical and categorical data

------------------------------------------------------------------------

## Limitations

-   Prone to overfitting
-   Sensitive to small data changes
-   Can become complex
-   Biased toward features with many levels

------------------------------------------------------------------------

## Solutions to Limitations

-   Pruning
-   Limiting tree depth
-   Minimum samples per leaf
-   Using ensemble methods (Random Forest, Boosting)

------------------------------------------------------------------------

## Tools Used

-   Python
-   NumPy
-   Scikit-learn
-   VS Code

------------------------------------------------------------------------

## Author

Likhitha J
