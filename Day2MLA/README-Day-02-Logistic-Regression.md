# Day 2: Logistic Regression -- Spam Email Classifier

## What is Logistic Regression?

Logistic Regression is a machine learning algorithm used for
**classification problems**.\
Instead of predicting a continuous value like Linear Regression, it
predicts the **probability of a class**.

It is commonly used for binary classification tasks where the output can
be: - 0 or 1 - Yes or No - True or False - Spam or Not Spam

------------------------------------------------------------------------

## How the Algorithm Works (Simple Intuition)

1.  The algorithm takes input features.
2.  It calculates a weighted sum of those features.
3.  It passes the result through a **sigmoid function**.
4.  The sigmoid function converts the result into a probability between
    0 and 1.
5.  Based on a threshold (usually 0.5), it classifies the result into a
    class.

Sigmoid function:

p = 1 / (1 + e\^(-z))

Where: - p = predicted probability - z = weighted sum of inputs

------------------------------------------------------------------------

## Project Description

In this project, we built a **Spam Email Classifier** using Logistic
Regression.

The model predicts whether an email is: - Spam (1) - Not Spam (0)

### Features used:

-   Email length
-   Number of links
-   Number of special characters

The model is trained on sample data and then used to predict new emails.

------------------------------------------------------------------------

## Steps Performed

1.  Created a sample dataset
2.  Split data into training and testing sets
3.  Trained the Logistic Regression model
4.  Made predictions
5.  Calculated accuracy
6.  Tested the model on a new email

------------------------------------------------------------------------

## How to Run This Project

### Step 1: Install dependencies

    pip install numpy pandas scikit-learn

### Step 2: Run the program

    python logistic_regression.py

### Step 3: Expected Output

You will see: - Model accuracy in the terminal - Prediction for a new
email (Spam or Not Spam)

------------------------------------------------------------------------

## Real-World Applications

1.  Spam email detection
2.  Credit card fraud detection
3.  Disease prediction (positive/negative)
4.  Customer churn prediction
5.  Loan approval systems

------------------------------------------------------------------------

## Advantages

-   Simple and easy to implement
-   Works well for binary classification
-   Provides probability-based outputs
-   Fast to train and predict

------------------------------------------------------------------------

## Limitations

-   Cannot model complex non-linear relationships
-   Sensitive to outliers
-   Requires proper feature scaling for best results

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
