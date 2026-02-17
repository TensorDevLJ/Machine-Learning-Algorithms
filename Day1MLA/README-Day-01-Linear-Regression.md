# Day 1: Linear Regression -- House Price Prediction

## What is Linear Regression?

Linear Regression is one of the simplest and most widely used machine
learning algorithms.\
It is used to predict a **continuous value** based on one or more input
features.

It works by finding the **best-fit line** that describes the
relationship between input (X) and output (y).

The goal is to minimize the difference between predicted values and
actual values.

------------------------------------------------------------------------

## How the Algorithm Works (Simple Intuition)

1.  You provide input data (for example, house sizes).
2.  You provide output data (for example, house prices).
3.  The algorithm finds the best straight line that fits the data.
4.  This line is used to predict prices for new house sizes.

Mathematically, the model tries to fit:

y = mx + b

Where: - y = predicted output - x = input feature - m = slope of the
line - b = intercept

------------------------------------------------------------------------

## Steps in This Project

1.  Create a simple dataset of house sizes and prices.
2.  Train a Linear Regression model.
3.  Predict the price for a new house.
4.  Plot the regression line and actual data points.

------------------------------------------------------------------------

## How to Run This Project

### Step 1: Install required libraries

Open terminal and run:

    pip install numpy matplotlib scikit-learn

### Step 2: Run the Python file

Navigate to the project folder and run:

    python linear_regression.py

### Step 3: Output

You will see: - Predicted house price in the terminal - A graph
showing: - Blue dots: actual data - Red line: regression line
![Lines regression output picture](image.png)

------------------------------------------------------------------------

## Real-World Applications of Linear Regression

### 1. House Price Prediction

Predicting property prices based on: - Size - Location - Number of rooms

### 2. Sales Forecasting

Predicting future sales based on: - Past sales data - Marketing spend -
Seasonal trends

### 3. Stock Market Analysis

Estimating stock price trends based on historical data.

### 4. Salary Prediction

Predicting salary based on: - Years of experience - Education level -
Job role

------------------------------------------------------------------------

## Advantages

-   Simple and easy to understand
-   Fast to train
-   Works well with linear relationships

## Limitations

-   Cannot capture complex non-linear patterns
-   Sensitive to outliers
-   Assumes a linear relationship between variables

------------------------------------------------------------------------

## Tools Used

-   Python
-   NumPy
-   Matplotlib
-   Scikit-learn
-   VS Code

------------------------------------------------------------------------

## Author

Likhitha J\
Machine Learning Learning Journey -- Day 1
