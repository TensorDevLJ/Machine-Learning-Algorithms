# Day 1: Linear Regression — House Price Prediction

## 📌 Project Overview
Welcome to **Day 1** of my 30-Day Machine Learning Journey! Today, I’m exploring **Linear Regression**, the foundation of predictive modeling. This project uses a simple dataset to predict house prices based on square footage.

## 🧠 What is Linear Regression?
Linear Regression models the relationship between a **dependent variable** (target) and one or more **independent variables** (features). 

In this project, we find the **line of best fit**—the straight line that minimizes the distance between our predicted values and the actual data points.

### The Mathematics
To represent this model professionally, we use the linear equation:

$$y = \beta_0 + \beta_1x + \epsilon$$

* **$y$**: Predicted output (House Price)
* **$\beta_0$**: Intercept (The price when size is 0)
* **$\beta_1$**: Slope/Coefficient (How much price increases per sq. ft.)
* **$x$**: Input feature (House Size)
* **$\epsilon$**: Error term (The difference between prediction and reality)

---

## 🛠️ How it Works (The Logic)
1.  **Data Collection:** We provide the model with examples (Size vs. Price).
2.  **Training:** The model uses **Ordinary Least Squares (OLS)** to calculate the best $\beta_0$ and $\beta_1$ values.
3.  **Cost Function:** We minimize the **Mean Squared Error (MSE)** to ensure the line is as accurate as possible.
4.  **Prediction:** Once trained, we can input any house size to get an estimated price.

---

## 💻 Quick Code Look
```python
from sklearn.linear_model import LinearRegression

# 1. Initialize the model
model = LinearRegression()

# 2. Train the model using our data
model.fit(X_train, y_train)

# 3. Predict for a new house size (e.g., 2000 sq ft)
predicted_price = model.predict([[2000]])
```
or 

Mathematically, the model tries to fit:
```
y = mx + b
```
Where: - y = predicted output - x = input feature - m = slope of the
line - b = intercept
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
![Lines regression output ](image.png)

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
