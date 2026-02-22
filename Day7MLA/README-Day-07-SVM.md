# Day 7: Support Vector Machine (SVM) --- Breast Cancer Classification

## 📌 Overview

Support Vector Machine (SVM) is a robust supervised learning algorithm
used for classification tasks.\
This project focuses on classifying breast cancer tumors as:

-   **Malignant (0)**
-   **Benign (1)**

using the **UCI Breast Cancer Diagnostic Dataset** (available in
scikit-learn).

------------------------------------------------------------------------

## 🧠 Core Concepts

### The Hyperplane & Maximum Margin

SVM finds the **optimal hyperplane** that separates classes by
maximizing the **margin** --- the distance between the decision boundary
and the closest data points.

-   **Support Vectors**: The data points closest to the hyperplane. They
    determine the position of the boundary.
-   **Goal**: Maximize the margin to improve generalization on unseen
    data.

Mathematically, a hyperplane is defined as:

$$
w \cdot x + b = 0
$$

The optimization objective (Soft Margin SVM) is:

$$
\min \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \xi_i
$$

Subject to:

$$
y_i (w \cdot x_i + b) \geq 1 - \xi_i
$$

Where:

-   $w$ = weight vector\
-   $b$ = bias\
-   $\xi_i$ = slack variables (allow small misclassifications)\
-   $C$ = regularization parameter

The margin is:

$$
\text{Margin} = \frac{2}{\|w\|}
$$

Minimizing $\|w\|$ maximizes the margin.

------------------------------------------------------------------------

### The Kernel Trick

When data is not linearly separable in its original space, SVM uses
**kernel functions** to project it into higher dimensions.

Common kernels:

**Linear Kernel:**

$$
K(x, x') = x \cdot x'
$$

**RBF (Radial Basis Function):**

$$
K(x, x') = \exp(-\gamma \|x - x'\|^2)
$$

This allows SVM to model non-linear decision boundaries.

------------------------------------------------------------------------

## 📊 Visual Understanding

Since SVM is geometric in nature, visualizing:

-   The separating hyperplane\
-   The margin\
-   The support vectors

makes the concept much clearer.

(You can include a matplotlib plot of support vectors if using a 2D
dataset for demonstration.)

------------------------------------------------------------------------

## 🛠️ Implementation Details

**Preprocessing:**\
Scaled features using `StandardScaler` (SVM is distance-based and
sensitive to scale).

**Model:**\
Linear `SVC` with regularization parameter:

$$
C = 1.0
$$

**Evaluation Metrics:**\
- Accuracy\
- Precision\
- Recall\
- Confusion Matrix

------------------------------------------------------------------------

## 📈 Results

Model Performance:

-   **Accuracy:** 96.5%\
-   **Precision (Malignant):** 0.95\
-   **Recall (Malignant):** 0.98

### Key Insight

In breast cancer detection, **Recall for the Malignant class is the most
critical metric**, as minimizing false negatives is crucial in medical
diagnosis.

------------------------------------------------------------------------

## 💻 Code Snippet

``` python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='linear', C=1.0))
])

model.fit(X_train, y_train)
```

Full implementation available in:

`svm_classifier.py`

------------------------------------------------------------------------

## 🚀 How to Use

1.  Clone the repository\
2.  Install dependencies:

``` bash
pip install scikit-learn pandas matplotlib seaborn
```

3.  Run the script:

``` bash
python svm_classifier.py
```

------------------------------------------------------------------------

## 💡 Lessons Learned

-   **Scaling is non-negotiable:** Without normalization, features with
    larger ranges bias the hyperplane.
-   **C-Parameter Trade-off:** A smaller $C$ allows some
    misclassifications but creates a wider, more generalizable margin.
-   **Margin Maximization:** Maximizing the margin directly improves
    robustness and generalization.

------------------------------------------------------------------------

## ✅ Advantages

-   Strong theoretical foundation\
-   Effective in high-dimensional spaces\
-   Robust against overfitting with proper regularization\
-   Kernel trick enables non-linear classification

------------------------------------------------------------------------

## ⚠️ Limitations

-   Computationally expensive for large datasets\
-   Kernel selection significantly impacts performance\
-   Less interpretable than tree-based models\
-   Sensitive to hyperparameter tuning

------------------------------------------------------------------------

## 🛠️ Tools Used

-   Python\
-   scikit-learn\
-   NumPy\
-   Matplotlib\
-   Seaborn\
-   VS Code

------------------------------------------------------------------------

## ✍️ Author

Likhitha J\
Machine Learning Learning Journey
