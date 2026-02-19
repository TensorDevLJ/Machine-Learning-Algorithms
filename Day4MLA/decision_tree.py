import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Create a synthetic dataset
data = {
    'Income': [45000, 75000, 30000, 80000, 20000, 95000, 55000, 60000, 15000, 85000],
    'Credit_Score': [600, 750, 580, 800, 500, 820, 680, 710, 450, 790],
    'Age': [25, 35, 22, 45, 19, 40, 30, 33, 20, 50],
    'Loan_Approved': [0, 1, 0, 1, 0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# 2. Split features and target
X = df[['Income', 'Credit_Score', 'Age']]
y = df['Loan_Approved']

# 3. Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Initialize and Train Decision Tree Classifier
# Using max_depth to prevent overfitting
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 5. Make Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Test on a New Applicant
# Example: Income 65k, Credit Score 720, Age 28
new_applicant = np.array([[65000, 720, 28]])
prediction = model.predict(new_applicant)
result = "Approved" if prediction[0] == 1 else "Rejected"

print(f"\nNew Applicant Data: Income=65k, Credit=720, Age=28")
print(f"Prediction: Loan {result}")

# 7. Visualize the Tree
plt.figure(figsize=(12, 8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Rejected', 'Approved'],
    filled=True
)
plt.title("Decision Tree for Loan Approval")
plt.show()
