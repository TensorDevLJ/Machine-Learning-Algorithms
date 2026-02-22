import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load and Prepare Data
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Feature Scaling (Crucial for SVM)
# SVM calculates distances between points; scaling ensures all features contribute equally.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Initialize and Train Model
# Using 'C' to control regularization (higher C = less misclassification allowed)
model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Cross-Validation
# Check if the model is consistent across different subsets of data
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

# 6. Evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}%")
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Visualize Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 8. Prediction on New Sample
# Note: New data MUST be scaled using the SAME scaler instance
new_sample = X_test[0].reshape(1, -1)
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)

result = data.target_names[prediction[0]]
print(f"\nPrediction for sample: {result.capitalize()}")