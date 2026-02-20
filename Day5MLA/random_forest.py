import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1.  Dataset (More samples for better learning)
data = {
    'Income': [45000, 75000, 30000, 80000, 20000, 95000, 55000, 60000, 15000, 85000, 
               40000, 120000, 32000, 70000, 25000, 110000, 48000, 62000, 18000, 90000],
    'Credit_Score': [600, 750, 580, 800, 500, 820, 680, 710, 450, 790, 
                     610, 850, 590, 720, 510, 830, 660, 700, 460, 810],
    'Age': [25, 35, 22, 45, 19, 40, 30, 33, 20, 50, 
            26, 42, 23, 38, 21, 44, 29, 34, 19, 48],
    'Loan_Approved': [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 
                      0, 1, 0, 1, 0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# 2. Split Features and Target
X = df[['Income', 'Credit_Score', 'Age']]
y = df['Loan_Approved']

# 3. Scaling (Standardization)
# This ensures 'Income' (high values) doesn't drown out 'Age' (low values)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# 5. Initialize & Train Random Forest
# Added 'n_jobs=-1' to use all CPU cores and 'oob_score' for internal validation
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
cv_scores = cross_val_score(model, X_scaled, y, cv=3)

print(f"--- Model Performance ---")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Cross-Validation Mean: {np.mean(cv_scores) * 100:.2f}%")
print(f"Out-of-Bag Score: {model.oob_score_ * 100:.2f}%")
print("\nDetailed Report:\n", classification_report(y_test, y_pred))

# 7. Feature Importance Visualization
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(8, 4))
plt.title('Which factors matter most for Loan Approval?')
plt.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# 8. Test New Applicant (Applying the same scaler!)
new_data = np.array([[65000, 720, 28]])
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
probability = model.predict_proba(new_data_scaled)[0][1]

result = "Approved" if prediction[0] == 1 else "Rejected"
print(f"\n--- Prediction for New Applicant ---")
print(f"Status: {result}")
print(f"Approval Probability: {probability * 100:.1f}%")