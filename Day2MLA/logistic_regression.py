import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Create simple dataset
# Features: [email_length, number_of_links, special_characters]
X = np.array([
    [50, 1, 0],
    [200, 5, 3],
    [45, 0, 0],
    [300, 8, 5],
    [60, 1, 1],
    [250, 6, 4],
    [40, 0, 0],
    [270, 7, 3]
])

# Labels: 0 = not spam, 1 = spam
y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 3: Create model
model = LogisticRegression()

# Step 4: Train model
model.fit(X_train, y_train)

# Step 5: Predict
predictions = model.predict(X_test)

# Step 6: Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 7: Predict new email
new_email = np.array([[220, 4, 2]])
result = model.predict(new_email)

if result[0] == 1:
    print("Prediction: Spam Email")
else:
    print("Prediction: Not Spam")
