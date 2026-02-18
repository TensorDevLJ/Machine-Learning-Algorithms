from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Step 4: Train model
model.fit(X_train, y_train)

# Step 5: Predict
predictions = model.predict(X_test)

# Step 6: Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 7: Predict a new sample
new_flower = [[5.1, 3.5, 1.4, 0.2]]
result = model.predict(new_flower)

print("Predicted flower class:", iris.target_names[result][0])
