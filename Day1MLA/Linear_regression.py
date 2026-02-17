import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create simple dataset (house size vs price)
# Size in square feet
X = np.array([500, 800, 1000, 1200, 1500, 1800]).reshape(-1, 1)

# Price in thousands
y = np.array([50, 80, 100, 120, 150, 180])

# Step 2: Create model
model = LinearRegression()

# Step 3: Train model
model.fit(X, y)

# Step 4: Predict price for a new house
new_size = np.array([[1300]])
predicted_price = model.predict(new_size)

print(f"Predicted price for 1300 sq ft house: {predicted_price[0]:.2f} thousand")

# Step 5: Plot results
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price (thousands)")
plt.title("Linear Regression: House Price Prediction")
plt.legend()
plt.show()
