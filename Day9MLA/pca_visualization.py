import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1️⃣ Load dataset
print("🚀 Loading Breast Cancer dataset...")
data = load_breast_cancer()
feature_names = data.feature_names
X, y = data.data, data.target
print(f"✅ Loaded {X.shape[0]} samples with {X.shape[1]} features.")

# 2️⃣ Scale features
print("\n⚖️  Scaling features using StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Scaling complete. Mean is now ~0 and Variance is 1.")

# 3️⃣ Apply PCA
n_comp = 10
print(f"\n🧬 Running PCA with {n_comp} components...")
pca = PCA(n_components=n_comp) 
X_pca = pca.fit_transform(X_scaled)

# 4️⃣ Terminal Analysis: Explained Variance
exp_var = pca.explained_variance_ratio_
cum_var = np.cumsum(exp_var)

print("-" * 30)
print("📊 EXPLAINED VARIANCE ANALYSIS")
for i, var in enumerate(exp_var[:3]):
    print(f"Principal Component {i+1}: explains {var*100:.2f}% variance")
print(f"TOTAL (Top 2): {cum_var[1]*100:.2f}%")
print(f"TOTAL (Top 3): {cum_var[2]*100:.2f}%")
print("-" * 30)

# 5️⃣ Terminal Analysis: Loadings (Interpretation)
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_comp)], index=feature_names)
top_pc1_feature = loadings['PC1'].abs().idxmax()
top_pc2_feature = loadings['PC2'].abs().idxmax()

print(f"\n💡 GRAPH INTERPRETATION:")
print(f"👉 PC1 is most influenced by: '{top_pc1_feature}'")
print(f"👉 PC2 is most influenced by: '{top_pc2_feature}'")
print("👉 In the scatter plots, clusters represent Malignant (0) vs Benign (1).")

# --- Visualizations ---

# Scree Plot
plt.figure(figsize=(8, 4))
plt.plot(range(1, n_comp+1), cum_var, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Variance Explained')
plt.grid(True)
plt.show()

# Loadings Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(loadings.iloc[:, :2], annot=False, cmap='coolwarm')
plt.title("Feature Loadings (PC1 vs PC2)")
plt.show()

# 3D Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', edgecolors='k')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D PCA Projection')
plt.colorbar(scatter, label='Malignant (0) vs Benign (1)')
plt.show()