# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# # 1️⃣ Create Synthetic Customer Dataset
# data = {
#     "Annual_Income": [15, 16, 17, 20, 23, 25, 30, 35, 40, 45,
#                       60, 62, 65, 70, 75, 80, 85, 90, 95, 100],
#     "Spending_Score": [39, 81, 6, 77, 40, 76, 6, 94, 3, 72,
#                        14, 99, 15, 98, 13, 90, 30, 88, 5, 95]
# }

# df = pd.DataFrame(data)

# # 2️⃣ Feature Scaling (VERY IMPORTANT)
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(df)

# # 3️⃣ Apply K-Means
# kmeans = KMeans(n_clusters=3, random_state=42)
# kmeans.fit(scaled_features)

# # 4️⃣ Add cluster labels to dataset
# df["Cluster"] = kmeans.labels_

# # 5️⃣ Print cluster centers (original scale)
# centers = scaler.inverse_transform(kmeans.cluster_centers_)
# print("Cluster Centers (Income, Spending):")
# print(centers)

# # 6️⃣ Visualize Clusters
# plt.figure(figsize=(8,6))

# for cluster in range(3):
#     clustered_data = df[df["Cluster"] == cluster]
#     plt.scatter(clustered_data["Annual_Income"],
#                 clustered_data["Spending_Score"],
#                 label=f"Cluster {cluster}")

# plt.scatter(centers[:, 0], centers[:, 1],
#             s=300, c='black', marker='X',
#             label='Centroids')

# plt.xlabel("Annual Income")
# plt.ylabel("Spending Score")
# plt.title("Customer Segmentation using K-Means")
# plt.legend()
# plt.show()



# // or
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1️⃣ Create Synthetic Customer Dataset
# Representing (Annual Income in $k, Spending Score 1-100)
data = {
    "Annual_Income": [15, 16, 17, 20, 23, 25, 30, 35, 40, 45,
                      60, 62, 65, 70, 75, 80, 85, 90, 95, 100],
    "Spending_Score": [39, 81, 6, 77, 40, 76, 6, 94, 3, 72,
                        14, 99, 15, 98, 13, 90, 30, 88, 5, 95]
}

df = pd.DataFrame(data)

# 2️⃣ Feature Scaling 
# K-Means uses Euclidean Distance. If Income is in thousands and Score is 1-100, 
# the distances will be skewed. Scaling puts them on the same playing field.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

# 3️⃣ Apply K-Means
# We choose K=3 for this example. random_state ensures reproducibility.
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(scaled_features)

# 4️⃣ Add cluster labels back to the original dataframe
df["Cluster"] = kmeans.labels_

# 5️⃣ Get Cluster Centers
# The centers are currently in "scaled" units. We transform them back 
# to original (Income, Score) units so we can actually read them.
centers = scaler.inverse_transform(kmeans.cluster_centers_)

print("--- Cluster Insights ---")
for i, center in enumerate(centers):
    print(f"Cluster {i}: Avg Income = ${center[0]:.2f}k, Avg Spending Score = {center[1]:.2f}")

# 6️⃣ Visualize Clusters
plt.figure(figsize=(10, 7))

# Define colors for the clusters
colors = ['#FF5733', '#33FF57', '#3357FF']

for cluster_id in range(3):
    clustered_data = df[df["Cluster"] == cluster_id]
    plt.scatter(clustered_data["Annual_Income"], 
                clustered_data["Spending_Score"], 
                s=100, 
                c=colors[cluster_id], 
                label=f'Cluster {cluster_id}',
                edgecolors='black', 
                alpha=0.7)

# Plotting the Centroids (The 'heart' of each cluster)
plt.scatter(centers[:, 0], centers[:, 1], 
            s=300, c='yellow', marker='X', 
            edgecolors='black',
            label='Centroids')

plt.xlabel("Annual Income (k$)", fontsize=12)
plt.ylabel("Spending Score (1-100)", fontsize=12)
plt.title("Customer Segments: Income vs Spending", fontsize=15)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()