import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

df = pd.read_csv(r"D:\Saksham\Desktop\Gen Ai\Data FIles(Practice)\Mall_Customers.csv")

df.rename(columns={'Genre': 'Gender'}, inplace=True)
print("Columns:", df.columns.tolist())

X = df.drop(columns=['CustomerID', 'Gender'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Z = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 6))
dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
plt.axhline(y=7.0, linestyle='--')
plt.title('Dendrogram')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.tight_layout()
plt.show()

k = 5
agg = AgglomerativeClustering(n_clusters=k, linkage='ward', metric='euclidean')
labels = agg.fit_predict(X_scaled)

print('Silhouette Score:', round(silhouette_score(X_scaled, labels), 3))

plt.figure(figsize=(7, 5))
for cl in range(k):
    pts = X_scaled[labels == cl]
    plt.scatter(
        pts[:, 0], pts[:, 1], s=60,
        label=f'Cluster {cl}', edgecolor='white'
    )

plt.title(f'Agglomerative Clustering (Ward, k={k})')
plt.legend()
plt.tight_layout()
plt.show()
