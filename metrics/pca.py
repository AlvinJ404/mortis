import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Define the model vectors
gpt = np.array([-5, 4, -12, 0, 24, 12])
gpt_jailbroken = np.array([-8, 0, 26, -6, -12, -8])
qwen = np.array([-9, 9, -5, 2, -1, 4])

# Step 2: Stack vectors into a matrix (models x dimensions)
data = np.vstack([gpt, gpt_jailbroken, qwen])
labels = ['GPT', 'GPT-Jailbroken', 'Qwen']

# Step 3: Center the data
data_centered = data - data.mean(axis=0)

# Step 4: Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_centered)

# Step 5: Plotting
plt.figure(figsize=(8, 6))
colors = ['blue', 'orange', 'green']

# Scatter plot with annotations
for i in range(len(labels)):
    plt.scatter(pca_result[i, 0], pca_result[i, 1], color=colors[i], label=labels[i])
    plt.text(pca_result[i, 0] + 0.5, pca_result[i, 1] + 0.5, labels[i], fontsize=12)

# Set labels with explained variance
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('PCA of Model Ethics Embeddings')
plt.grid(True)
plt.axis('equal')

# Adjust plot limits
x_margin = (pca_result[:, 0].max() - pca_result[:, 0].min()) * 0.2
y_margin = (pca_result[:, 1].max() - pca_result[:, 1].min()) * 0.2
plt.xlim(pca_result[:, 0].min() - x_margin, pca_result[:, 0].max() + x_margin)
plt.ylim(pca_result[:, 1].min() - y_margin, pca_result[:, 1].max() + y_margin)

plt.legend()
plt.tight_layout()
plt.show()
