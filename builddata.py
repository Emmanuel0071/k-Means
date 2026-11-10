from sklearn.datasets import make_blobs, make_classification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


print("Generating synthetic data...")

def generate_data(dim: int,distribution: str = 'gaussian', n_samples: int = 100):
    if distribution == 'gaussian':
        data = make_blobs(n_samples=n_samples, n_features=dim, random_state=42)
        return data[0],data[1]
    elif distribution == 'class':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=dim,
            n_informative=dim,      
            n_redundant=0,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            random_state=42)
        return X, y
    else:
        raise ValueError("Unsupported distribution")


def visualize_data(X, y=None):
    plt.figure(figsize=(8,4))

    if X.shape[1] == 0:
        print("No features to visualize.")
        return
    elif X.shape[1] == 1:
        plt.scatter(X[:, 0], np.zeros_like(X[:, 0]), c=y, cmap='viridis', alpha=0.5)
        plt.yticks([]) # Masque l'axe Y inutile
        plt.xlabel('Feature 1')

    elif X.shape[1] == 2:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.5)    
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

    elif X.shape[1] == 3:
        ax = plt.axes(projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', alpha=0.5)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')

    else: 
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)

        var = pca.explained_variance_ratio_
        print(f"Explained variance ratio by PCA components: {var}")
        print(f"Total explained variance by first 3 components: {np.sum(var)}")

        ax = plt.axes(projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', alpha=0.5)
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_zlabel('PCA Component 3')



    plt.title('Generated Data')
    plt.show()

