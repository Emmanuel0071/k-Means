from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

print("Generating synthetic data...")

def generate_data(dim: int,distribution: str = 'gaussian', n_samples: int = 100):
    if distribution == 'gaussian':
        data = make_blobs(n_samples=n_samples, n_features=dim, random_state=42)
        return data[0],data[1]
    elif distribution == 'uniform':
        return np.random.uniform(low=-3, high=3, size=(n_samples, dim))
    else:
        raise ValueError("Unsupported distribution")


def visualize_data(X, y=None):
    plt.figure(figsize=(8,4))

    if X.shape[1] == 1:
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


    plt.title('Generated Data')
    plt.show()

X, y = generate_data(dim=3, distribution='gaussian', n_samples=100)
visualize_data(X, y)