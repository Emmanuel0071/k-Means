import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def visualize_data(X, y=None, centers=None):
    plt.figure(figsize=(10, 8))
    n_features = X.shape[1]

    if n_features == 0:
        print("No features to visualize.")
        return

    elif n_features == 1:
        plt.scatter(X[:, 0], np.zeros_like(X[:, 0]), c=y, cmap='viridis', alpha=0.6)
        if centers is not None:
            plt.scatter(centers[:, 0], np.zeros_like(centers[:, 0]), c='red', marker='x', s=200, linewidths=3)
        plt.yticks([])
        plt.xlabel('Feature 1')

    elif n_features == 2:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
        if centers is not None:
            plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

    elif n_features == 3:
        ax = plt.axes(projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', alpha=0.6)
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', marker='x', s=200, linewidths=3)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')

    else:

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)

        ax = plt.axes(projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', alpha=0.6)

        if centers is not None:
            centers_scaled = scaler.transform(centers)
            centers_pca = pca.transform(centers_scaled)
            ax.scatter(centers_pca[:, 0], centers_pca[:, 1], centers_pca[:, 2], c='red', marker='x', s=200, linewidths=3)

        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_zlabel('PCA Component 3')
    if centers is not None:
        plt.title('K-Means Clusters')
    else :
        plt.title('Generated Data')
    plt.show()


def visualize_kmeans(history_errors):
    plt.subplots(figsize=(10,10))
    plt.plot(history_errors, "bo-")
    plt.xlabel('Iteration')
    plt.ylabel('SSE Error')
    plt.title('K-Means Convergence')
    plt.show()
