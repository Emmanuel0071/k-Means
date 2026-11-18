import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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


def visualize_kmeans(history_errors):
    plt.subplots(figsize=(10,10))
    plt.plot(history_errors, "bo-")
    plt.xlabel('Iteration')
    plt.ylabel('SSE Error')
    plt.title('K-Means Convergence')
    plt.show()

def visualize_clusters(X, y = None, centers=None):
    plt.figure(figsize=(10,10))
    plt.scatter(X[:,0],X[:,1],c=y)
    if centers is not None:
        plt.scatter(centers[:,0], centers[:,1], c='red', marker='x', s=200)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()