import numpy as np
import random

def initialize_centers(X, k):

    indices = random.sample(range(len(X)), k)
    return np.array(X[indices], dtype=float)

def initialize_centers_plus_plus(X,k):
    n_simples = len(X)
    centers = [X[random.randint(0, n_simples - 1)].astype(float)]

    for i in range(1,k):
        dist_sq = np.array([min(sse_distance(x,c) for c in centers)for x in X])
        probs = dist_sq / np.sum(dist_sq)

        next_idx = np.random.choice(n_simples,p=probs)
        centers.append(X[next_idx].astype(float))
    return np.array(centers)


def sse_distance(point, center):
    return np.sum((point - center) ** 2)

def find_closest_center(point, centers):
    index_closest = 0
    min_sse = float('inf')

    for i in range(len(centers)):
        sse = sse_distance(point, centers[i])
        if sse < min_sse:
            min_sse = sse
            index_closest = i
    return index_closest

def compute_clusters(X, centers):
    cluster = []
    for i in range(len(X)):  
        index_closest = find_closest_center(X[i], centers)
        cluster.append(index_closest)
    return cluster

def sse_error(X, centers, cluster):
    error = 0 
    for i in range(len(X)):
        error += sse_distance(X[i], centers[cluster[i]])
    return error

def kmeans(X, k, max_iter=None , version = "classic"):
    if version == "classic":
        centers = initialize_centers(X, k)
    elif version == "++":
        centers = initialize_centers_plus_plus(X,k)

    history_centers = [centers.copy()]  
    history_error = []
    i = 0

    while True:
        cluster = np.array(compute_clusters(X, centers))

        for j in range(k):
            points_in_cluster = X[cluster == j]
            if len(points_in_cluster) > 0:
                centers[j] = np.mean(points_in_cluster, axis=0)

        cluster = np.array(compute_clusters(X, centers))
        
        history_centers.append(centers.copy())
        history_error.append(sse_error(X, centers, cluster)) 
        i += 1

        converged = np.allclose(history_centers[-1], history_centers[-2])
        reached_max_iterations = max_iter is not None and i >= max_iter
        
        if converged or reached_max_iterations:
            return history_centers, history_error, i


