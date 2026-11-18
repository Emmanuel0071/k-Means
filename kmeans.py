import numpy as np
import random

def initialize_centers(X,k):
    list_centers = []
    for i in range (k):
        list_centers.append(X[random.randint(0,len(X)-1)])
    return np.array(list_centers)

def  sse_distance(point,centers):
    return np.sum((point-centers)**2)

def find_closest_center(point,centers):
    index_closest = 0
    min_sse = float('inf')

    for i in range(len(centers)):
        sse = sse_distance(point,centers[i])
        if sse < min_sse:
            min_sse = sse
            index_closest = i
    return index_closest

def compute_clusters(X,centers):
    cluster = []
    for i in range(len(centers)):
        index_clostest = find_closest_center(X[i],centers)
        cluster.append(index_clostest)
    return cluster

def sse_error(X, centers):
    error = 0 
    cluster = compute_clusters(X, centers)
    for i in range(len(X)):
        error += sse_distance(X[i], centers[cluster[i]])
    return error

def kmeans(X,k,max_iter=None):
    history_centers = []
    history_error = []
    centers = initialize_centers(X,k)
    i = 0
    while True:
        cluster = np.array(compute_clusters(X,centers))

        for j in range(k):
            points_in_cluster = X[cluster == j]

            if len(points_in_cluster) > 0:
                centers[j] = np.mean(points_in_cluster,axis=0)
        history_centers.append(centers.copy())
        history_error.append(sse_error(X,centers,cluster))