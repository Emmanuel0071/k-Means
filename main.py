from builddata import generate_data
from kmeans import kmeans
from visualize import visualize_kmeans, visualize_data , visualize_clusters

dim = 3

X, y = generate_data(dim=dim, distribution='gaussian', n_samples=100)
visualize_data(X, y)

history_centers , history_errors , iter =  kmeans(X,3,1)

visualize_kmeans(history_errors)

