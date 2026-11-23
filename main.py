from builddata import generate_data
from kmeans import kmeans , minibatch_kmeans
from visualize import visualize_kmeans, visualize_data

dim = 3

X, y = generate_data(dim=dim, distribution='gaussian', n_samples=10000)
visualize_data(X, y)

history_centers , history_errors , iter =  minibatch_kmeans(X,3)

visualize_kmeans(history_errors)
visualize_data(X,centers=history_centers[-1])

