from builddata import generate_data, visualize_data


dim = 100

X, y = generate_data(dim=dim, distribution='gaussian', n_samples=100)
visualize_data(X, y)

