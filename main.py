from builddata import generate_data, visualize_data


dim = 3

X, y = generate_data(dim=dim, distribution='class', n_samples=100)
visualize_data(X, y)

