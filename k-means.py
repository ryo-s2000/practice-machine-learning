from main import mglearn, train_test_split, plt, np

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# mglearn.plots.plot_kmeans_algorithm()
# mglearn.plots.plot_kmeans_boundaries()

X, y = make_blobs(random_state=1)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print("{}".format(kmeans.labels_))

mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers="o")
mglearn.discrete_scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2], markers='^', markeredgewidth=2
)

plt.show()
