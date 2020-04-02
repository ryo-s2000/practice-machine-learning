from main import mglearn, train_test_split, plt, np
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
# fig, axes = plt.subplots(15, 2, figsize=(10, 20))
# malignant = cancer['data'][cancer['target'] == 0]
# benign = cancer['data'][cancer['target'] == 1]

# ax = axes.ravel()

# for i in range(30):
#     _, bins = np.histogram(cancer['data'][:, i], bins=50)
#     ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
#     ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)
#     ax[i].set_title(cancer['feature_names'][i])
#     ax[i].set_yticks(())
# ax[0].set_xlabel("Feature magnitude")
# ax[0].set_ylabel("Fr")

# ax[0].legend(['mailnant', 'begin'], loc='best')
# fig.tight_layout()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(cancer['data'])
X_scaled = scaler.transform(cancer['data'])

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
print("{}".format(str(X_scaled.shape)))
print("{}".format(str(X_pca.shape)))

plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer['target'])
plt.legend(cancer['target_names'], loc='best')
plt.gca().set_aspect('equal')
plt.xlabel("First")
plt.ylabel("Seccond")

plt.show()
