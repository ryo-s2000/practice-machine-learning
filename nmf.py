from main import mglearn, plt, np
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA

S = mglearn.datasets.make_signals()
# plt.figure(figsize=(6,1))
# plt.plot(S, '-')
# plt.xlabel("Time")
# plt.ylabel("Signal")

A = np.random.RandomState(0).uniform(size=(100, 3))
X = np.dot(S, A.T)
print("{}".format(X.shape))

nmf = NMF(n_components=3, random_state=42)
S_ = nmf.fit_transform(X)
print("{}".format(S_.shape))

pca = PCA(n_components=3)
H = pca.fit_transform(X)

models = [X, S, S_, H]
names = ['X', 'S', "S_", "H"]
fix, axes = plt.subplots(4, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})

for model, name, ax in zip(models, names, axes):
    ax.set_title(name)
    ax.plot(model[:, :3], '-')
plt.show()
