from main import mglearn, train_test_split, plt, np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.datasets import load_digits
digits = load_digits()
# データセット
# fig, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={"xticks":(), 'yticks':()})
# for ax, img in zip(axes.ravel(), digits['images']):
#     ax.imshow(img)

colors = ['#476A2A','#7851B8','#BD3430','#4A2D4E','#875525','#A83683','#4E655E','#853541','#3A3120','#535D8E']
plt.figure(figsize=(10, 10))

# tsne
tsne = TSNE(random_state=42)
digits_tsne = tsne.fit_transform(digits['data'])
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max())
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max())
for i in range(len(digits['data'])):
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits['target'][i]), color = colors[digits['target'][i]], fontdict={"weight": 'bold', 'size': 9})

# pca
# pca = PCA(n_components=2)
# pca.fit(digits['data'])
# digits_pca = pca.transform(digits['data'])
# plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
# plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
# for i in range(len(digits['data'])):
#     plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits['target'][i]), color = colors[digits['target'][i]], fontdict={"weight": 'bold', 'size': 9})


plt.xlabel("First")
plt.ylabel("Second")
plt.show()

