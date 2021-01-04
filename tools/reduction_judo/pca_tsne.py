import os
import sys
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.getcwd())

def main():
    files = os.listdir(os.path.join('data', 'JUDO', 'UNTRIMMED', 'i3d_224x224_chunk9'))
    X = None
    y = None
    for filename in files[::10]:
        feat_vects = np.load(os.path.join('data', 'JUDO', 'UNTRIMMED', 'i3d_224x224_chunk9', filename))
        targets = np.load(os.path.join('data', 'JUDO', 'UNTRIMMED', '4s_target_frames_25fps', filename))
        if X is None:
            X = feat_vects
            y = targets
        else:
            X = np.concatenate((X, feat_vects), axis=0)
            y = np.concatenate((y, targets), axis=0)
    y = y.argmax(axis=1)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    pca = PCA(n_components=512)
    pca.fit(X)
    X = pca.transform(X)

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    X = tsne.fit_transform(X)

    plt.figure(figsize=(16, 10))
    sns.scatterplot(X[:, 0], X[:, 1], hue=y, legend='full', palette=sns.color_palette("bright", 5))
    plt.savefig('tsne.png')

if __name__ == '__main__':
    main()