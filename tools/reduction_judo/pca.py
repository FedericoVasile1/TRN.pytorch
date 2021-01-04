import os
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

def main():
    files = os.listdir(os.path.join('data', 'JUDO', 'UNTRIMMED', 'i3d_224x224_chunk9'))
    X = None
    for filename in files[::2]:
        feat_vects = np.load(os.path.join('data', 'JUDO', 'UNTRIMMED', 'i3d_224x224_chunk9', filename))
        if X is None:
            X = feat_vects
        else:
            X = np.concatenate((X, feat_vects), axis=0)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    pca = PCA(n_components=1024)
    pca.fit(X)
    X = pca.transform(X)
    # print(pca.explained_variance_ratio_ * 100)
    # print(np.cumsum(pca.explained_variance_ratio_ * 100))

    plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100))
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance')
    plt.savefig('components.png')


if __name__ == '__main__':
    main()