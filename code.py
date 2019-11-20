from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def kmeans(x):
    data = pd.DataFrame(x.data)  # convert to pandas dataframe
    data.columns = x.feature_names  # set column names to feature names

    distance = []

    for k in range(1, 10):
        model = KMeans(n_clusters=k).fit(data)
        model.fit(data)
        distance.append(sum(np.min(cdist(data, model.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

    plt.plot(range(1, 10), distance, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distance')
    plt.title('The Elbow Method')
    plt.savefig('Clusters.pdf')
    plt.show()

if __name__ == '__main__':
    kmeans(load_iris())
