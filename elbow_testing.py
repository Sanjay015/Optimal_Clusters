"""Find optimal number of clustres from a Dataset."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans
from sklearn.datasets import load_iris
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist


def load_dataset():
    """Load dataset."""
    # Loading dataset.
    return load_iris().data


def find_clusters(dataset):
    """Function to find optimal number of clusters in dataset."""
    # cluster data into K=1..10 clusters
    num_clusters = range(1, 50)
    k_means = [kmeans(dataset, k) for k in num_clusters]
    # cluster's centroids
    centroids = [cent for (cent, var) in k_means]
    clusters_dist = [cdist(dataset, cent, 'euclidean') for cent in centroids]
    cidx = [np.argmin(_dist, axis=1) for _dist in clusters_dist]
    dist = [np.min(_dist, axis=1) for _dist in clusters_dist]
    # Mean within-cluster (sum of squares)
    avg_within_sum_sqrd = [sum(d) / dataset.shape[0] for d in dist]
    return {'cidx': cidx, 'avg_within_sum_sqrd': avg_within_sum_sqrd,
            'K': num_clusters}


def plot_elbow_curv(details):
    """Function to plot elbo curv."""
    kidx = 2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(details['K'], details['avg_within_sum_sqrd'], 'b*-')
    ax.plot(details['K'][kidx], details['avg_within_sum_sqrd'][kidx],
            marker='o', markersize=12, markeredgewidth=2,
            markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')


def scatter_plot(dataset, details):
    """Function to plot scatter plot of clusters."""
    kidx = 2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    clr = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(details['K'][kidx]):
        ind = (details['cidx'][kidx] == i)
        ax.scatter(dataset[ind, 2], dataset[ind, 1],
                   s=30, c=clr[i], label='Cluster %d' % i)
    plt.xlabel('Petal Length')
    plt.ylabel('Sepal Width')
    plt.title('Iris Dataset, KMeans clustering with K=%d' % details['K'][kidx])
    plt.legend()
    plt.show()


def eblow(n):
    """Elbow testing."""
    cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    df = pd.read_csv('data.csv', usecols=cols).values
    kmeans_var = [KMeans(n_clusters=k).fit(df) for k in range(1, n)]
    centroids = [x.cluster_centers_ for x in kmeans_var]
    k_euclid = [cdist(df, cent) for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(df)**2) / df.shape[0]
    bss = tss - wcss
    plt.plot(bss)
    plt.show()

dataset = load_dataset()
details = find_clusters(dataset)
plot_elbow_curv(details)
scatter_plot(dataset, details)
