import random
import numpy as np


def has_converged(old_mu, new_mu):
    return np.alltrue(old_mu == new_mu)


def assign_clusters(X, mu):
    center_indices = [np.linalg.norm(mu - x, axis=-1).argmin() for x in X]
    clusters = [[] for _ in range(len(mu))]
    for i, ind in enumerate(center_indices):
        clusters[ind].append(X[i])
    return clusters


def reevaluate_centers(clusters):
    return np.array([np.array(c).mean(axis=0) for c in clusters])


def find_centers(X, k):
    old_mu = np.array(random.sample(X, k))
    new_mu = np.array(random.sample(X, k))
    clusters = assign_clusters(X, old_mu)
    while not has_converged(old_mu, new_mu):
        clusters = assign_clusters(X, new_mu)
        old_mu = new_mu
        new_mu = reevaluate_centers(clusters)

    return new_mu, clusters


def cost(mus, clusters):
    return sum(((np.array(cluster) - mu) ** 2).sum() for mu, cluster in
               zip(mus, clusters))


def k_means(X, k, iters=1):
    best_mus, best_clusters = find_centers(X, k)
    best_cost = cost(best_mus, best_clusters)
    for _ in xrange(iters - 1):
        curr_mus, curr_clusters = find_centers(X, k)
        curr_cost = cost(curr_mus, curr_clusters)
        if curr_cost < best_cost:
            best_mus, best_clusters = curr_mus, curr_clusters
    return best_mus, best_clusters
