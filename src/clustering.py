from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

def run_clustering_analysis(data_scaled):
    """
    Performs K-Means, Hierarchical, and DBSCAN clustering on the processed data.
    """
    return {
        'KMeans': KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(data_scaled),
        'Hierarchical': AgglomerativeClustering(n_clusters=2).fit_predict(data_scaled),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5).fit_predict(data_scaled)
    }