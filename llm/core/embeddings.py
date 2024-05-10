from sklearn.manifold import MDS, Isomap, TSNE
from sklearn.decomposition import PCA


def apply_pca(embeddings):
    """Applies Principal Component Analysis (PCA) to the embeddings."""
    # Initialize PCA with 2 components
    pca = PCA(n_components=2)
    # Fit PCA on the embeddings and transform the data
    pca_result = pca.fit_transform(embeddings)
    return pca_result


def apply_mds(embeddings):
    """Applies Multidimensional Scaling (MDS) to the embeddings."""
    mds = MDS(n_components=2, random_state=42)
    return mds.fit_transform(embeddings)


def apply_isomap(embeddings, n_neighbors=10):
    """Applies Isomap to the embeddings."""
    n_neighbors_isomap = min(10, len(embeddings) - 1)  # Ensure n_neighbors is less than n_samples
    if n_neighbors_isomap > 1:  # Isomap needs at least 2 points
        isomap = Isomap(n_components=2, n_neighbors=n_neighbors_isomap)
        embeddings_isomap = isomap.fit_transform(embeddings)
    else:
        embeddings_isomap = embeddings  # Fallback to original embeddings if too few points
    return embeddings_isomap


def apply_tsne(embeddings, perplexity=False):
    """Applies t-SNE to the embeddings."""
    if not perplexity:
        perplexity = min(5, len(embeddings) - 1)  # Set minimum permissible perplexity

    if perplexity > 1:  # t-SNE needs at least a perplexity of 2 to make sense
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_tsne = tsne.fit_transform(embeddings)
    else:
        embeddings_tsne = embeddings  # Fallback to original embeddings if not enough points

    return embeddings_tsne
