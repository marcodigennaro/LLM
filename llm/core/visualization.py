from sklearn.manifold import MDS, Isomap, TSNE
import matplotlib.pyplot as plt

import numpy as np


def map_labels_to_colors(labels):
    # Define a color for each category
    color_map = {
        'Noun': 'blue',
        'Verb': 'green',
        'Adjective': 'red',
        'Adverb': 'purple',
        'Other': 'gray'
    }
    # Map each label to a color
    return [color_map[label] for label in labels]


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


def plot_embeddings(embeddings_list, words, titles, colors):
    """
    Plots the results of dimensionality reductions.

    Args:
        embeddings_list (list): A list of embeddings from different reductions.
        words (list): List of words corresponding to embeddings.
        titles (list): Titles for each subplot.
        colors (list or str): Colors or a colormap for the points.
        :param annotate:
    """
    plt.figure(figsize=(18, 6))
    for i, (embed, title) in enumerate(zip(embeddings_list, titles)):
        plt.subplot(1, 3, i + 1)
        plt.scatter(embed[:, 0], embed[:, 1], c=colors, cmap='RdYlGn')

        for j, (w,c) in enumerate(zip(words, colors)):
            if abs(c) > 0.1:
                plt.annotate(w, (embed[j, 0], embed[j, 1]), fontsize=10)
        plt.title(title)
        plt.xlabel('z1')
        plt.ylabel('z2')
    plt.tight_layout()
    plt.show()
