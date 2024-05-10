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


def rescale_data(data):
    """
    Rescales the given data to a new range (-1, 1) using NumPy operations for efficiency.

    Args:
        data (ndarray): The original data to be rescaled.

    Returns:
        ndarray: Rescaled data.
    """
    # Use NumPy's ptp (peak-to-peak, i.e., max - min) function for more efficient range calculation
    return -1 + 2 * (data - np.min(data)) / np.ptp(data)


def plot_embeddings(embeddings_list, words, titles, colors):
    """
    Plots a series of scatter plots for visualizing word embeddings, each corresponding
    to different dimensionality reductions, with annotations based on certain conditions.

    Args:
        embeddings_list (list of ndarray): List of 2D arrays where each array represents
                                           word embeddings reduced to two dimensions.
        words (list of str): List of words corresponding to the points in embeddings.
        titles (list of str): Titles for each subplot to describe the dimensionality reduction method used.
        colors (list of str or ndarray): Colors or a colormap for the points, used for marking sentiment or categories.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plotted subplots.
    """

    fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)

    for i, (embed, title) in enumerate(zip(embeddings_list, titles)):
        rr, cc = i // 2, i % 2
        # Rescale the embeddings data before plotting
        scaled_embed = rescale_data(embed)

        ax[rr, cc].scatter(scaled_embed[:, 0], scaled_embed[:, 1], c=colors, cmap='RdYlGn')
        for j, (w, c) in enumerate(zip(words, colors)):
            if abs(c) > 0.1:
                ax[rr, cc].annotate(w, (scaled_embed[j, 0], scaled_embed[j, 1]), fontsize=10)

        ax[rr, cc].annotate(
            title, xy=(0.1, 0.9), xycoords='axes fraction',
            fontsize=12, fontweight='bold',
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))

        if rr == 1:
            ax[rr, cc].set_xlabel('z1', fontsize=14)
        if cc == 0:
            ax[rr, cc].set_ylabel('z2', fontsize=14)

    plt.tight_layout()

    return fig
