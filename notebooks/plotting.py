import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_filters(filters, title="Filters"):
    """Plot filters as separate channels
    filters has shape (out_channel, in_channel, kernel_height, kernel_width)
    """
    # filters = filters.squeeze(1)
    fig, axs = plt.subplots(
        filters.size(1), filters.size(0), sharey=True, sharex=True, squeeze=False
    )
    # print(axs)

    for i, row in enumerate(axs):
        for ax, weights in zip(row, filters[:, i, ...]):
            # ax.set_yticks(range(weights.size(-1)))
            ax.axis("off")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            cmap = ["Reds", "Greens", "Blues"][i]
            ax.imshow(weights, cmap=cmap)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.title(title)
    plt.show()


def plot_rgb_filters(filters, max_items=25, title="", fig=None, width=12):
    MAX_PER_ROW = 5

    n_items = min(len(filters), max_items)
    n_rows = int(np.ceil(n_items / MAX_PER_ROW))
    n_cols = min(MAX_PER_ROW, n_items)

    img_width = width / n_cols

    # fig, axs = plt.subplots(n_rows, n_cols, sharey=True, sharex=True, squeeze=True)
    if fig is None:
        fig = plt.figure(figsize=(width, n_rows * img_width))

    all_min = torch.min(filters).numpy()
    all_max = torch.max(filters).numpy()

    for i, weights in enumerate(filters[:n_items]):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        img = torch.moveaxis(weights, 0, -1).numpy()
        img -= np.min(img)  # Normalize
        img /= np.max(img)

        ax.imshow(img)

    plt.title(title)

    plt.show()
